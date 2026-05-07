# disparity_estimator.py
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ORIG_H, ORIG_W = 1024, 1280

# Repo roots on the cluster.
FOUNDATIONSTEREO_REPO = '/home/jhan3/endo_stereo/FoundationStereo'
FAST_FOUNDATIONSTEREO_REPO = '/home/jhan3/endo_stereo/Fast-FoundationStereo'
WAFT_STEREO_REPO = '/home/jhan3/endo_stereo/WAFT-Stereo'

# WAFT-Stereo SynLarge variants: (config rel. path, ckpt rel. path) under the repo root.
WAFT_VARIANTS = {
    'S': ('configs/SynLarge/DAv2S-4.yaml', 'ckpts/SynLarge/DAv2S-4.pth'),
    'B': ('configs/SynLarge/DAv2B-4.yaml', 'ckpts/SynLarge/DAv2B-4.pth'),
    'L': ('configs/SynLarge/DAv2L-5.yaml', 'ckpts/SynLarge/DAv2L-5.pth'),
}

SUPPORTED_MODELS = ('foundationstereo', 'fast-foundationstereo', 'waft-stereo')


class StereoDisparityEstimator:
    """
    Unified wrapper that dispatches between FoundationStereo, Fast-FoundationStereo,
    and WAFT-Stereo for batched disparity estimation on DV5/SCARED-style stereo
    images ([B, H, W, 3] float tensors in [0, 255] on CUDA).
    """

    def __init__(
        self,
        model_name: str = 'foundationstereo',
        # (Fast-)FoundationStereo
        fs_ckpt_dir: str = "/home/jhan3/endo_stereo/FoundationStereo/pretrained_models/23-51-11",
        # WAFT-Stereo
        waft_variant: str = 'L',
        waft_repo_dir: str = WAFT_STEREO_REPO,
        # Shared
        scale: float = 1.0,
        max_batch_size: int = 1,
    ):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f'invalid model name {model_name!r}. Must be one of {SUPPORTED_MODELS}'
            )
        self.model_name = model_name
        self._scale = scale

        if self.model_name in ('foundationstereo', 'fast-foundationstereo'):
            self._ckpt_dir = Path(fs_ckpt_dir)
            self._fs_cfg = self._load_fs_cfg()
            self.model = self._prepare_fs_model()

            img_input_shape = (
                max_batch_size, 3,
                int(ORIG_H * self._scale),
                int(ORIG_W * self._scale),
            )
            # _InputPadder is captured from sys.path in _prepare_fs_model.
            self.padder = self._InputPadder(img_input_shape, divis_by=32, force_square=False)

        else:  # waft-stereo
            if waft_variant not in WAFT_VARIANTS:
                raise ValueError(
                    f'invalid WAFT variant {waft_variant!r}. '
                    f'Must be one of {list(WAFT_VARIANTS)}'
                )
            self._waft_variant = waft_variant
            self._waft_repo_dir = Path(waft_repo_dir)
            self.model, self._waft_cfg = self._prepare_waft_model()
            self.padder = None

    # ============================================================================
    #  FoundationStereo / Fast-FoundationStereo
    # ============================================================================
    def _load_fs_cfg(self):
        fs_cfg = OmegaConf.load(str(self._ckpt_dir / "cfg.yaml"))
        if "vit_size" not in fs_cfg:
            fs_cfg["vit_size"] = "vitl"
        fs_cfg["scale"] = 1.0
        fs_cfg["hiera"] = 0.0
        fs_cfg["z_far"] = 1000
        fs_cfg["valid_iters"] = 32
        fs_cfg["get_pc"] = 0
        fs_cfg["remove_invisible"] = 1
        fs_cfg["denoise_cloud"] = 1
        fs_cfg["denoise_nb_points"] = 30
        fs_cfg["denoise_radius"] = 0.03
        return fs_cfg

    def _prepare_fs_model(self):
        if self.model_name == 'foundationstereo':
            sys.path.insert(0, FOUNDATIONSTEREO_REPO)
            from core.utils.utils import InputPadder
            from core.foundation_stereo import FoundationStereo
            self._InputPadder = InputPadder

            m = FoundationStereo(self._fs_cfg)
            ckpt_path = self._ckpt_dir / "model_best_bp2.pth"
            if ckpt_path.exists():
                ckpt = torch.load(str(ckpt_path), map_location="cpu")
                m.load_state_dict(ckpt["model"])
            else:
                print("[WARNING] FoundationStereo checkpoint not found; skipping load")
        else:  # fast-foundationstereo
            sys.path.append(FAST_FOUNDATIONSTEREO_REPO)
            from core.utils.utils import InputPadder
            self._InputPadder = InputPadder

            ckpt_path = self._ckpt_dir / 'model_best_bp2_serialize.pth'
            m = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            m.args.valid_iters = 8
            m.args.max_disp = ORIG_W // 4
        m.cuda()
        m.eval()
        return m

    # ============================================================================
    #  WAFT-Stereo
    # ============================================================================
    def _prepare_waft_model(self):
        sys.path.insert(0, str(self._waft_repo_dir))
        from bridgedepth.config import get_cfg
        from algorithms.waft import WAFT

        cfg_rel, ckpt_rel = WAFT_VARIANTS[self._waft_variant]
        cfg_path = self._waft_repo_dir / cfg_rel
        ckpt_path = self._waft_repo_dir / ckpt_rel
        if not cfg_path.exists():
            raise FileNotFoundError(f'[WAFT-Stereo] config not found: {cfg_path}')
        if not ckpt_path.exists():
            raise FileNotFoundError(f'[WAFT-Stereo] checkpoint not found: {ckpt_path}')

        cfg = get_cfg()
        cfg.merge_from_file(str(cfg_path))
        cfg.freeze()

        m = WAFT(cfg).cuda().eval()

        print(f'[WAFT-Stereo] loading variant={self._waft_variant} from {ckpt_path}')
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        weights = ckpt['model'] if 'model' in ckpt else ckpt
        missing, unexpected = m.load_state_dict(weights, strict=False)
        if missing:
            print(f'[WAFT-Stereo] {len(missing)} missing keys '
                  f'(first 5: {list(missing)[:5]})')
        if unexpected:
            print(f'[WAFT-Stereo] {len(unexpected)} unexpected keys '
                  f'(first 5: {list(unexpected)[:5]})')
        return m, cfg

    # ============================================================================
    #  Inference
    # ============================================================================
    @torch.no_grad()
    def estimate_batch(self, img0, img1: torch.Tensor) -> dict:
        """
        Args:
            img0, img1: [B, H, W, 3] float tensors in [0, 255] on CUDA.
                        H=1024, W=1280 (original SCARED rectified resolution).
        Returns:
            {'rgb': [B, 3, H', W'], 'disp': [B, H', W']} at --scale resolution.
        """
        if self.model_name in ('foundationstereo', 'fast-foundationstereo'):
            return self._estimate_fs(img0, img1)
        return self._estimate_waft(img0, img1)

    def _maybe_rescale(self, x):
        if self._scale == 1.0:
            return x
        return F.interpolate(
            x, scale_factor=self._scale, mode="bicubic", align_corners=False
        ).clamp(0, 255)

    def _estimate_fs(self, img0, img1):
        # [B, H, W, 3] -> [B, 3, H, W]
        img0 = img0.permute(0, 3, 1, 2)
        img1 = img1.permute(0, 3, 1, 2)

        img0 = self._maybe_rescale(img0)
        img1 = self._maybe_rescale(img1)

        img0_p, img1_p = self.padder.pad(img0, img1)
        if self.model_name == 'foundationstereo':
            with torch.autocast(device_type="cuda"):
                if not self._fs_cfg["hiera"]:
                    pred_p = self.model.forward(
                        img0_p, img1_p,
                        iters=self._fs_cfg["valid_iters"],
                        test_mode=True,
                    )
                else:
                    pred_p = self.model.run_hierarchical(
                        img0_p, img1_p,
                        iters=self._fs_cfg["valid_iters"],
                        test_mode=True,
                        small_ratio=0.5,
                    )
        else:  # fast-foundationstereo
            with torch.autocast(device_type='cuda'):
                if not self._fs_cfg["hiera"]:
                    pred_p = self.model.forward(
                        img0_p, img1_p,
                        iters=8, test_mode=True,
                        optimize_build_volume='pytorch1',
                    )
                else:
                    pred_p = self.model.run_hierachical(
                        img0_p, img1_p,
                        iters=8, test_mode=True,
                        small_ratio=0.5,
                    )
        pred = self.padder.unpad(pred_p.float())
        pred = pred.squeeze(dim=1)  # [B, H', W']
        return {"rgb": img0, "disp": pred}

    def _estimate_waft(self, img0, img1):
        """
        WAFT-Stereo branch. model.inference() handles its own internal padding,
        so we just hand it [B, 3, H, W] tensors in [0, 255] (matching RAFT-/WAFT-
        style conventions inherited from BridgeDepth).
        """
        img0 = img0.permute(0, 3, 1, 2).contiguous()
        img1 = img1.permute(0, 3, 1, 2).contiguous()

        img0 = self._maybe_rescale(img0)
        img1 = self._maybe_rescale(img1)

        sample = {"img1": img0, "img2": img1}
        results = self.model.inference(sample, size=None)
        pred = results['disp_pred']
        # Defensive: collapse any singleton channel dim if present.
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        return {"rgb": img0, "disp": pred.float()}