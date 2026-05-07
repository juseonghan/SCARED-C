# dataset.py 
import torch 
import glob 
from pathlib import Path 
from PIL import Image 
import numpy as np
import cv2
import json 

import decord 
decord.bridge.set_bridge('native')

from typing import Union, Iterator, Tuple 
import tarfile 
import re 
import tifffile as tiff 
from io import BytesIO

class TarLoader:
    """ Extract SCARED ground truth .tiff sequence stored in .tar one by one."""

    def __init__(self, tar_filepath: Union[Path, str]) -> None:
        tar_p = Path(tar_filepath)
        assert tar_p.is_file()
        self.tardata = tarfile.open(tar_p, "r:gz")
        self.tarnames = {
            int(re.sub(r"\D", "", i.name)): i for i in self.tardata.getmembers()
        }
        self.num_frames = len(self.tarnames)

    def __getitem__(self, key: int) -> np.ndarray:
        assert isinstance(key, int)
        # assert key < self.num_frames
        data = self.tardata.extractfile(self.tarnames[key]).read()
        img = tiff.imread(BytesIO(data))
        img[img[:, :, 2] == 0] = np.nan
        return img

    def __iter__(self) -> Iterator[np.ndarray]:
        self.idx = 0
        return self

    def __next__(self) -> np.ndarray:
        if self.idx < self.num_frames:
            x = self.idx
            self.idx += 1
            return self.__getitem__(x)
        else:
            raise StopIteration

    def __len__(self) -> int:
        return self.num_frames

    def __exit__(self) -> None:
        self.tardata.close()
        del self.tardata
        del self.tarnames
        del self.num_frames


class OriginalSCAREDDataset(torch.utils.data.Dataset):

    def __init__(self, data_root='/nfs/home/jhan3/scared_data/datasets'):
        data_root = Path(data_root)
        self.left_imgs = sorted(glob.glob(str(data_root / 'dataset_*/keyframe_*/data/left_rectified/*')))
        self.right_imgs = sorted(glob.glob(str(data_root / 'dataset_*/keyframe_*/data/right_rectified/*')))
        self.gt_disp = sorted(glob.glob(str(data_root / 'dataset_*/keyframe_*/data/disparity/*')))
        self.gt_depth = sorted(glob.glob(str(data_root / 'dataset_*/keyframe_*/data/depthmap_rectified/*')))

        assert len(self.left_imgs) == len(self.right_imgs)
        assert len(self.right_imgs) == len(self.gt_disp)
        assert len(self.gt_disp) == len(self.gt_depth)

    def __len__(self):
        return len(self.left_imgs)
    
    def _get_fname_code(self, p):
        dataset_num_idx = p.index('dataset_')
        dataset_num = p[dataset_num_idx+8]
        keyframe_num = p[dataset_num_idx+19]
        img_num = Path(p).stem
        return f'{dataset_num}_{keyframe_num}_{img_num}'

    def _get_calib_path(self, p):
        p = Path(p)
        return str(p.parent.parent.parent / 'stereo_calib.json')

    def __getitem__(self, idx):
        path_L = self.left_imgs[idx]
        path_R = self.right_imgs[idx]
        path_gt = self.gt_disp[idx]
        path_depth = self.gt_depth[idx]
        calib_path = self._get_calib_path(path_L)

        imgL = np.array(Image.open(path_L))
        imgR = np.array(Image.open(path_R))
        gt = cv2.imread(path_gt, cv2.IMREAD_UNCHANGED) / 256.0 
        gt_depth = cv2.imread(path_depth, cv2.IMREAD_UNCHANGED) / 256.0 
        fname_code = self._get_fname_code(path_L)
        with open(calib_path, 'r') as f:
            _calib = json.load(f)

        return {
            'img0': imgL, 
            'img1': imgR, 
            'gt': gt, 
            'depth': gt_depth, 
            'fname': fname_code, 
            'fx': _calib['K1']['data'][0],
            'baseline': -_calib['T']['data'][0],
        }


class CorrectedSCAREDDataset:

    def __init__(self, data_root='/nfs/home/jhan3/scared_data/corrected'):
        data_root = Path(data_root)
        self.gt_tar_paths = sorted(glob.glob(
            str(data_root / 'dataset_*/keyframe_*/data/scene_points.tar.gz')
        ))
        # 7_4 doesn't have 'R1' key inside stereo_calib.json
        self.gt_tar_paths = [p for p in self.gt_tar_paths if 'dataset_7/keyframe_4' not in p]

    def _find_img_path(self, p):
        p_split = p.split('/')
        p_split[5] = 'datasets'
        img_dir = '/'.join(p_split[:-1])
        imgL_dir = Path(img_dir) / 'left_rectified'
        imgR_dir = Path(img_dir) / 'right_rectified'
        return imgL_dir, imgR_dir 

    def __len__(self):
        return len(self.gt_tar_paths)
    
    def _find_calib_path(self, p):
        p_split = p.split('/')
        p_split[5] = 'datasets'
        p_fixed = '/'.join(p_split)
        p = Path(p_fixed).parent / '../stereo_calib.json'
        return str(p)
    
    def get_sequence_metadata(self, idx):
        """Return paths + calib without opening the tar file."""
        tar_path = self.gt_tar_paths[idx]
        imgL_dir, imgR_dir = self._find_img_path(tar_path)
        imgL_paths = sorted(glob.glob(str(imgL_dir / '*.png')))
        imgR_paths = sorted(glob.glob(str(imgR_dir / '*.png')))
        calib_path = self._find_calib_path(tar_path)
        with open(calib_path, 'r') as f:
            calib = json.load(f)
        return tar_path, imgL_paths, imgR_paths, calib
    
    def get_readers(self, idx):
        tar_path = self.gt_tar_paths[idx]
        tar_reader = TarLoader(tar_path)

        imgL_dir, imgR_dir = self._find_img_path(tar_path)
        imgL_paths = sorted(glob.glob(str(imgL_dir / '*.png')))
        imgR_paths = sorted(glob.glob(str(imgR_dir / '*.png')))
        
        calib_path = self._find_calib_path(tar_path)
        with open(calib_path, 'r') as f:
            calib = json.load(f)

        return imgL_paths, imgR_paths, tar_reader, calib


class SCAREDDatasetTar:

    def __init__(self, data_root='/nfs/home/jhan3/scared_data/corrected'):
        data_root = Path(data_root)
        self.gt_tar_paths = sorted(glob.glob(
            str(data_root / 'dataset_*/keyframe_*/data/scene_points.tar.gz')
        ))
        # 7_4 doesn't have 'R1' key inside stereo_calib.json
        self.gt_tar_paths = [p for p in self.gt_tar_paths if
                                'dataset_7/keyframe_4' not in p or
                                'dataset_4' not in p or 
                                'dataset_5' not in p or 
                                'test' not in p
                            ]

    def _find_video_path(self, p):
        """Video is colocated with scene_points.tar.gz in the data/ dir."""
        return str(Path(p).parent / 'rgb.mp4')

    def __len__(self):
        return len(self.gt_tar_paths)

    def _find_calib_path(self, p):
        p_split = p.split('/')
        p_split[5] = 'datasets'
        p_fixed = '/'.join(p_split)
        p = Path(p_fixed).parent / '../stereo_calib.json'
        return str(p)

    def get_sequence_metadata(self, idx):
        """Return paths + calib without opening the tar file."""
        tar_path = self.gt_tar_paths[idx]
        video_path = self._find_video_path(tar_path)
        calib_path = self._find_calib_path(tar_path)
        with open(calib_path, 'r') as f:
            calib = json.load(f)
        # Probe frame count without reading pixels
        vr = decord.VideoReader(video_path, ctx=decord.cpu())
        num_frames = len(vr)
        return tar_path, vr, num_frames, calib

    def get_readers(self, idx):
        tar_path = self.gt_tar_paths[idx]
        tar_reader = TarLoader(tar_path)

        video_path = self._find_video_path(tar_path)
        vr = decord.VideoReader(video_path, ctx=decord.cpu())

        calib_path = self._find_calib_path(tar_path)
        with open(calib_path, 'r') as f:
            calib = json.load(f)

        return vr, tar_reader, calib

"""
SCARED stereo depth evaluation dataset.

For each sample returns:
    - left_rgb:       (H, W, 3) uint8 ndarray  — from corrected rgb_frames.tar.gz
    - right_rgb:      (H, W, 3) uint8 ndarray  — from original rgb.mp4 (bottom half)
    - scene_points:   (H, W, 3) float32 ndarray — XYZ from scene_points.tar.gz
    - calib:          dict                       — stereo calibration (raw JSON)
    - frame_id:       int                        — 1-indexed frame number

The corrected tar only contains frames registered in COLMAP, so frame numbers
will have gaps. The original rgb.mp4 is indexed to retrieve the matching right
image for each registered left frame.

Usage:
    dataset = SCAREDStereoDataset(keys=["1_1", "1_2", "2_1"])
    left, right, scene_pts, calib, fid = dataset[0]
"""


CORRECTED_ROOT = Path("/nfs/home/jhan3/scared_data/corrected")
ORIGINAL_ROOT = Path("/nfs/home/jhan3/scared_data/original")
DATASETS_ROOT = Path("/nfs/home/jhan3/scared_data/datasets")


class _KeyframeSequence:
    """Lazy-loaded handles for a single keyframe's tar archives and video."""

    def __init__(self, dataset_id: str, keyframe_id: str):
        self.ds = dataset_id
        self.kf = keyframe_id

        corrected_kf = CORRECTED_ROOT / f"dataset_{dataset_id}" / f"keyframe_{keyframe_id}"
        original_kf = ORIGINAL_ROOT / f"dataset_{dataset_id}" / f"keyframe_{keyframe_id}"
        datasets_kf = DATASETS_ROOT / f"dataset_{dataset_id}" / f"keyframe_{keyframe_id}"

        rgb_tar_path = corrected_kf / "data" / "rgb_frames.tar.gz"
        sp_tar_path = corrected_kf / "data" / "scene_points.tar.gz"
        video_path = original_kf / "data" / "rgb.mp4"
        calib_path = datasets_kf / "stereo_calib.json"

        assert rgb_tar_path.exists(), f"Not found: {rgb_tar_path}"
        assert sp_tar_path.exists(), f"Not found: {sp_tar_path}"
        assert video_path.exists(), f"Not found: {video_path}"
        assert calib_path.exists(), f"Not found: {calib_path}"

        self.video_path = video_path

        # --- Load stereo calibration ---
        self.calib = self._load_stereo_calib(calib_path)

        # --- Open rgb_frames tar and index by frame number ---
        self._rgb_tar = tarfile.open(str(rgb_tar_path), "r:gz")
        self._rgb_members = {}  # frame_num -> TarInfo
        for member in self._rgb_tar.getmembers():
            if not member.isfile():
                continue
            fnum = self._parse_frame_number(member.name, prefix="frame")
            if fnum is not None:
                self._rgb_members[fnum] = member

        # --- Open scene_points tar and index by frame number ---
        self._sp_tar = tarfile.open(str(sp_tar_path), "r:gz")
        self._sp_members = {}  # frame_num -> TarInfo
        for member in self._sp_tar.getmembers():
            if not member.isfile():
                continue
            fnum = self._parse_frame_number(member.name, prefix="scene_points")
            if fnum is not None:
                self._sp_members[fnum] = member

        # --- Frame numbers present in BOTH tars ---
        common = sorted(set(self._rgb_members) & set(self._sp_members))
        assert len(common) > 0, (
            f"No common frames between rgb_frames and scene_points "
            f"for dataset_{dataset_id}/keyframe_{keyframe_id}"
        )
        self.frame_nums = common

        # Video capture opened lazily (not pickle-friendly otherwise)
        self._cap = None

    @staticmethod
    def _parse_frame_number(filename: str, prefix: str) -> int | None:
        """Extract 1-indexed frame number from e.g. 'frame000003.png'."""
        name = Path(filename).stem
        if not name.startswith(prefix):
            return None
        digits = name[len(prefix):]
        try:
            return int(digits)
        except ValueError:
            return None

    @staticmethod
    def _load_stereo_calib(calib_path: Path) -> dict:
        """Load stereo_calib.json and return the raw JSON dict."""
        with open(calib_path) as f:
            return json.load(f)

    def _ensure_video(self):
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.video_path))
            assert self._cap.isOpened(), f"Cannot open {self.video_path}"

    def get_left_rgb(self, frame_num: int) -> np.ndarray:
        """Read left RGB from the corrected tar. Returns (H, W, 3) uint8 RGB."""
        data = self._rgb_tar.extractfile(self._rgb_members[frame_num]).read()
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_right_rgb(self, frame_num: int) -> np.ndarray:
        """Read matching right frame from original rgb.mp4. Returns (H, W, 3) uint8 RGB."""
        self._ensure_video()
        # frame_num is 1-indexed; cv2 frame positions are 0-indexed
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = self._cap.read()
        assert ret, f"Failed to read frame {frame_num - 1} from {self.video_path}"
        # rgb.mp4 = left (top half) stacked on right (bottom half)
        h = frame.shape[0] // 2
        right_bgr = frame[h:, :]
        return cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    def get_scene_points(self, frame_num: int) -> np.ndarray:
        """Read scene points TIFF. Returns (H, W, 3) float32 XYZ. Invalid pixels are (0,0,0)."""
        data = self._sp_tar.extractfile(self._sp_members[frame_num]).read()
        img = tiff.imread(BytesIO(data)).astype(np.float32)  # (H, W, 3) XYZ
        assert img.ndim == 3 and img.shape[2] == 3, (
            f"Expected (H,W,3) scene points, got {img.shape}"
        )
        img[~np.isfinite(img)] = 0.0
        return img

    def close(self):
        self._rgb_tar.close()
        self._sp_tar.close()
        if self._cap is not None:
            self._cap.release()

    def __del__(self):
        self.close()


class SCAREDStereoDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for stereo depth evaluation on corrected SCARED data.

    Args:
        keys: List of "x_y" strings (e.g. ["1_1", "1_2", "2_1"]).
              Each maps to dataset_x/keyframe_y.

    Returns per __getitem__:
        left_rgb:      (H, W, 3) uint8 ndarray
        right_rgb:     (H, W, 3) uint8 ndarray
        scene_points:  (H, W, 3) float32 ndarray (XYZ in camera coords, (0,0,0) = invalid)
        calib:         dict — raw stereo_calib.json contents
        frame_id:      int (1-indexed frame number)
    """

    def __init__(self, keys: list[str]):
        self.sequences: list[_KeyframeSequence] = []
        # (sequence_idx, frame_num) for global indexing
        self.index_map: list[tuple[int, int]] = []

        for key in keys:
            parts = key.split("_")
            assert len(parts) == 2, f"Key must be x_y, got: {key}"
            ds, kf = parts
            seq = _KeyframeSequence(ds, kf)
            seq_idx = len(self.sequences)
            self.sequences.append(seq)
            for fnum in seq.frame_nums:
                self.index_map.append((seq_idx, fnum))

        print(
            f"[SCAREDStereoDataset] {len(keys)} keyframe(s), "
            f"{len(self.index_map)} total frames"
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, int]:
        seq_idx, frame_num = self.index_map[idx]
        seq = self.sequences[seq_idx]

        left_rgb = seq.get_left_rgb(frame_num)
        right_rgb = seq.get_right_rgb(frame_num)
        scene_points = seq.get_scene_points(frame_num)
        calib = seq.calib

        return left_rgb, right_rgb, scene_points, calib, frame_num

    def close(self):
        for seq in self.sequences:
            seq.close()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    # Quick sanity check
    ds = SCAREDStereoDataset(keys=["1_1"])
    print(f"Dataset length: {len(ds)}")
    left, right, scene_pts, calib, fid = ds[0]
    depth = scene_pts[..., 2]
    print(f"Frame {fid}: left={left.shape}, right={right.shape}, scene_pts={scene_pts.shape}")
    print(f"Depth range: [{depth[depth > 0].min():.1f}, {depth.max():.1f}]")
    print(f"Baseline: {-calib['T']['data'][0]:.2f} mm")
    ds.close()