import argparse 
from pathlib import Path 
import glob 
import cv2 
import numpy as np
from PIL import Image 
import tqdm 
import json 
import time

import torch 
import torch.nn.functional as F 

from disparity_estimator import StereoDisparityEstimator

class SCARED_Keyframes(torch.utils.data.Dataset):

    def __init__(self, data_root='/nfs/home/jhan3/scared_data/keyframes', scale_factor=256.0):
        data_root = Path(data_root) if not isinstance(data_root, Path) else data_root 
        self.paths = sorted(glob.glob(str(data_root / 'dataset_*/keyframe_*')))
        self.paths = [p for p in self.paths if 'dataset_4' not in p and 'dataset_5' not in p]
        print(f'Number of keyframes: {len(self.paths)}')
        self.scale_factor = scale_factor 
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        keyframe_dir = self.paths[idx]

        # RGB images
        img0_path = Path(keyframe_dir) / 'left_rectified.png'
        img0 = np.array(Image.open(str(img0_path)))
        img1_path = Path(keyframe_dir) / 'right_rectified.png'
        img1 = np.array(Image.open(str(img1_path)))

        # disparity
        disp_path = Path(keyframe_dir) / 'disparity.png'
        disp = np.array(Image.open(disp_path)) / self.scale_factor

        # depth
        depth_path = Path(keyframe_dir) / 'depthmap_rectified.png'
        depth = np.array(Image.open(depth_path)) / self.scale_factor

        # calibration
        calib_path = Path(keyframe_dir) / 'stereo_calib.json'
        with open(str(calib_path), 'r') as f:
            _calib = json.load(f)

        return {
            'img0': img0, 
            'img1': img1, 
            'disp': disp,
            'depth': depth,
            'path': keyframe_dir,
            'fx': _calib['K1']['data'][0],
            'baseline': -_calib['T']['data'][0],
        }


def build_valid_mask(gt_full, scale):
    """Downsample a ground truth map and build a conservative valid mask 
    using nearest-neighbor interpolation to avoid boundary blending artifacts."""
    gt_tensor = torch.from_numpy(gt_full).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    valid_orig = (gt_tensor > 0).float()
    valid_down = F.interpolate(valid_orig, scale_factor=scale, mode='nearest')
    gt_down = F.interpolate(gt_tensor, scale_factor=scale, mode='bilinear', align_corners=False)
    gt_np = gt_down.squeeze().numpy().astype(np.float64)
    valid = (valid_down.squeeze().numpy() > 0) & (gt_np > 0)
    return gt_np, valid


def run_eval(stereo_model, dataloader, scale, output_dir):

    # ---- Disparity accumulators ----
    disp_accum = {
        "epe": 0.0, "rmse": 0.0,
        "bad_1": 0.0, "bad_2": 0.0, "bad_3": 0.0,
    }
    disp_total_valid = 0
    disp_all_abs_err = []

    # ---- Depth accumulators ----
    depth_accum = {
        "abs_rel": 0.0, "sq_rel": 0.0,
        "rmse": 0.0, "rmse_log": 0.0, "mae": 0.0,
        "delta_1": 0.0, "delta_2": 0.0, "delta_3": 0.0,
    }
    depth_total_valid = 0
    depth_all_abs_err = []

    total_samples = 0
    per_image = []

    pbar = tqdm.tqdm(dataloader, total=len(dataloader), desc='Evaluating')
    for sample in pbar:
        img0 = sample['img0'].cuda().float()
        img1 = sample['img1'].cuda().float()

        # ---- Inference (shared) ----
        with torch.no_grad():
            start_time = time.time()
            result = stereo_model.estimate_batch(img0, img1)
            inf_time = time.time() - start_time 
            tqdm.tqdm.write(f'Inference took {1000 * inf_time} ms')
        pred_disp = result["disp"].cpu().numpy().astype(np.float64)  # [B, H', W']

        for b in range(pred_disp.shape[0]):
            name = Path(sample['path'][b]).parts[-2] + "/" + Path(sample['path'][b]).parts[-1]
            fx = float(sample['fx'][b])
            baseline = float(sample['baseline'][b])
            fx_scaled = fx * scale
            img_results = {"name": name}

            # ================================================================
            #  DISPARITY METRICS
            # ================================================================
            gt_disp_full = sample['disp'][b].numpy().astype(np.float64)  # [H, W]
            gt_disp, disp_valid = build_valid_mask(gt_disp_full, scale)
            gt_disp = gt_disp * scale  # scale disparity values to match resolution

            p_disp = pred_disp[b]
            mask_d = disp_valid
            n_d = mask_d.sum()

            if n_d > 0:
                pd = p_disp[mask_d]
                gd = gt_disp[mask_d]
                abs_err_d = np.abs(pd - gd)
                sq_err_d = (pd - gd) ** 2

                disp_accum["epe"] += abs_err_d.sum()
                disp_accum["rmse"] += sq_err_d.sum()
                for thresh, key in [(1, "bad_1"), (2, "bad_2"), (3, "bad_3")]:
                    bad = abs_err_d > np.maximum(thresh, 0.05 * gd)
                    disp_accum[key] += bad.sum()

                disp_total_valid += n_d
                disp_all_abs_err.append(abs_err_d)

                mean_err_d = abs_err_d.mean()
                mean_sq_d = sq_err_d.mean()
                img_results["disp"] = {
                    "epe": float(mean_err_d),
                    "rmse": float(np.sqrt(mean_sq_d)),
                    "std": float(np.sqrt(mean_sq_d - mean_err_d ** 2)),
                    "bad_1": float((abs_err_d > np.maximum(1, 0.05 * gd)).mean()) * 100.0,
                    "bad_2": float((abs_err_d > np.maximum(2, 0.05 * gd)).mean()) * 100.0,
                    "bad_3": float((abs_err_d > np.maximum(3, 0.05 * gd)).mean()) * 100.0,
                    "n_valid": int(n_d),
                }

            # ================================================================
            #  DEPTH METRICS
            # ================================================================
            gt_depth_full = sample['depth'][b].numpy().astype(np.float64)  # [H, W]
            gt_depth, depth_valid = build_valid_mask(gt_depth_full, scale)

            pred_depth = np.where(p_disp > 0, (fx_scaled * baseline) / p_disp, 0.0)
            mask_z = depth_valid & (pred_depth > 0)
            n_z = mask_z.sum()

            if n_z > 0:
                pz = pred_depth[mask_z]
                gz = gt_depth[mask_z]
                abs_err_z = np.abs(pz - gz)
                sq_err_z = (pz - gz) ** 2
                ratio = np.maximum(pz / gz, gz / pz)

                depth_accum["abs_rel"] += (abs_err_z / gz).sum()
                depth_accum["sq_rel"] += (sq_err_z / gz).sum()
                depth_accum["rmse"] += sq_err_z.sum()
                depth_accum["rmse_log"] += ((np.log(pz) - np.log(gz)) ** 2).sum()
                depth_accum["mae"] += abs_err_z.sum()
                depth_accum["delta_1"] += (ratio < 1.25).sum()
                depth_accum["delta_2"] += (ratio < 1.25 ** 2).sum()
                depth_accum["delta_3"] += (ratio < 1.25 ** 3).sum()

                depth_total_valid += n_z
                depth_all_abs_err.append(abs_err_z)

                img_results["depth"] = {
                    "abs_rel": float((abs_err_z / gz).mean()),
                    "sq_rel": float((sq_err_z / gz).mean()),
                    "rmse": float(np.sqrt(sq_err_z.mean())),
                    "rmse_log": float(np.sqrt(((np.log(pz) - np.log(gz)) ** 2).mean())),
                    "mae": float(abs_err_z.mean()),
                    "median": float(np.median(abs_err_z)),
                    "delta_1": float((ratio < 1.25).mean()) * 100.0,
                    "delta_2": float((ratio < 1.25 ** 2).mean()) * 100.0,
                    "delta_3": float((ratio < 1.25 ** 3).mean()) * 100.0,
                    "n_valid": int(n_z),
                }

            per_image.append(img_results)
            total_samples += 1

            # ================================================================
            #  VISUALIZATION: [img0 | pred_disp | gt_disp | pred_depth | gt_depth | err_disp | err_depth]
            # ================================================================
            img0_vis = img0[b].cpu().numpy().astype(np.uint8)
            ph, pw = p_disp.shape
            img0_vis = cv2.resize(img0_vis, (pw, ph))
            img0_bgr = cv2.cvtColor(img0_vis, cv2.COLOR_RGB2BGR)

            def to_colormap(arr, vmin, vmax, valid_mask, cmap=cv2.COLORMAP_INFERNO):
                norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)
                norm[~valid_mask] = 0
                color = cv2.applyColorMap(norm, cmap)
                color[~valid_mask] = 0
                return color

            def to_errmap(err, valid_mask, clamp=50.0):
                err_c = err.copy()
                err_c[~valid_mask] = 0
                norm = np.clip(err_c / clamp * 255, 0, 255).astype(np.uint8)
                color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                color[~valid_mask] = 0
                return color

            # Disparity vis
            if n_d > 0:
                dmin = min(pd.min(), gd.min())
                dmax = max(pd.max(), gd.max())
                pred_disp_vis = to_colormap(p_disp, dmin, dmax, mask_d)
                gt_disp_vis = to_colormap(gt_disp, dmin, dmax, mask_d)
                err_disp_vis = to_errmap(np.abs(p_disp - gt_disp), mask_d, clamp=10.0)
            else:
                pred_disp_vis = np.zeros((ph, pw, 3), dtype=np.uint8)
                gt_disp_vis = np.zeros((ph, pw, 3), dtype=np.uint8)
                err_disp_vis = np.zeros((ph, pw, 3), dtype=np.uint8)

            # Depth vis
            if n_z > 0:
                zmin = min(pz.min(), gz.min())
                zmax = max(pz.max(), gz.max())
                pred_depth_vis = to_colormap(pred_depth, zmin, zmax, mask_z)
                gt_depth_vis = to_colormap(gt_depth, zmin, zmax, mask_z)
                err_depth_vis = to_errmap(np.abs(pred_depth - gt_depth), mask_z, clamp=50.0)
            else:
                pred_depth_vis = np.zeros((ph, pw, 3), dtype=np.uint8)
                gt_depth_vis = np.zeros((ph, pw, 3), dtype=np.uint8)
                err_depth_vis = np.zeros((ph, pw, 3), dtype=np.uint8)

            grid = np.concatenate([
                img0_bgr, pred_disp_vis, gt_disp_vis, err_disp_vis,
                pred_depth_vis, gt_depth_vis, err_depth_vis
            ], axis=1)
            save_path = Path(output_dir) / f'{total_samples - 1:06d}.png'
            cv2.imwrite(str(save_path), grid)

    # ================================================================
    #  AGGREGATE
    # ================================================================
    results = {"disparity": {}, "depth": {}, "per_image": per_image}

    if disp_total_valid > 0:
        mean_err = disp_accum["epe"] / disp_total_valid
        mean_sq = disp_accum["rmse"] / disp_total_valid
        results["disparity"] = {
            "epe": float(mean_err),
            "rmse": float(np.sqrt(mean_sq)),
            "std": float(np.sqrt(mean_sq - mean_err ** 2)),
            "median": float(np.median(np.concatenate(disp_all_abs_err))),
            "bad_1": float(disp_accum["bad_1"] / disp_total_valid) * 100.0,
            "bad_2": float(disp_accum["bad_2"] / disp_total_valid) * 100.0,
            "bad_3": float(disp_accum["bad_3"] / disp_total_valid) * 100.0,
            "total_valid_pixels": int(disp_total_valid),
            "total_samples": total_samples,
        }

    if depth_total_valid > 0:
        results["depth"] = {
            "abs_rel": float(depth_accum["abs_rel"] / depth_total_valid),
            "sq_rel": float(depth_accum["sq_rel"] / depth_total_valid),
            "rmse": float(np.sqrt(depth_accum["rmse"] / depth_total_valid)),
            "rmse_log": float(np.sqrt(depth_accum["rmse_log"] / depth_total_valid)),
            "mae": float(depth_accum["mae"] / depth_total_valid),
            "median": float(np.median(np.concatenate(depth_all_abs_err))),
            "delta_1": float(depth_accum["delta_1"] / depth_total_valid) * 100.0,
            "delta_2": float(depth_accum["delta_2"] / depth_total_valid) * 100.0,
            "delta_3": float(depth_accum["delta_3"] / depth_total_valid) * 100.0,
            "total_valid_pixels": int(depth_total_valid),
            "total_samples": total_samples,
        }

    return results


def print_results(results):
    d = results["disparity"]
    z = results["depth"]
    per = results["per_image"]

    # Disparity table
    print(f"\n{'='*100}")
    print(f"  DISPARITY METRICS")
    print(f"{'='*100}")
    print(f"  {'Image':<30s} {'EPE':>7s} {'RMSE':>7s} {'Std':>7s} {'Bad1%':>7s} {'Bad2%':>7s} {'D1%':>7s}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for m in per:
        if "disp" not in m:
            continue
        md = m["disp"]
        print(f"  {m['name']:<30s} {md['epe']:7.2f} {md['rmse']:7.2f} {md['std']:7.2f} {md['bad_1']:7.2f} {md['bad_2']:7.2f} {md['bad_3']:7.2f}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    print(f"  {'MEAN':<30s} {d['epe']:7.2f} {d['rmse']:7.2f} {d['std']:7.2f} {d['bad_1']:7.2f} {d['bad_2']:7.2f} {d['bad_3']:7.2f}")
    print(f"  {'MEDIAN ABS ERR':<30s} {d['median']:7.2f}")

    # Depth table
    print(f"\n{'='*100}")
    print(f"  DEPTH METRICS (mm)")
    print(f"{'='*100}")
    print(f"  {'Image':<30s} {'AbsRel':>7s} {'SqRel':>7s} {'RMSE':>8s} {'RMSElog':>8s} {'MAE':>8s} {'Med':>8s} {'d1':>7s} {'d2':>7s} {'d3':>7s}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
    for m in per:
        if "depth" not in m:
            continue
        mz = m["depth"]
        print(f"  {m['name']:<30s} {mz['abs_rel']:7.4f} {mz['sq_rel']:7.3f} {mz['rmse']:8.3f} {mz['rmse_log']:8.4f} {mz['mae']:8.3f} {mz['median']:8.3f} {mz['delta_1']:7.2f} {mz['delta_2']:7.2f} {mz['delta_3']:7.2f}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
    print(f"  {'MEAN':<30s} {z['abs_rel']:7.4f} {z['sq_rel']:7.3f} {z['rmse']:8.3f} {z['rmse_log']:8.4f} {z['mae']:8.3f} {z['median']:8.3f} {z['delta_1']:7.2f} {z['delta_2']:7.2f} {z['delta_3']:7.2f}")
    print(f"{'='*100}\n")


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    vis_dir = output_dir / f'vis_scale{args.scale}_keyframes'
    vis_dir.mkdir(exist_ok=True)

    dataset = SCARED_Keyframes(args.data_root, args.scale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    stereo_model = StereoDisparityEstimator(scale=args.scale, max_batch_size=1, model_name='fast-foundationstereo', ckpt_dir='/home/jhan3/endo_stereo/Fast-FoundationStereo/weights/23-36-37')

    results = run_eval(stereo_model, dataloader, scale=args.scale, output_dir=vis_dir)
    results["config"] = {"scale": args.scale, "scale_factor": args.scale_factor}

    print_results(results)

    results_fname = output_dir / 'results.json'
    with open(results_fname, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_fname}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/nfs/home/jhan3/scared_data/keyframes')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--scale_factor', type=float, default=256.0)
    parser.add_argument('--scale', type=float, default=1.0, help='resolution scaling factor')
    args = parser.parse_args()
    main(args)