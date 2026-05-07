# main.py 
import argparse 
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from PIL import Image 
import cv2 

import torch 
import torch.nn.functional as F 

from disparity_estimator import StereoDisparityEstimator
# from dataset import OriginalSCAREDDataset, CorrectedSCAREDDataset, TarLoader, CorrectedSCAREDDataseWithRGBmp4
from dataset import SCAREDStereoDataset
import geometry 

VIS_FREQ = 1

def save_depth_overlay_comparison(
    rgb: np.ndarray,         # (H, W, 3) uint8
    pred_depth: np.ndarray,  # (H, W)
    gt_depth: np.ndarray,    # (H, W)
    valid: np.ndarray,       # (H, W) bool
    save_path: str,
    alpha: float = 0.7,
):
    """Save side-by-side pred vs GT depth overlay on RGB."""
    def overlay(image, depth, mask):
        if mask.sum() == 0:
            return image
        lower, upper = np.percentile(depth[mask], [1, 99])
        normed = np.clip(depth, lower, upper)
        if upper > lower:
            normed = (normed - lower) / (upper - lower)
            normed[~mask] = 0.0
        else:
            normed = np.zeros_like(depth)
        colored = cv2.applyColorMap((normed * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        out = image.copy()
        out[mask] = (alpha * colored[mask] + (1 - alpha) * image[mask]).astype(np.uint8)
        return out

    # Resize RGB to match depth resolution if needed
    dh, dw = pred_depth.shape[:2]
    if rgb.shape[:2] != (dh, dw):
        rgb = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)

    gt_valid = valid & (gt_depth > 0)
    pred_valid = valid & (pred_depth > 0)

    pred_vis = overlay(rgb, pred_depth, pred_valid)
    gt_vis = overlay(rgb, gt_depth, gt_valid)

    # Side-by-side: GT left, Pred right
    combined = np.concatenate([gt_vis, pred_vis], axis=1)

    # Add labels
    cv2.putText(combined, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(combined, "Pred", (dw + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    Image.fromarray(combined).save(save_path)

def build_valid_mask(gt_full, scale):
    """Downsample GT and build conservative valid mask via nearest-neighbor."""
    gt_tensor = torch.from_numpy(gt_full).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    valid_orig = (np.isfinite(gt_full) & (gt_full > 0))
    valid_tensor = torch.from_numpy(valid_orig.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    valid_down = F.interpolate(valid_tensor, scale_factor=scale, mode='nearest')

    gt_safe = gt_tensor.clone()
    gt_safe[torch.isnan(gt_safe)] = 0.0
    gt_down = F.interpolate(gt_safe, scale_factor=scale, mode='bilinear', align_corners=False)

    gt_np = gt_down.squeeze().numpy().astype(np.float64)
    valid = (valid_down.squeeze().numpy() > 0) & (gt_np > 0)
    return gt_np, valid


def compute_disp_metrics(pred_disp, gt_disp, valid):
    """Compute disparity metrics for a single image. All values in pixel units."""
    p = pred_disp[valid]
    g = gt_disp[valid]
    n = valid.sum()
    if n == 0:
        return None

    abs_err = np.abs(p - g)
    sq_err = (p - g) ** 2
    mean_err = abs_err.mean()
    mean_sq = sq_err.mean()

    return {
        "epe": float(mean_err),
        "rmse": float(np.sqrt(mean_sq)),
        "std": float(np.sqrt(max(mean_sq - mean_err ** 2, 0.0))),
        "bad_1": float((abs_err > np.maximum(1, 0.05 * g)).mean()) * 100.0,
        "bad_2": float((abs_err > np.maximum(2, 0.05 * g)).mean()) * 100.0,
        "bad_3": float((abs_err > np.maximum(3, 0.05 * g)).mean()) * 100.0,
        "n_valid": int(n),
        "_abs_err": abs_err,  # kept for accumulation, stripped before JSON
        "_sq_err": sq_err,
        "disp_bad_1_count": int((abs_err > np.maximum(1, 0.05 * g)).sum()),
        "disp_bad_2_count": int((abs_err > np.maximum(2, 0.05 * g)).sum()),
        "disp_bad_3_count": int((abs_err > np.maximum(3, 0.05 * g)).sum()),
    }


def compute_depth_metrics(pred_depth, gt_depth, valid):
    """Compute depth metrics for a single image. Units match GT (typically mm)."""
    mask = valid & (pred_depth > 0)
    p = pred_depth[mask]
    g = gt_depth[mask]
    n = mask.sum()
    if n == 0:
        return None

    abs_err = np.abs(p - g)
    sq_err = (p - g) ** 2
    ratio = np.maximum(p / g, g / p)

    return {
        "abs_rel": float((abs_err / g).mean()),
        "sq_rel": float((sq_err / g).mean()),
        "rmse": float(np.sqrt(sq_err.mean())),
        "rmse_log": float(np.sqrt(((np.log(p) - np.log(g)) ** 2).mean())),
        "mae": float(abs_err.mean()),
        "median": float(np.median(abs_err)),
        "delta_1": float((ratio < 1.25).mean()) * 100.0,
        "delta_2": float((ratio < 1.25 ** 2).mean()) * 100.0,
        "delta_3": float((ratio < 1.25 ** 3).mean()) * 100.0,
        "n_valid": int(n),
        "_abs_err": abs_err,
        "_sq_err": sq_err,
        "_abs_rel_arr": abs_err / g,
        "_sq_rel_arr": sq_err / g,
        "_log_sq_arr": (np.log(p) - np.log(g)) ** 2,
        "_delta_1_arr": (ratio < 1.25),
        "_delta_2_arr": (ratio < 1.25 ** 2),
        "_delta_3_arr": (ratio < 1.25 ** 3),
    }


class MetricsAccumulator:
    def __init__(self):
        # Disparity – running sums
        self.disp_sum_abs = 0.0
        self.disp_sum_sq = 0.0
        self.disp_bad_1 = 0
        self.disp_bad_2 = 0
        self.disp_bad_3 = 0
        self.disp_total = 0

        # Depth – running sums
        self.depth_sum_abs = 0.0
        self.depth_sum_sq = 0.0
        self.depth_sum_abs_rel = 0.0
        self.depth_sum_sq_rel = 0.0
        self.depth_sum_log_sq = 0.0
        self.depth_delta_1 = 0
        self.depth_delta_2 = 0
        self.depth_delta_3 = 0
        self.depth_total = 0

        self.per_image = []
        self.total_samples = 0

    def add_disp(self, m):
        n = m["n_valid"]
        self.disp_sum_abs += m["_abs_err"].sum()
        self.disp_sum_sq += m["_sq_err"].sum()
        self.disp_bad_1 += m["disp_bad_1_count"]
        self.disp_bad_2 += m["disp_bad_2_count"]
        self.disp_bad_3 += m["disp_bad_3_count"]
        self.disp_total += n

    def add_depth(self, m):
        n = m["n_valid"]
        self.depth_sum_abs += m["_abs_err"].sum()
        self.depth_sum_sq += m["_sq_err"].sum()
        self.depth_sum_abs_rel += m["_abs_rel_arr"].sum()
        self.depth_sum_sq_rel += m["_sq_rel_arr"].sum()
        self.depth_sum_log_sq += m["_log_sq_arr"].sum()
        self.depth_delta_1 += m["_delta_1_arr"].sum()
        self.depth_delta_2 += m["_delta_2_arr"].sum()
        self.depth_delta_3 += m["_delta_3_arr"].sum()
        self.depth_total += n

    def add_image(self, name, disp_m, depth_m):
        entry = {"name": name}
        if disp_m is not None:
            entry["disp"] = {k: v for k, v in disp_m.items() if not k.startswith("_")}
            self.add_disp(disp_m)
        if depth_m is not None:
            entry["depth"] = {k: v for k, v in depth_m.items() if not k.startswith("_")}
            self.add_depth(depth_m)
        self.per_image.append(entry)
        self.total_samples += 1

    def aggregate(self):
        results = {"per_image": self.per_image, "total_samples": self.total_samples}

        if self.disp_total > 0:
            # all_abs = np.concatenate(self.disp_abs_err)
            # all_sq = np.concatenate(self.disp_sq_err)
            mean_err = self.disp_sum_abs / self.disp_total
            mean_sq = self.disp_sum_sq / self.disp_total

            # Bad-pixel rates: weighted average across images by valid pixel count
            total_bad_1 = sum(m["disp"]["bad_1"] / 100.0 * m["disp"]["n_valid"] for m in self.per_image if "disp" in m)
            total_bad_2 = sum(m["disp"]["bad_2"] / 100.0 * m["disp"]["n_valid"] for m in self.per_image if "disp" in m)
            total_bad_3 = sum(m["disp"]["bad_3"] / 100.0 * m["disp"]["n_valid"] for m in self.per_image if "disp" in m)

            results["disparity"] = {
                "epe": float(mean_err),
                "rmse": float(np.sqrt(mean_sq)),
                "std": float(np.sqrt(max(mean_sq - mean_err ** 2, 0.0))),
                # "median": float(np.median(all_abs)),
                "bad_1": float(total_bad_1 / self.disp_total) * 100.0,
                "bad_2": float(total_bad_2 / self.disp_total) * 100.0,
                "bad_3": float(total_bad_3 / self.disp_total) * 100.0,
                "total_valid_pixels": int(self.disp_total),
            }

        if self.depth_total > 0:
            # all_abs = np.concatenate(self.depth_abs_err)
            # all_sq = np.concatenate(self.depth_sq_err)
            # all_abs_rel = np.concatenate(self.depth_abs_rel)
            # all_sq_rel = np.concatenate(self.depth_sq_rel)
            # all_log_sq = np.concatenate(self.depth_log_sq)
            
            mean_abs_rel = self.depth_sum_abs_rel / self.depth_total
            mean_sq_rel = self.depth_sum_sq_rel / self.depth_total
            mean_sq = self.depth_sum_sq / self.depth_total
            mean_sq_log = self.depth_sum_log_sq / self.depth_total 
            mean_abs = self.depth_sum_abs / self.depth_total 
            mean_d1 = self.depth_delta_1 / self.depth_total 
            mean_d2 = self.depth_delta_2 / self.depth_total 
            mean_d3 = self.depth_delta_3 / self.depth_total 

            results["depth"] = {
                "abs_rel": float(mean_abs_rel),
                "sq_rel": float(mean_sq_rel),
                "rmse": float(np.sqrt(mean_sq)),
                "rmse_log": float(np.sqrt(mean_sq_log)),
                "mae": float(mean_abs),
                # "median": float(np.median(all_abs)),
                "delta_1": float(mean_d1) * 100.0,
                "delta_2": float(mean_d2) * 100.0,
                "delta_3": float(mean_d3) * 100.0,
                "total_valid_pixels": int(self.depth_total),
            }

        return results

def evaluate_original(original_dataloader, stereo_model, scale=1.0, debug=False, output_dir=None):
    accum = MetricsAccumulator()

    vis_dir = None
    if output_dir is not None:
        vis_dir = Path(output_dir) / "visualizations" / "original"
        vis_dir.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        for sample in tqdm(original_dataloader, desc="Evaluating Original"):
            img0 = sample["img0"].cuda().float()
            img1 = sample["img1"].cuda().float()
            result = stereo_model.estimate_batch(img0, img1)
            pred_disp = result["disp"].cpu().numpy().astype(np.float64)  # [B, H', W']

            for b in range(pred_disp.shape[0]):
                name = sample["fname"][b]
                fx = float(sample["fx"][b])
                baseline = float(sample["baseline"][b])
                fx_scaled = fx * scale

                # -- Disparity --
                gt_disp_full = sample["gt"][b].numpy().astype(np.float64)
                gt_disp, disp_valid = build_valid_mask(gt_disp_full, scale)
                gt_disp = gt_disp * scale  # scale disparity to match resolution
                disp_m = compute_disp_metrics(pred_disp[b], gt_disp, disp_valid)

                # -- Depth --
                gt_depth_full = sample["depth"][b].numpy().astype(np.float64)
                gt_depth, depth_valid = build_valid_mask(gt_depth_full, scale)
                pred_depth = np.where(pred_disp[b] > 0, (fx_scaled * baseline) / pred_disp[b], 0.0)
                depth_m = compute_depth_metrics(pred_depth, gt_depth, depth_valid)

                accum.add_image(name, disp_m, depth_m)

                # Visualize every 100 frames
                if vis_dir is not None and accum.total_samples % VIS_FREQ == 0:
                    rgb_np = sample["img0"][b].cpu().numpy().astype(np.uint8)
                    save_depth_overlay_comparison(
                        rgb_np, pred_depth, gt_depth, depth_valid,
                        str(vis_dir / f"{accum.total_samples:06d}.png"),
                    )

            if debug and accum.total_samples >= 20:
                break

    return accum.aggregate()

def evaluate_corrected(corrected_ds, stereo_model, scale=1.0, batch_size=1, debug=False, output_dir=None):
    accum = MetricsAccumulator()

    vis_dir = None
    if output_dir is not None:
        vis_dir = Path(output_dir) / "visualizations" / "corrected"
        vis_dir.mkdir(exist_ok=True, parents=True)

    num_seq = 1 if debug else len(corrected_ds)

    for i in tqdm(range(num_seq), desc='Sequences'):
        vr, tar_reader, calib = corrected_ds.get_readers(i)

        # Pre-extract geometry matrices for this sequence
        Rot = np.array(calib['R1']['data']).reshape((3, 3))
        RT = geometry.create_RT(R=Rot)
        P1 = np.array(calib['P1']['data']).reshape(3, 4)
        P2 = np.array(calib['P2']['data']).reshape(3, 4)
        K1 = P1[:, :3]
        fx = calib['K1']['data'][0]
        baseline = -calib['T']['data'][0]

        # -- Rectification setup (computed once per sequence on first frame) --
        K1_raw = np.array(calib['K1']['data']).reshape(3, 3)
        K2_raw = np.array(calib['K2']['data']).reshape(3, 3)
        D1 = np.array(calib.get('D1', {}).get('data', [0, 0, 0, 0, 0])).astype(np.float64)
        D2 = np.array(calib.get('D2', {}).get('data', [0, 0, 0, 0, 0])).astype(np.float64)
        R = np.array(calib['R']['data']).reshape(3, 3).astype(np.float64)
        T = np.array(calib['T']['data']).reshape(3, 1).astype(np.float64)
        left_rect_map = None
        right_rect_map = None

        num_frames = min(len(tar_reader), len(vr))
        if debug:
            num_frames = min(num_frames, 50 - accum.total_samples)
            if num_frames <= 0:
                break

        seq_name = corrected_ds.gt_tar_paths[i].split('/')
        seq_label = f"{seq_name[-4]}/{seq_name[-3]}"

        with torch.no_grad():
            for batch_start in tqdm(range(1, num_frames, batch_size), desc=seq_label):
                batch_end = min(batch_start + batch_size, num_frames)
                B = batch_end - batch_start

                imgL_list, imgR_list = [], []
                gt_disp_list, gt_depth_list = [], []
                names = []

                for j in range(batch_start, batch_end):

                    frame = vr[j].asnumpy()
                    h = frame.shape[0] // 2
                    imgL = frame[:h]
                    imgR = frame[h:]

                    # Build rectification maps on first frame
                    if left_rect_map is None:
                        image_size = imgL.shape[:2]
                        R1_rect, R2_rect, P1_rect, P2_rect, Q, roi1, roi2 = cv2.stereoRectify(
                            K1_raw, D1, K2_raw, D2,
                            image_size[::-1],  # (W, H) for OpenCV
                            R, T,
                            alpha=-1,
                        )
                        left_rect_map = cv2.initUndistortRectifyMap(
                            K1_raw, D1, R1_rect, P1_rect,
                            image_size[::-1], cv2.CV_32FC1,
                        )
                        right_rect_map = cv2.initUndistortRectifyMap(
                            K2_raw, D2, R2_rect, P2_rect,
                            image_size[::-1], cv2.CV_32FC1,
                        )

                    imgL = cv2.remap(imgL, left_rect_map[0], left_rect_map[1], cv2.INTER_LINEAR)
                    imgR = cv2.remap(imgR, right_rect_map[0], right_rect_map[1], cv2.INTER_LINEAR)

                    # Read single GT frame from tar
                    gt_img3d = tar_reader[j]
                    gt_ptcloud = geometry.img3d_to_ptcloud(gt_img3d)
                    ptcloud_rotated = geometry.transform_pts(gt_ptcloud, RT)
                    gt_disp = geometry.ptcloud_to_disparity(ptcloud_rotated, P1, P2, imgL.shape[:2])
                    gt_depth = geometry.ptcloud_to_depthmap(ptcloud_rotated, K1, np.zeros(5), imgL.shape[:2])
                    del gt_img3d, gt_ptcloud, ptcloud_rotated

                    imgL_list.append(imgL)
                    imgR_list.append(imgR)
                    gt_disp_list.append(gt_disp)
                    gt_depth_list.append(gt_depth)
                    names.append(f"{seq_label}/frame_{j:04d}")

                img0 = torch.from_numpy(np.stack(imgL_list, axis=0)).cuda().float()
                img1 = torch.from_numpy(np.stack(imgR_list, axis=0)).cuda().float()
                result = stereo_model.estimate_batch(img0, img1)
                pred_disp = result["disp"].cpu().numpy().astype(np.float64)
                del img0, img1, result

                for b in range(B):
                    fx_scaled = fx * scale

                    gt_d, dv = build_valid_mask(gt_disp_list[b], scale)
                    gt_d = gt_d * scale
                    disp_m = compute_disp_metrics(pred_disp[b], gt_d, dv)

                    gt_z, zv = build_valid_mask(gt_depth_list[b], scale)
                    pred_depth_map = np.where(pred_disp[b] > 0, (fx_scaled * baseline) / pred_disp[b], 0.0)
                    depth_m = compute_depth_metrics(pred_depth_map, gt_z, zv)

                    accum.add_image(names[b], disp_m, depth_m)

                    if vis_dir is not None and accum.total_samples % VIS_FREQ == 0:
                        save_depth_overlay_comparison(
                            imgL_list[b], pred_depth_map, gt_z, zv,
                            str(vis_dir / f"{accum.total_samples:06d}.png"),
                        )

                del imgL_list, imgR_list, gt_disp_list, gt_depth_list, pred_disp

        tar_reader.__exit__()

    return accum.aggregate()

# ============================================================
#  Printing
# ============================================================

def print_results(label, results):
    per = results["per_image"]

    if "disparity" in results:
        d = results["disparity"]
        print(f"\n{'='*100}")
        print(f"  {label} — DISPARITY METRICS (px)")
        print(f"{'='*100}")
        print(f"  {'Image':<35s} {'EPE':>7s} {'RMSE':>7s} {'Std':>7s} {'Bad1%':>7s} {'Bad2%':>7s} {'D1%':>7s}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for m in per:
            if "disp" not in m:
                continue
            md = m["disp"]
            print(f"  {m['name']:<35s} {md['epe']:7.2f} {md['rmse']:7.2f} {md['std']:7.2f} {md['bad_1']:7.2f} {md['bad_2']:7.2f} {md['bad_3']:7.2f}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        print(f"  {'MEAN':<35s} {d['epe']:7.2f} {d['rmse']:7.2f} {d['std']:7.2f} {d['bad_1']:7.2f} {d['bad_2']:7.2f} {d['bad_3']:7.2f}")
        # print(f"  {'MEDIAN ABS ERR':<35s} {d['median']:7.2f}")

    if "depth" in results:
        z = results["depth"]
        print(f"\n{'='*120}")
        print(f"  {label} — DEPTH METRICS (mm)")
        print(f"{'='*120}")
        print(f"  {'Image':<35s} {'AbsRel':>7s} {'SqRel':>7s} {'RMSE':>8s} {'RMSElog':>8s} {'MAE':>8s} {'Med':>8s} {'d1':>7s} {'d2':>7s} {'d3':>7s}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
        for m in per:
            if "depth" not in m:
                continue
            mz = m["depth"]
            print(f"  {m['name']:<35s} {mz['abs_rel']:7.4f} {mz['sq_rel']:7.3f} {mz['rmse']:8.3f} {mz['rmse_log']:8.4f} {mz['mae']:8.3f} {mz['median']:8.3f} {mz['delta_1']:7.2f} {mz['delta_2']:7.2f} {mz['delta_3']:7.2f}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
        print(f"  {'MEAN':<35s} {z['abs_rel']:7.4f} {z['sq_rel']:7.3f} {z['rmse']:8.3f} {z['rmse_log']:8.4f} {z['mae']:8.3f} {z['delta_1']:7.2f} {z['delta_2']:7.2f} {z['delta_3']:7.2f}")
        print(f"{'='*120}\n")


# ============================================================
#  Main
# ============================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Model (shared)
    # stereo_model = StereoDisparityEstimator(scale=args.scale, max_batch_size=args.batch_size)
    stereo_model = StereoDisparityEstimator(
        scale=args.scale, 
        max_batch_size=args.batch_size, 
        model_name='fast-foundationstereo', 
        ckpt_dir='/home/jhan3/endo_stereo/Fast-FoundationStereo/weights/23-36-37'
    )

    # if args.dataset == 'original':
    #     original_ds = OriginalSCAREDDataset()
    #     original_dl = torch.utils.data.DataLoader(original_ds, batch_size=args.batch_size, shuffle=False)
    #     results = evaluate_original(original_dl, stereo_model, scale=args.scale, debug=args.debug, output_dir=args.output_dir)
    # elif args.dataset == 'corrected':
    #     corrected_ds = CorrectedSCAREDDataseWithRGBmp4()
    #     results = evaluate_corrected(corrected_ds, stereo_model, scale=args.scale,
    #                                        batch_size=args.batch_size, debug=args.debug, output_dir=args.output_dir)
    # else:
    #     raise Exception('huh?')
    ds = SCAREDStereoDataset(keys=['1_1', '1_2', '1_3', '2_1', '2_2', '2_3', '2_4', '3_1', '3_2', '3_3', '3_4', '6_1', '6_2', '6_3', '6_4', '7_1', '7_2', '7_3', '7_4'])

    results = evaluate_corrected(ds, stereo_model, scale=args.scale, 
                                 batch_size=args.batch_size, debug=args.debug,
                                 output_dir=args.output_dir)

    print_results(args.dataset, results)

    results_path = output_dir / f'eval_results_{args.dataset}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=float, default=1.0, help='resolution scaling factor')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--dataset', type=str, default='original', choices=['original', 'corrected'], help='which dataset to run eval on')
    parser.add_argument('--output_dir', default='/home/jhan3/scared_correction/eval/results', help='output directory')
    parser.add_argument('--debug', action='store_true', help='run on small subset')
    args = parser.parse_args()
    main(args)