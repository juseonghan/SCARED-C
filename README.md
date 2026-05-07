# SCARED-C: Corrected Camera Poses for Endoscopic Depth Estimation
The SCARED dataset is a widely used benchmark for endoscopic depth estimation, offering ground-truth 3D reconstructions captured with a structured light sensor. However, the depth maps for non-keyframe images rely on robot kinematics that introduce substantial pose errors, limiting the reliably labeled portion of the dataset to 35 keyframes. We present SCARED-C, a corrected version of the SCARED dataset that expands the number of reliable RGB-D pairs from 35 to 17,135. Our pipeline applies COLMAP, a Structure-from-Motion system, to re-estimate camera poses for all frames, followed by a scale recovery step that aligns the resulting reconstructions to metric space using the ground-truth keyframe depth maps. We validate the corrected poses through (1) stereo disparity evaluation and (2) monocular depth estimation experiments. 

[Link to Corrected Dataset](https://huggingface.co/datasets/juseonghan/SCARED-C)

## Usage
You should only consider running this code if you're interested in re-running COLMAP and/or the scale recovery/reformatting scripts. Due to space constraints, we are unable to provide COLMAP output files. Note that running COLMAP at the original resolution (1024 x 1280) took numerous weeks to complete. 

The only real dependency is COLMAP, which you can install [here](https://colmap.github.io/install.html). Then, you can run

```
$ chmod +x run_colmap.sh
$ ./run_colmap.sh /path/to/downloaded/SCARED/dataset /path/to/output/dir
```
At the top of `run_colmap.sh` you can change lines 15 and 16 to specify which specific sequences you'd like to process.

To run the scale recovery code, you can run the following command.
```
$ python correct_scared.py \
    --colmap_dir /path/to/colmap/output/dir \
    --original_dir /path/to/original/SCARED/dataset \
    --output_dir /path/to/output/dir \
    --key x_y # can be 1_1, 1_2 to specify dataset_keyframe num
```
## Eval
To evaluate, you need to install the dependencies of FoundationStereo which you can find [here](https://github.com/NVlabs/FoundationStereo). There's also compatibility for [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) and [WAFT-Stereo](https://github.com/princeton-vl/WAFT-Stereo). Documentation and cleaner code coming soon!