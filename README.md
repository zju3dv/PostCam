# PostCam

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/CCQAQ/PostCam/)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://cccqaq.github.io/PostCam.github.io/)
[![License](https://img.shields.io/badge/License-Apache--2.0-orange)](https://github.com/zju3dv/PostCam/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2604.07209-b31b1b)](https://arxiv.org/abs/2511.17185)

## Environment

- Python 3.10
- CUDA 12.1

Create the environment with:

```bash
conda env create -f environment.yml
conda activate postcam
```

## Model Weights

Download the following model checkpoints into the `checkpoints/` directory:

| Model | Purpose | Source |
|---|---|---|
| **PostCam-1.3B** | v2v inference — 1.3B (Step 3) | [HuggingFace](https://huggingface.co/CCQAQ/PostCam/) |
| **Wan2.1-T2V-1.3B** | Text encoder + VAE + base model for 1.3B (Step 3) | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| **DA3 (Depth-Anything-3)** | Depth estimation (Step 2) | [HuggingFace](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE) |
| **Florence-2-large** | Video captioning (Step 1) | [HuggingFace](https://huggingface.co/microsoft/Florence-2-large) |

```bash
bash scripts/download.sh
```

Expected directory structure after downloading:
```
checkpoints/
├── PostCam/
│   └── postcam.ckpt
├── Wan2.1-T2V-1.3B/
├── DA3/
├── Florence-2-large/
```

## Supported Entry Points

### Inference

The full pipeline runs in three steps:

1. **Step 1** - Generate video captions using Florence-2.
2. **Step 2** - Estimate depth and camera poses with DA3, then convert outputs to the inference format.
3. **Step 3** - Run PostCam v2v inference.

All steps are wrapped in a single script:

```bash
bash run_pipeline.sh \
  --input_dir ./test \
  --traj_txt_path ./traj/y_left_30.txt
```

To run all bundled demo trajectories, use:

```bash
bash run_example.sh
```

`run_example.sh` runs the full pipeline once for the first trajectory, then reuses the generated captions and depth outputs for the remaining trajectories.

### Quick Start

```bash
# 1. Place your .mp4 video(s) in a folder
mkdir -p my_videos
cp your_video.mp4 my_videos/

# 2. Run the full pipeline
bash run_pipeline.sh \
  --input_dir ./my_videos \
  --traj_txt_path ./traj/y_left_30.txt \
  --step1_gpu 0 \
  --step2_gpu 0 \
  --step3_gpu 0
```

For multi-GPU task parallelism, pass comma-separated GPU IDs:

```bash
bash run_pipeline.sh \
  --input_dir ./my_videos \
  --traj_txt_path ./traj/y_left_30.txt \
  --step1_gpu 0,1,2,3 \
  --step2_gpu 0,1,2,3 \
  --step3_gpu 0,1,2,3
```

### Trajectory Control

The `--traj_txt_path` argument controls the camera trajectory for novel-view synthesis. Predefined trajectories are provided in the `traj/` directory:

| File | Motion |
|---|---|
| `y_left_30.txt` | Arc left 30 degrees |
| `y_right_30.txt` | Arc right 30 degrees |
| `x_up_30.txt` | Translate Up 30 degrees |
| `x_down_30.txt` | Translate Down 30 degrees |
| `zoom_in.txt` | Zoom in |
| `zoom_out.txt` | Zoom out |

#### Trajectory File Format

A trajectory file is a plain text file with **3 lines**, each containing space-separated keyframe values that are automatically interpolated to match the input video length:

```text
<line 1>  pitch (degrees): positive = orbit up, negative = orbit down
<line 2>  yaw (degrees):   positive = orbit left, negative = orbit right
<line 3>  displacement:    relative camera displacement scale
```

**Line 3 (displacement)** is a relative scale multiplied by the scene's estimated foreground depth:

- When pitch/yaw are non-zero, it controls the orbit radius.
- When both pitch and yaw are zero, it becomes a dolly zoom.

### All Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input_dir` | Yes | - | Input folder containing `.mp4` files |
| `--traj_txt_path` | Yes | - | Trajectory file, e.g. `./traj/y_left_30.txt` |
| `--checkpoint_path` | No | `./checkpoints/PostCam/postcam.ckpt` | PostCam checkpoint |
| `--config_path` | No | `./inference.yaml` | Inference config file |
| `--da3_model_path` | No | `./checkpoints/DA3` | DA3 depth model path |
| `--florence_model_path` | No | `./checkpoints/Florence-2-large` | Florence-2 model path |
| `--step1_gpu` | No | `0` | GPU ID(s) for Step 1, comma-separated for parallel captioning |
| `--step2_gpu` | No | `0` | GPU ID(s) for Step 2, comma-separated for parallel depth estimation |
| `--step3_gpu` | No | `0` | GPU ID(s) for Step 3, comma-separated for parallel inference |
| `--output_dir` | No | `./output` | Output root directory |
| `--skip_step1` | No | false | Skip caption generation |
| `--skip_step2` | No | false | Skip depth estimation and format conversion |
| `--skip_step3` | No | false | Skip PostCam inference |

### Skip Already-Completed Steps

If Step 1 or Step 2 outputs already exist, you can skip them:

```bash
bash run_pipeline.sh \
  --input_dir ./my_videos \
  --traj_txt_path ./traj/y_right_30.txt \
  --skip_step1 --skip_step2
```

This is useful when generating multiple camera trajectories for the same input videos.


## License

This project is licensed under the [Apache-2.0 License](https://github.com/zju3dv/PostCam/blob/main/LICENSE). Note that this license only applies to code in our library, the dependencies and submodules of which ([Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3), [Florence-2](https://github.com/anyantudre/Florence-2-Vision-Language-Model)) are separate and individually licensed.

---

## Acknowledgement
InSpatio-World utilizes a backbone based on [Wan2.1](https://github.com/Wan-Video/Wan2.1), with its training code referencing [ReCamMaster](https://github.com/KlingAIResearch/ReCamMaster). We sincerely thank the Wan and ReCamMaster team for their foundational work and open-source contribution. We also gratefully acknowledge [Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3), [Florence-2](https://github.com/anyantudre/Florence-2-Vision-Language-Model) for their excellent work that inspired and supported this project.