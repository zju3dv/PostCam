import torch
import os
import argparse
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from diffsynth.models.wan_video_dit import WanModel
from diffsynth import WanVideoReCamMasterPipeline, ModelManager, load_state_dict
from diffsynth import save_video
from video_dataset import VideoDataset
from warper import Warper


MODEL_INPUT_FRAMES = 41
CAM_EMB_FRAMES = 11


def build_temporal_indices(source_frames, target_frames, device):
    if source_frames <= 0:
        raise ValueError("source_frames must be positive")
    if source_frames == target_frames:
        return torch.arange(source_frames, device=device, dtype=torch.long)
    if source_frames > target_frames:
        return torch.linspace(0, source_frames - 1, steps=target_frames, device=device).round().long()
    pad = torch.full((target_frames - source_frames,), source_frames - 1, device=device, dtype=torch.long)
    return torch.cat([torch.arange(source_frames, device=device, dtype=torch.long), pad], dim=0)


def resample_temporal_tensor(tensor, target_frames, time_dim=1):
    indices = build_temporal_indices(tensor.shape[time_dim], target_frames, tensor.device)
    return torch.index_select(tensor, time_dim, indices)


def save_video_tensor(video_tensor, output_path, file_stem, name):
    video_tensor = ((video_tensor.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    video_tensor = [Image.fromarray(frame) for frame in video_tensor]
    save_video(video_tensor, os.path.join(output_path, f"{file_stem}_{name}.mp4"), fps=15, quality=5)


class PostCamInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load VAE & text_encoder
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([
            args.checkpoints.text_encoder_path,
            args.checkpoints.vae_path,
        ])

        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)

        # Init DiT model
        print(f"Initializing DiT with kwargs: {args.dit_kwargs}")
        self.pipe.dit = WanModel(**args.dit_kwargs)

        # Load checkpoint
        if args.experiment.resume_ckpt_path:
            print(f"Loading checkpoint from {args.experiment.resume_ckpt_path}")
            state_dict = load_state_dict(args.experiment.resume_ckpt_path)
            missing, unexpected = self.pipe.dit.load_state_dict(state_dict, assign=True, strict=False)
            print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.requires_grad_(False)
        self.pipe.eval()

        # Move all models to device
        self.pipe.dit = self.pipe.dit.to(dtype=torch.bfloat16, device=self.device)
        self.pipe.vae = self.pipe.vae.to(dtype=torch.bfloat16, device=self.device)
        self.pipe.text_encoder = self.pipe.text_encoder.to(dtype=torch.bfloat16, device=self.device)

        self.funwarp = Warper(device='cuda')
        self.tiler_kwargs = args.vae_kwargs

    @torch.no_grad()
    def run(self, batch, batch_idx, output_path):
        self.pipe.device = self.device
        os.makedirs(output_path, exist_ok=True)
        file_stem = batch['source_video_name'][0] if "source_video_name" in batch else str(batch_idx)

        batch_source_video = resample_temporal_tensor(batch['source_video'], MODEL_INPUT_FRAMES, time_dim=1)
        batch_depths = resample_temporal_tensor(batch['depths'], MODEL_INPUT_FRAMES, time_dim=1)
        batch_source_extrinsics = resample_temporal_tensor(batch['source_extrinsics'], MODEL_INPUT_FRAMES, time_dim=1)
        batch_target_extrinsics = resample_temporal_tensor(batch['target_extrinsics'], MODEL_INPUT_FRAMES, time_dim=1)
        batch_intrinsics = resample_temporal_tensor(batch['intrinsics'], MODEL_INPUT_FRAMES, time_dim=1)
        batch_target_video = None
        if 'target_video' in batch:
            batch_target_video = resample_temporal_tensor(batch['target_video'], MODEL_INPUT_FRAMES, time_dim=1)

        # Encode source video
        source_video = batch_source_video  # (-1,1), btchw
        bs, num_frames, c, h, w = source_video.shape
        source_video = rearrange(source_video, 'b t c h w -> b c t h w')
        source_video = source_video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        source_latents = self.pipe.encode_video(source_video, **self.tiler_kwargs)

        if batch_target_video is not None:
            target_video = batch_target_video
            target_video = rearrange(target_video, 'b t c h w -> b c t h w')
            target_video = target_video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        else:
            target_video = None

        # Depth-based warping
        render_video = []
        warp_source_video = batch_source_video.float().to(device=self.device)
        warp_depths = batch_depths.float().to(device=self.device)
        warp_pose_s = batch_source_extrinsics.float().to(device=self.device)
        warp_pose_t = batch_target_extrinsics.float().to(device=self.device)
        warp_K = batch_intrinsics.float().to(device=self.device)

        for i in range(bs):
            warp_depths = warp_depths.clip(0.0001, 10000.0)
            render, mask, _, _ = self.funwarp.forward_warp(
                warp_source_video[i], None, warp_depths[i],
                warp_pose_s[i], warp_pose_t[i], warp_K[i], None,
                mask=False, twice=False
            )
            render_video.append(render)

        render_video = torch.stack(render_video)
        save_render_video = render_video.clone()
        render_video = rearrange(render_video, 'b t c h w -> (b t) c h w')
        render_video = F.interpolate(render_video, scale_factor=(0.25, 0.25), mode='bilinear', align_corners=False)
        render_video = rearrange(render_video, '(b t) c h w -> b c t h w', b=bs)
        render_video = render_video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        render_latents = self.pipe.encode_video(render_video, **self.tiler_kwargs)

        # Camera embedding
        cam_emb = resample_temporal_tensor(batch_target_extrinsics, CAM_EMB_FRAMES, time_dim=1)
        cam_emb = cam_emb.to(dtype=self.pipe.torch_dtype, device=self.device)

        # Sanity check: save source and render videos
        if self.args.experiment.enable_sanity_check:
            source_tensor = rearrange(source_video, "b c t h w-> (b t) h w c")
            render_tensor = rearrange(save_render_video, "b t c h w-> (b t) h w c")
            if target_video is not None:
                target_tensor = rearrange(target_video, "b c t h w-> (b t) h w c")
                save_video_tensor(target_tensor, output_path, file_stem, "target_video")
            save_video_tensor(source_tensor, output_path, file_stem, "source_video")
            save_video_tensor(render_tensor, output_path, file_stem, "render_video")

        # Generate video
        text = batch['text']
        video = self.pipe(
            prompt=text,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            source_latents=source_latents,
            cam_emb=cam_emb,
            render_latents=render_latents,
            cfg_scale=self.args.experiment.cfg_scale,
            num_inference_steps=50,
            num_frames=MODEL_INPUT_FRAMES,
            seed=0, tiled=True,
        )

        pred_path = os.path.join(output_path, f"{file_stem}_pred_video.mp4")
        save_video(video, pred_path, fps=15, quality=5)
        print(f"Saved prediction to {pred_path}")


def main():
    parser = argparse.ArgumentParser(description='PostCam Inference')
    parser.add_argument('--config', type=str, default='inference.yaml', help='Config file path')
    parser.add_argument('--traj_txt_path', type=str, default=None, help='Trajectory txt path')
    parser.add_argument('--cam_idx', type=int, default=None, help='Camera index')
    parser.add_argument('--output_path', type=str, default=None, help='Output path')
    parser.add_argument('--resume_ckpt_path', type=str, default=None, help='Checkpoint path')
    _args = parser.parse_args()

    args = OmegaConf.load(_args.config)
    if _args.traj_txt_path is not None:
        args.dataset.traj_txt_path = _args.traj_txt_path
    if _args.cam_idx is not None:
        args.dataset.cam_idx = _args.cam_idx
    if _args.output_path is not None:
        args.experiment.output_path = _args.output_path
    if _args.resume_ckpt_path is not None:
        args.experiment.resume_ckpt_path = _args.resume_ckpt_path

    # Build dataset
    dataset = VideoDataset(**args.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.dataloader.batch_size,
        num_workers=args.dataloader.num_workers,
    )

    # Build model
    model = PostCamInference(args)

    # Determine output path
    if args.dataset.traj_txt_path is not None:
        cam_name = args.dataset.traj_txt_path.split('/')[-1].split('.')[0]
    else:
        cam_name = str(args.dataset.cam_idx)
    recam_name = args.experiment.resume_ckpt_path.split('/')[-1].split('.')[0]
    output_path = os.path.join(args.experiment.output_path, f"results_{recam_name}_{cam_name}")

    # Run inference
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
        model.run(batch, batch_idx, output_path)


if __name__ == '__main__':
    main()
