import os
import torch
from PIL import Image
import numpy as np
import json
import ast
from torchvision import transforms
from dataset.utils import read_frames, generate_traj_txt
import torch.nn.functional as F


class TestDataset():
    def __init__(self, sample_n_frames, sample_stride, sample_size, cam_idx=1, traj_txt_path=None):
        self.sample_n_frames = sample_n_frames
        self.sample_stride = sample_stride
        self.sample_size = sample_size  # h, w
        self.traj_txt_path = traj_txt_path
        self.cam_idx = cam_idx

    def temporal_frame_indices(self, source_frames, target_frames):
        if source_frames <= 0:
            raise ValueError("source_frames must be positive")
        if source_frames == target_frames:
            return torch.arange(source_frames, dtype=torch.long)
        if source_frames > target_frames:
            return torch.linspace(0, source_frames - 1, steps=target_frames).round().long()
        pad = torch.full((target_frames - source_frames,), source_frames - 1, dtype=torch.long)
        return torch.cat([torch.arange(source_frames, dtype=torch.long), pad], dim=0)

    def temporal_resample(self, tensor, target_frames):
        indices = self.temporal_frame_indices(tensor.shape[0], target_frames)
        return torch.index_select(tensor, 0, indices.to(tensor.device))

    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def read_matrix(self, file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        parsed_data = []
        for line in data.strip().split('\n'):
            if line.strip():
                parsed_data.append(ast.literal_eval(line.strip()))
        return np.array(parsed_data, dtype=np.float32)

    def resize_crop(self, K, source_video, target_video=None, depths=None):
        """
        source_video: t c h w, [0,1]
        depths: t c h w
        K: t 3 3
        """
        original_height, original_width = source_video.shape[-2:]
        target_height, target_width = self.sample_size

        scale_x = original_width / depths.shape[-1]
        scale_y = original_height / depths.shape[-2]

        K[:, 0, 0] *= scale_x
        K[:, 1, 1] *= scale_y
        K[:, 0, 2] = original_width / 2
        K[:, 1, 2] = original_height / 2

        scale = max(target_height / original_height, target_width / original_width)
        new_height = int(original_height * scale)
        new_width = int(original_width * scale)

        pixel_transforms = transforms.Compose([
            transforms.Resize((new_height, new_width)),
            transforms.CenterCrop(self.sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        source_video = pixel_transforms(source_video)
        if target_video is not None:
            target_video = pixel_transforms(target_video)

        if depths is not None:
            depth_transforms = transforms.Compose([
                transforms.Resize((new_height, new_width)),
                transforms.CenterCrop(self.sample_size),
            ])
            depths = depth_transforms(depths)

        K[:, 0, 0] *= scale
        K[:, 1, 1] *= scale
        K[:, 0, 2] = K[:, 0, 2] * scale - (new_width - target_width) / 2
        K[:, 1, 2] = K[:, 1, 2] * scale - (new_height - target_height) / 2
        return K, source_video, target_video, depths

    def get_traj_Twc(self, cam_data, cam_idx, view_idx):
        T_cw_colmap_ue = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
        traj = np.stack(traj).transpose(0, 2, 1)

        Tcws = []
        for Twc_ue in traj:
            Tcw_colmap = T_cw_colmap_ue @ np.linalg.inv(Twc_ue)
            Tcws.append(Tcw_colmap)
        Tcws = torch.from_numpy(np.array(Tcws))

        Tc0w = Tcws[0]
        Twcs_aligned = []
        for i in range(len(Tcws)):
            Twcs_aligned.append(Tc0w @ (Tcws[i].inverse()))
        Twcs_aligned = torch.stack(Twcs_aligned)
        Twcs_aligned[:, :3, 3] /= 100

        return Twcs_aligned

    def get_data(self, source_conf):
        data = {}
        data['text'] = source_conf['text']
        source_video = read_frames(source_conf['video_path'])  # t c h w
        source_video_name = os.path.splitext(os.path.basename(source_conf['video_path']))[0]
        data['source_video_name'] = source_video_name
        n_frames, c, h, w = source_video.shape
        frame_ids = list(range(n_frames))

        # ===================== depth =====================
        depth_path = os.path.join(source_conf["vggt_depth_path"], 'depth')
        depth_files_list = os.listdir(depth_path)
        depth_files = sorted(depth_files_list, key=lambda x: int(os.path.splitext(x)[0]))
        depth_files = [depth_files[i] for i in frame_ids]
        depths = []
        for depth_file in depth_files:
            depth = np.array(Image.open(os.path.join(depth_path, depth_file))).astype(np.uint16)
            depths.append(torch.tensor(depth))
        with open(os.path.join(source_conf["vggt_depth_path"], 'metadata.txt'), 'r') as f:
            depths_min, depths_max = tuple([float(t) for t in f.readline().strip().split(' ')])
        depths = torch.stack(depths, dim=0).float().unsqueeze(1)
        depths = (depths / 65535.0) * (depths_max - depths_min) + depths_min

        # ===================== intrinsics =====================
        intrinsics = self.read_matrix(source_conf['vggt_extrinsics_path'].replace('extrinsics.txt', 'intrinsics.txt'))
        intrinsics = intrinsics.repeat(len(frame_ids), axis=0)
        intrinsics = torch.tensor(intrinsics)

        intrinsics, source_video, _, depths = self.resize_crop(intrinsics, source_video, None, depths)
        data['intrinsics'] = intrinsics
        data['source_video'] = source_video
        data['depths'] = depths

        # ===================== target extrinsics =====================
        scale_ratio = 1.0
        if self.traj_txt_path is not None:
            radius = depths[0].min() * source_conf['radius_ratio']
            print('Foreground mean (radius):', radius)
            with open(self.traj_txt_path, 'r') as file:
                lines = file.readlines()
                x_up_angle = [float(i) for i in lines[0].split()]
                y_left_angle = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
                r_zoom = [float(i) * depths[0].min() for i in lines[2].split()]
            target_extrinsics = generate_traj_txt(x_up_angle, y_left_angle, r, r_zoom, n_frames)
            target_extrinsics = torch.tensor(target_extrinsics).inverse()
            data['target_extrinsics'] = target_extrinsics

            # Scale ratio calibration
            with open("./traj/y_left_30.txt", 'r') as file:
                lines = file.readlines()
                x_up_angle = [float(i) for i in lines[0].split()]
                y_left_angle = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
                r_zoom = [float(i) * depths[0].min() for i in lines[2].split()]
            target_extrinsics_30_left = generate_traj_txt(x_up_angle, y_left_angle, r, r_zoom, n_frames)
            tgt_Tcws_30_left_pred = torch.tensor(target_extrinsics_30_left).inverse()

            tgt_idx = 9
            target_camera_path = "./camera_extrinsics.json"
            with open(target_camera_path, 'r') as file:
                cam_data = json.load(file)
            cam_idx = list(range(len(cam_data)))
            tgt_Tcws_30_left_gt = self.get_traj_Twc(cam_data, cam_idx, tgt_idx).inverse()

            tgt_Twcs_gt = tgt_Tcws_30_left_gt
            tgt_Twcs_pred = tgt_Tcws_30_left_pred
            dist_gt = (tgt_Twcs_gt[-1, :3, 3] - tgt_Twcs_gt[0, :3, 3]).norm()
            dist_pred = (tgt_Twcs_pred[-1, :3, 3] - tgt_Twcs_pred[0, :3, 3]).norm()
            if dist_gt > 1e-2 and dist_pred > 1e-2:
                scale_ratio = dist_gt / dist_pred
            print(f'scale_ratio: {scale_ratio}')

            data['target_extrinsics'][:, :3, 3] = data['target_extrinsics'][:, :3, 3] * scale_ratio
            data['depths'] = data['depths'] * scale_ratio
        else:
            tgt_idx = self.cam_idx
            target_camera_path = "./camera_extrinsics.json"
            with open(target_camera_path, 'r') as file:
                cam_data = json.load(file)
            cam_idx = list(range(len(cam_data)))
            tgt_Tcws = self.get_traj_Twc(cam_data, cam_idx, tgt_idx).inverse()
            data['target_extrinsics'] = self.temporal_resample(tgt_Tcws, n_frames)

        # ===================== source extrinsics (from VGGT) =====================
        source_vggt_extrinsics = self.read_matrix(source_conf['vggt_extrinsics_path'])
        source_vggt_extrinsics = torch.tensor(source_vggt_extrinsics)
        n_frames = len(source_vggt_extrinsics)
        bottom_row = torch.tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(n_frames, 1, 1)
        source_vggt_extrinsics = torch.cat([source_vggt_extrinsics, bottom_row], dim=1)[frame_ids]
        source_vggt_extrinsics = source_vggt_extrinsics @ source_vggt_extrinsics[0].inverse()
        data['source_extrinsics'] = source_vggt_extrinsics
        data['source_extrinsics'][:, :3, 3] = data['source_extrinsics'][:, :3, 3] * scale_ratio

        assert data['intrinsics'].shape[0] == data['source_video'].shape[0] == data['depths'].shape[0] == data['source_extrinsics'].shape[0] == data['target_extrinsics'].shape[0]

        return data
