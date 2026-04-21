import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange


class Warper:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.dtype = torch.float32

    def forward_warp(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        Tc1w: torch.Tensor,
        Tc2w: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor],
        mask=False,
        twice=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Warp frame1 to next view using bilinear splatting.
        :param frame1: (b, 3, h, w) in range [-1, 1]
        :param mask1: (b, 1, h, w) optional
        :param depth1: (b, 1, h, w)
        :param Tc1w: (b, 4, 4) extrinsic of first view
        :param Tc2w: (b, 4, 4) extrinsic of second view
        :param intrinsic1: (b, 3, 3)
        :param intrinsic2: (b, 3, 3) optional
        """
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        frame1 = frame1.to(self.device).to(self.dtype)
        mask1 = mask1.to(self.device).to(self.dtype)
        depth1 = depth1.to(self.device).to(self.dtype)
        Tc1w = Tc1w.to(self.device).to(self.dtype)
        Tc2w = Tc2w.to(self.device).to(self.dtype)
        intrinsic1 = intrinsic1.to(self.device).to(self.dtype)
        intrinsic2 = intrinsic2.to(self.device).to(self.dtype)

        trans_points1 = self.compute_transformed_points(depth1, Tc1w, Tc2w, intrinsic1, intrinsic2)
        mask1 = rearrange(trans_points1[:, :, :, 2:3, 0] > 0, 'b h w t -> b t h w')

        trans_coordinates = trans_points1[:, :, :, :2, 0] / trans_points1[:, :, :, 2:3, 0]
        trans_depth1 = trans_points1[:, :, :, 2, 0]
        grid = self.create_grid(b, h, w).to(trans_coordinates)
        flow12 = trans_coordinates.permute(0, 3, 1, 2) - grid

        warped_frame2, mask2 = self.bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)
        return warped_frame2, mask2, None, flow12

    def compute_transformed_points(self, depth1, Tc1w, Tc2w, intrinsic1, intrinsic2):
        b, _, h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        Tc2c1 = torch.bmm(Tc2w, torch.linalg.inv(Tc1w))

        x1d = torch.arange(0, w)[None]
        y1d = torch.arange(0, h)[:, None]
        x2d = x1d.repeat([h, 1]).to(depth1)
        y2d = y1d.repeat([1, w]).to(depth1)
        ones_2d = torch.ones(size=(h, w)).to(depth1)
        ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]

        intrinsic1_inv = torch.linalg.inv(intrinsic1)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]
        intrinsic2_4d = intrinsic2[:, None, None]
        depth_4d = depth1[:, 0][:, :, :, None, None]
        trans_4d = Tc2c1[:, None, None]

        unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)
        world_points = depth_4d * unnormalized_pos
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)
        trans_world = trans_world_homo[:, :, :, :3]
        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)
        return trans_norm_points

    def bilinear_splatting(self, frame1, mask1, depth1, flow12, flow12_mask, is_image=False):
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame1)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1),
        ], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1),
        ], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1),
        ], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = torch.moveaxis(prox_weight_nw * mask1 * flow12_mask / depth_weights.unsqueeze(1), [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw * mask1 * flow12_mask / depth_weights.unsqueeze(1), [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne * mask1 * flow12_mask / depth_weights.unsqueeze(1), [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se * mask1 * flow12_mask / depth_weights.unsqueeze(1), [0, 1, 2, 3], [0, 3, 1, 2])

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).to(frame1)
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(frame1)

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), frame1_cl * weight_nw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]), frame1_cl * weight_sw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]), frame1_cl * weight_ne, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]), frame1_cl * weight_se, accumulate=True)

        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), weight_nw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]), weight_sw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]), weight_ne, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]), weight_se, accumulate=True)

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        mask = cropped_weights > 0
        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights, zero_tensor)
        mask2 = mask.to(frame1)

        if is_image:
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
        return warped_frame2, mask2

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid
