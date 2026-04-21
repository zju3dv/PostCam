import torch
import numpy as np
import decord
from einops import rearrange
from scipy.interpolate import UnivariateSpline, interp1d

decord.bridge.set_bridge('torch')


def read_frames(path):
    vr = decord.VideoReader(uri=path, height=-1, width=-1)
    frames = vr.get_batch(range(len(vr)))
    frames = rearrange(frames, 'T H W C -> C T H W').contiguous()
    frames = frames.float() / 255.0
    frames = frames.permute(1, 0, 2, 3)  # t c h w
    return frames  # t c h w (0,1)


def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew


def sphere2pose(x_up_angle, y_left_angle, r, is_zoom=False):
    angle_y = np.deg2rad(y_left_angle)
    sin_value_y = np.sin(angle_y)
    cos_value_y = np.cos(angle_y)
    rot_mat_y = np.array([
        [cos_value_y, 0, sin_value_y],
        [0, 1, 0],
        [-sin_value_y, 0, cos_value_y],
    ])
    angle_x = np.deg2rad(x_up_angle)
    sin_value_x = np.sin(angle_x)
    cos_value_x = np.cos(angle_x)
    rot_mat_x = np.array([
        [1, 0, 0],
        [0, cos_value_x, sin_value_x],
        [0, -sin_value_x, cos_value_x],
    ])

    R = rot_mat_y @ rot_mat_x
    T = np.array([-r * cos_value_x * sin_value_y, -r * sin_value_x, r - r * cos_value_x * cos_value_y])
    if is_zoom:
        T = np.array([0, 0, r])

    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = T
    return c2w  # 4x4


def generate_traj_txt(x_up_angles, y_left_angles, r, r_zoom, frame):
    """Generate camera trajectory from angle/radius parameters (COLMAP coordinate)."""
    if len(x_up_angles) > 3:
        x_up_angles = txt_interpolation(x_up_angles, frame, mode='smooth')
    else:
        x_up_angles = txt_interpolation(x_up_angles, frame, mode='linear')

    if len(y_left_angles) > 3:
        y_left_angles = txt_interpolation(y_left_angles, frame, mode='smooth')
    else:
        y_left_angles = txt_interpolation(y_left_angles, frame, mode='linear')

    if len(r) > 3:
        rs = txt_interpolation(r, frame, mode='smooth')
    else:
        rs = txt_interpolation(r, frame, mode='linear')

    if len(r_zoom) > 3:
        r_zooms = txt_interpolation(r_zoom, frame, mode='smooth')
    else:
        r_zooms = txt_interpolation(r_zoom, frame, mode='linear')

    c2ws_list = []
    is_zoom = all(x == 0 for x in x_up_angles) and all(y == 0 for y in y_left_angles)
    is_not_y = all(y == 0 for y in y_left_angles)
    for x_up_angle, y_left_angle, r_val, r_zoom_val in zip(x_up_angles, y_left_angles, rs, r_zooms):
        if is_not_y:
            c2w_new = sphere2pose(
                np.float32(x_up_angle), np.float32(y_left_angle), np.float32(r_zoom_val),
                is_zoom=is_zoom
            )
        else:
            c2w_new = sphere2pose(
                np.float32(x_up_angle), np.float32(y_left_angle), np.float32(r_val),
                is_zoom=is_zoom
            )
        c2ws_list.append(c2w_new)
    c2ws = np.stack(c2ws_list, axis=0)  # N, 4, 4
    return c2ws  # Twc
