import torch
import os
import random
import json

random.seed(42)

from dataset.test_dataset import TestDataset


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_paths, dataset_types, use_saved_latent, sample_size, sample_stride, sample_n_frames, cam_idx=1, traj_txt_path=None):
        self.dataset = {}
        for metadata_path, dataset_type in zip(metadata_paths, dataset_types):
            if dataset_type in ['test']:
                metadata_list = json.load(open(metadata_path, 'r'))
                for metadata in metadata_list:
                    metadata['dataset_type'] = dataset_type
                    self.dataset[metadata['video_path']] = metadata
            else:
                raise ValueError(f'Invalid dataset type: {dataset_type}')

        self.test_dataset = TestDataset(sample_n_frames, sample_stride, sample_size, cam_idx=cam_idx, traj_txt_path=traj_txt_path)
        self.use_saved_latent = use_saved_latent
        print(f"{len(self.dataset)} samples in metadata.")

    def __getitem__(self, index):
        while True:
            try:
                source_data_key = list(self.dataset.keys())[index]
                data = self.test_dataset.get_data(self.dataset[source_data_key])
                break
            except Exception as e:
                import traceback
                print("Error:", e)
                traceback.print_exc()
                index = random.randrange(len(self.dataset))
        return data

    def __len__(self):
        return len(self.dataset)
