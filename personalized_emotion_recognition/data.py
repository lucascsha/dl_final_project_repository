import os
import pandas as pd

import torch
from torch.utils.data import Dataset


class MspPodcast(Dataset):
    def __init__(self, csv_path, opts):
        self.file_list = pd.read_csv(csv_path)
        self.opts = opts
        self.audios = self.file_list['audio_name']
        self.labels = self.file_list[opts.label]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_path = os.path.join(self.opts.data_root, 'MSP-Podcast/audios', self.audios[index])
        label = self.labels[index]

        return audio_path, label


class MspPodcastMultiTask(Dataset):
    def __init__(self, csv_path, opts):
        self.file_list = pd.read_csv(csv_path)
        self.opts = opts
        self.audios = self.file_list['audio_name']
        self.arousal = self.file_list['arousal']
        self.valence = self.file_list['valence']
        self.dominance = self.file_list['dominance']

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_path = os.path.join(self.opts.data_root, 'MSP-Podcast/audios', self.audios[index])
        arousal = self.arousal[index]
        valence = self.valence[index]
        dominance = self.dominance[index]

        return audio_path, torch.Tensor([arousal, valence, dominance])
