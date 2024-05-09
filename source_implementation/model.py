import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import librosa
import numpy as np
import soundfile as sf

import fairseq
from transformers import Wav2Vec2FeatureExtractor, HubertModel

class AudioModel(nn.Module):
    def __init__(self, opts, device):
        super().__init__()

        self.sample_rate = opts.sample_rate
        self.max_audio_len = opts.max_audio_len
        self.device = device

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960')
        self.encoder = HubertModel.from_pretrained('facebook/hubert-base-ls960')
        self.interpreter = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(32, opts.num_labels)
        )

    def preprocess(self, x):
        audio_list = []
        for file_path in x:
            data, sample_rate = librosa.load(file_path, sr=None, mono=True)
            if len(data) > self.max_audio_len:
                data = data[-self.max_audio_len:]
            audio_list.append(data)

        feats = self.feature_extractor(audio_list, sampling_rate=self.sample_rate,
                                       return_tensors='pt', padding=True)
        input_values = feats['input_values'].to(self.device)

        return input_values

    def forward(self, audio_paths):
        hidden_states = self.preprocess(audio_paths)
        outputs = self.encoder(hidden_states).last_hidden_state[:, 0, :]
        outputs = self.interpreter(outputs)

        return outputs


class FairseqAudioModel(nn.Module):
    def __init__(self, opts, device):
        super().__init__()

        self.sample_rate = opts.sample_rate
        self.max_audio_len = opts.max_audio_len
        self.device = device

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([opts.fairseq_ckpt_path])
        self.encoder = model[0]
        self.normalize = cfg.task.normalize
        self.encoder.feature_grad_mult = 0.0
        self.encoder.encoder.layerdrop = 0.0
        if opts.backbone == 'base':
            hidden_dim = 768
        else:
            hidden_dim = 1024
        self.interpreter = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(32, opts.num_labels)
        )

    def preprocess(self, x):
        wavs = []
        for file_path in x:
            data, _ = sf.read(file_path)
            if len(data) > self.max_audio_len:
                data = data[-self.max_audio_len:]
            wavs.append(torch.FloatTensor(data).to(self.device))

        if self.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(self.device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(self.device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, _ = self.encoder.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )

        return features

    def forward(self, audio_paths):
        hidden_states = self.preprocess(audio_paths).mean(dim=1)
        outputs = self.interpreter(hidden_states)

        return outputs

    def extract_features(self, audio_paths):
        hidden_states = self.preprocess(audio_paths).mean(dim=1)

        return hidden_states

class FairseqAudioModel_f(nn.Module):
    def __init__(self, opts, device):
        super().__init__()

        self.sample_rate = opts.sample_rate
        self.max_audio_len = opts.max_audio_len
        self.device = device

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([opts.fairseq_ckpt_path])
        self.encoder = model[0]
        self.normalize = cfg.task.normalize
        self.encoder.feature_grad_mult = 0.0
        self.encoder.encoder.layerdrop = 0.0
        if opts.backbone == 'base':
            hidden_dim = 768
        else:
            hidden_dim = 1024
        self.interpreter_f = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(32, opts.num_labels)
        )

    def preprocess(self, x):
        wavs = []
        for file_path in x:
            data, _ = sf.read(file_path)
            if len(data) > self.max_audio_len:
                data = data[-self.max_audio_len:]
            wavs.append(torch.FloatTensor(data).to(self.device))

        if self.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(self.device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(self.device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, _ = self.encoder.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )

        return features

    def forward(self, audio_paths):
        hidden_states = self.preprocess(audio_paths).mean(dim=1)
        outputs = self.interpreter(hidden_states)

        return outputs

    def extract_features(self, audio_paths):
        hidden_states = self.preprocess(audio_paths).mean(dim=1)

        return hidden_states
