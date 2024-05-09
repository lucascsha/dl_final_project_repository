import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model import FairseqAudioModel
from data import MspPodcastMultiTask
from utils import set_seed


def get_data_loaders(opts):
    train_csv_path = os.path.join(opts.data_root, opts.data, 'labels/train.csv')
    val_csv_path = os.path.join(opts.data_root, opts.data, 'labels/val.csv')
    test_csv_path = os.path.join(opts.data_root, opts.data, 'labels/test.csv')

    train_set = MspPodcastMultiTask(train_csv_path, opts)
    val_set = MspPodcastMultiTask(val_csv_path, opts)
    test_set = MspPodcastMultiTask(test_csv_path, opts)

    train_loader = DataLoader(train_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)

    return train_loader, val_loader, test_loader


def extract_features(model, data_loader, device):
    os.makedirs('./data/MSP-Podcast/fine_tuned_hubert_{}_features/'.format(opts.backbone), exist_ok=True)
    model.eval()
    with torch.no_grad():
        for (audio_paths, _) in tqdm(data_loader):
            batch_size = len(audio_paths)
            audio_features = model.extract_features(audio_paths).reshape(batch_size, -1).detach().cpu().numpy()

            for i in range(batch_size):
                outfile = os.path.join('./data/MSP-Podcast/fine_tuned_hubert_{}_features/'.format(opts.backbone), audio_paths[i].split('/')[-1][:-4]+'.npy') # 768
                np.save(outfile, audio_features[i])


def main(opts):
    # load checkpoint
    device = torch.device('cuda:0')
    model = FairseqAudioModel(opts, device).to(device)
    ckpt_path = os.path.join(opts.ckpt_path, opts.data, 'hubert_{}_multitask_{}.pt'.format(opts.backbone, opts.seed))
    model.load_state_dict(torch.load(ckpt_path))

    # extract features
    train_loader, val_loader, test_loader = get_data_loaders(opts)
    extract_features(model, train_loader, device)
    extract_features(model, val_loader, device)
    extract_features(model, test_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=100)

    # storage
    parser.add_argument('--data_root', type=str, default='/home/ICT2000/yin/per_er/data/')
    parser.add_argument('--data', type=str, default='MSP-Podcast')
    parser.add_argument('--num_labels', type=int, default=3)
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints')

    # data
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_audio_len', type=int, default=250000)
    parser.add_argument('--sample_rate', type=int, default=16000)

    # training
    parser.add_argument('--backbone', type=str, default='base')
    parser.add_argument('--batch_size', type=int, default=32)

    opts = parser.parse_args()

    if opts.backbone == 'base':
        opts.fairseq_ckpt_path = './checkpoints/hubert_base_ls960.pt'
    else:
        opts.fairseq_ckpt_path = './checkpoints/hubert_large_ll60k.pt'

    print(opts)
    set_seed(opts.seed)
    main(opts)
