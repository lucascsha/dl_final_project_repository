import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from model import FairseqAudioModel
from data import MspPodcast
from utils import set_seed, CCC, CCC_loss


def get_data_loaders(opts):
    print('Data loading ...')

    train_csv_path = os.path.join(opts.data_root, opts.data, 'per_labels', opts.train_csv+'.csv')
    val_csv_path = os.path.join(opts.data_root, opts.data, 'per_labels', opts.val_csv+'.csv')
    test_csv_path = os.path.join(opts.data_root, opts.data, 'per_labels', opts.test_csv+'.csv')

    train_set = MspPodcast(train_csv_path, opts)
    dev_set = MspPodcast(val_csv_path, opts)
    test_set = MspPodcast(test_csv_path, opts)

    train_loader = DataLoader(train_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(dev_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)

    return train_loader, val_loader, test_loader


def get_speaker_data_loader(test_speaker, opts):
    if opts.test_csv == 'test_b':
        test_csv_path = os.path.join(opts.data_root, opts.data, 'per_labels/per_speaker_b', test_speaker+'.csv')
    else:
        test_csv_path = os.path.join(opts.data_root, opts.data, 'per_labels/per_speaker', test_speaker+'.csv')

    test_set = MspPodcast(test_csv_path, opts)
    test_loader = DataLoader(test_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)

    return test_loader


def test_model(model, data_loader, device):
    gt_list, pred_list = [], []
    model.eval()
    with torch.no_grad():
        for (audio_paths, labels) in tqdm(data_loader):
            batch_size = len(audio_paths)
            labels = labels.float().to(device)
            preds = model(audio_paths).reshape(batch_size)
            gt_list += labels.tolist()
            pred_list += preds.tolist()

        ccc_score = CCC(np.array(gt_list), np.array(pred_list))

        return ccc_score


def test_model_per_speakers(model, test_speaker_list, device, opts):
    ccc_score_list = []
    for test_speaker in tqdm(test_speaker_list):
        test_loader = get_speaker_data_loader(test_speaker, opts)
        ccc_score = test_model_per_speaker(model, test_loader, test_speaker, device, opts)
        ccc_score_list.append(ccc_score)
    ccc_score_list = np.array(ccc_score_list)
    ccc_score_list = np.mean(ccc_score_list, axis=0)

    print('Per speaker evaluation')
    print('Test {} CCC: {:.3f}'.format(opts.label, ccc_score_list))

    opts.total_gt = np.array(opts.total_gt)
    opts.total_pred = np.array(opts.total_pred)
    ccc_score_list = CCC(opts.total_gt, opts.total_pred)
    print('Overall evaluation')
    print('Test {} CCC: {:.3f}'.format(opts.label, ccc_score_list))


def test_model_per_speaker(model, test_loader, test_speaker, device, opts):
    gt_list, pred_list = [], []
    model.eval()
    with torch.no_grad():
        for (audio_paths, labels) in test_loader:
            batch_size = len(audio_paths)
            labels = labels.float()
            preds = model(audio_paths).reshape(batch_size)
            
            gt_list += labels.tolist()
            pred_list += preds.tolist()

            opts.total_gt += labels.tolist()
            opts.total_pred += preds.tolist()

        gt_list = np.array(gt_list)
        pred_list = np.array(pred_list)
        ccc_score = CCC(gt_list, pred_list)

        print(test_speaker, ccc_score)

        return ccc_score


def main(opts):
    # data loader
    train_loader, val_loader, _ = get_data_loaders(opts)

    # model and optimizer
    device = torch.device('cuda:0')
    if opts.backbone == 'base':
        opts.fairseq_ckpt_path = './checkpoints_per/hubert_base_ls960.pt'
    else:
        opts.fairseq_ckpt_path = './checkpoints_per/hubert_large_ll60k.pt'
    model = FairseqAudioModel(opts, device).to(device)
    if not opts.pretrain_ckpt_name == 'none':
        pretrain_ckpt_path = os.path.join(opts.ckpt_path, opts.data, opts.label, '{}_hubert_{}_{}.pt'.format(opts.pretrain_ckpt_name, opts.backbone, opts.seed))
        model.load_state_dict(torch.load(pretrain_ckpt_path))

    optimizer = optim.AdamW(model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay)
    criterion = CCC_loss

    # train and test
    best_val = 0.
    patience = opts.patience

    for epoch in range(1, opts.num_epochs+1):
        print('Epoch: %d/%d' % (epoch, opts.num_epochs))

        # train model
        model.train()
        train_loss = []
        for (audio_paths, labels) in tqdm(train_loader):
            optimizer.zero_grad()

            batch_size = len(audio_paths)
            labels = labels.float().to(device)
            preds = model(audio_paths).reshape(batch_size)

            loss = criterion(preds, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss.append(loss.item())

        train_loss_avg = sum(train_loss) / len(train_loss)

        # validate model
        val_ccc = test_model(model, val_loader, device)
        print('Train loss: {:.4f} Val {} CCC: {:.3f}'.format(train_loss_avg, opts.label, val_ccc))

        if val_ccc > best_val:
            patience = opts.patience
            best_val = val_ccc
            os.makedirs(os.path.join(opts.ckpt_path, opts.data, opts.label), exist_ok=True)
            save_ckpt_path = os.path.join(opts.ckpt_path, opts.data, opts.label, '{}_hubert_{}_{}.pt'.format(opts.ckpt_name, opts.backbone, opts.seed))
            torch.save(model.state_dict(), save_ckpt_path)
            print('save to:', save_ckpt_path)
        else:
            patience -= 1
            if patience == 0:
                break

    # load the best model
    save_ckpt_path = os.path.join(opts.ckpt_path, opts.data, opts.label, '{}_hubert_{}_{}.pt'.format(opts.ckpt_name, opts.backbone, opts.seed))
    model.load_state_dict(torch.load(save_ckpt_path))

    # test model
    opts.total_gt, opts.total_pred = [], []
    if opts.test_csv == 'test_c':
        test_speaker_list = ['747', '505', '504', '1106', '1107', '1105', '1108', '1110', '1109', '1114', '1112', '1113', '1115', '1121', '1122', '1129', '1123', '1126', '1125', '1127', '1130', '1116', '1169', '1168', '1170', '1172', '1171', '1173', '1117', '1118', '1133', '1132', '1131', '1242', '1136', '1135', '1134', '1138', '1137', '1087', '1088', '1089', '1092', '1091', '1090', '746', '1177', '1176', '1141', '1140', '1155', '1154', '1156', '1139', '1259', '1158', '1157', '1159', '1174', '1200', '1198', '1196', '1197']
    else:
        test_speaker_list = ['134', '150', '23', '7', '133', '94', '170', '148', '80', '4', '19', '20', '2', '143', '5', '130', '60', '28', '131', '132', '36', '141', '8', '39', '33', '16', '162', '164', '30', '9', '18', '27', '22', '155', '17', '32', '160', '42', '59', '159', '152', '151', '81', '63', '35', '53', '95', '34', '29', '96']

    test_model_per_speakers(model, test_speaker_list, device, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=100)

    # storage
    parser.add_argument('--data_root', type=str, default='/home/ICT2000/yin/per_er/data/')
    parser.add_argument('--data', type=str, default='MSP-Podcast')
    parser.add_argument('--label', type=str, default='valence', choices=['arousal', 'valence', 'dominance'])
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints_per')

    # data
    parser.add_argument('--train_csv', type=str, default='train')
    parser.add_argument('--val_csv', type=str, default='val')
    parser.add_argument('--test_csv', type=str, default='test_b')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_audio_len', type=int, default=250000)
    parser.add_argument('--sample_rate', type=int, default=16000)

    # training
    parser.add_argument('--backbone', type=str, default='base')
    parser.add_argument('--pretrain_ckpt_name', type=str, default='none')
    parser.add_argument('--ckpt_name', type=str, default='finetune')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3, help='early stopping')

    opts = parser.parse_args()

    print(opts)
    set_seed(opts.seed)
    main(opts)
