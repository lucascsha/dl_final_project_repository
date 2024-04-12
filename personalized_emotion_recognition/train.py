import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from model import FairseqAudioModel
from data import MspPodcastMultiTask
from utils import set_seed, CCC, CCC_loss


def get_data_loaders(opts):
    print('Data loading ...')

    train_csv_path = os.path.join(opts.data_root, opts.data, 'per_labels', opts.train_csv+'.csv')
    val_csv_path = os.path.join(opts.data_root, opts.data, 'per_labels', opts.val_csv+'.csv')
    test_csv_path = os.path.join(opts.data_root, opts.data, 'per_labels', opts.test_csv+'.csv')

    train_set = MspPodcastMultiTask(train_csv_path, opts)
    dev_set = MspPodcastMultiTask(val_csv_path, opts)
    test_set = MspPodcastMultiTask(test_csv_path, opts)

    train_loader = DataLoader(train_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(dev_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)

    return train_loader, val_loader, test_loader


def get_speaker_data_loader(test_speaker, opts):
    if opts.test_csv == 'test':
        test_csv_path = os.path.join(opts.data_root, opts.data, 'per_labels/per_speaker', test_speaker+'.csv')
    else:
        test_csv_path = os.path.join(opts.data_root, opts.data, 'per_labels/per_speaker_b', test_speaker+'.csv')

    test_set = MspPodcastMultiTask(test_csv_path, opts)
    test_loader = DataLoader(test_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)

    return test_loader


def test_model(model, data_loader, device):
    gt_list, pred_list = [[], [], []], [[], [], []]
    model.eval()
    with torch.no_grad():
        for (audio_paths, labels) in tqdm(data_loader):
            batch_size = len(audio_paths)
            labels = labels.float().to(device)
            preds = model(audio_paths).reshape(batch_size, 3)
            for i in range(3):
                gt_list[i] += labels[:,i].tolist()
                pred_list[i] += preds[:,i].tolist()

        ccc_score = []
        for i in range(3):
            ccc_score.append(CCC(np.array(gt_list[i]), np.array(pred_list[i])))

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
    print('Test Arousal CCC: {:.3f} Valence CCC: {:.3f} Dominance CCC: {:.3f}'.format(ccc_score_list[0], ccc_score_list[1], ccc_score_list[2]))

    opts.total_gt = np.array(opts.total_gt)
    opts.total_pred = np.array(opts.total_pred)
    ccc_score_list = []
    for i in range(3):
        ccc_score_list.append(CCC(opts.total_gt[i], opts.total_pred[i]))
    print('Overall evaluation')
    print('Test Arousal CCC: {:.3f} Valence CCC: {:.3f} Dominance CCC: {:.3f}'.format(ccc_score_list[0], ccc_score_list[1], ccc_score_list[2]))


def test_model_per_speaker(model, test_loader, test_speaker, device, opts):
    gt_list, pred_list = [[], [], []], [[], [], []]
    model.eval()
    with torch.no_grad():
        for (audio_paths, labels) in test_loader:
            batch_size = len(audio_paths)
            labels = labels.float()
            preds = model(audio_paths).reshape(batch_size, 3)
            for i in range(3):
                gt_list[i] += labels[:,i].tolist()
                pred_list[i] += preds[:,i].tolist()

                opts.total_gt[i] += labels[:,i].tolist()
                opts.total_pred[i] += preds[:,i].tolist()

        ccc_score = []
        for i in range(3):
            gt_list[i] = np.array(gt_list[i])
            pred_list[i] = np.array(pred_list[i])
            ccc_score.append(CCC(gt_list[i], pred_list[i]))

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
        pretrain_ckpt_path = os.path.join(opts.ckpt_path, opts.data, '{}_hubert_{}_{}.pt'.format(opts.pretrain_ckpt_name, opts.backbone, opts.seed))
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
            preds = model(audio_paths).reshape(batch_size, 3)

            loss = 0.
            for i in range(3):
                loss += criterion(preds[:,i], labels[:,i])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss.append(loss.item())

        train_loss_avg = sum(train_loss) / len(train_loss)

        # validate model
        val_ccc = test_model(model, val_loader, device)
        print('Train loss: {:.4f} Val Arousal CCC: {:.3f} Valence CCC: {:.3f} Dominance CCC: {:.3f}'.format(train_loss_avg, val_ccc[0], val_ccc[1], val_ccc[2]))

        if sum(val_ccc)/len(val_ccc) > best_val:
            patience = opts.patience
            best_val = sum(val_ccc)/len(val_ccc)
            os.makedirs(os.path.join(opts.ckpt_path, opts.data), exist_ok=True)
            save_ckpt_path = os.path.join(opts.ckpt_path, opts.data, '{}_hubert_{}_{}.pt'.format(opts.ckpt_name, opts.backbone, opts.seed))
            torch.save(model.state_dict(), save_ckpt_path)
            print('save to:', save_ckpt_path)
        else:
            patience -= 1
            if patience == 0:
                break

    # load the best model
    save_ckpt_path = os.path.join(opts.ckpt_path, opts.data, '{}_hubert_{}_{}.pt'.format(opts.ckpt_name, opts.backbone, opts.seed))
    model.load_state_dict(torch.load(save_ckpt_path))

    # test model
    opts.total_gt, opts.total_pred = [[], [], []], [[], [], []]
    test_speaker_list = ['134', '150', '23', '7', '133', '94', '170', '148', '80', '4', '19', '20', '2', '143', '5', '130', '60', '28', '131', '132', '36', '141', '8', '39', '33', '16', '162', '164', '30', '9', '18', '27', '22', '155', '17', '32', '160', '42', '59', '159', '152', '151', '81', '63', '35', '53', '95', '34', '29', '96']
    test_ccc = test_model_per_speakers(model, test_speaker_list, device, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=100)

    # storage
    parser.add_argument('--data_root', type=str, default='/home/ICT2000/yin/per_er/data/')
    parser.add_argument('--data', type=str, default='MSP-Podcast')
    parser.add_argument('--num_labels', type=int, default=3)
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints_per')

    # data
    parser.add_argument('--train_csv', type=str, default='train')
    parser.add_argument('--val_csv', type=str, default='val')
    parser.add_argument('--test_csv', type=str, default='test')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_audio_len', type=int, default=250000)
    parser.add_argument('--sample_rate', type=int, default=16000)

    # training
    parser.add_argument('--backbone', type=str, default='large')
    parser.add_argument('--pretrain_ckpt_name', type=str, default='none')
    parser.add_argument('--ckpt_name', type=str, default='finetune')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10, help='early stopping')

    opts = parser.parse_args()

    print(opts)
    set_seed(opts.seed)
    main(opts)
