import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model import FairseqAudioModel
from data import MspPodcastMultiTask
from utils import set_seed, CCC


def get_data_loaders(opts, test_speaker):
    test_csv_path = os.path.join(opts.data_root, opts.data, 'labels/per', test_speaker, 'test.csv')
    test_set = MspPodcastMultiTask(test_csv_path, opts)
    test_loader = DataLoader(test_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)

    return test_loader


global_gt_list, global_pred_list = [[], [], []], [[], [], []]
train_speaker_list = ['3', '85', '415', '1', '115', '163', '31', '40', '41', '25', '48', '101', '100', '285', '286', '273', '479', '227', '228', '15', '89', '275', '274', '38', '135', '82', '355', '137', '13', '284', '74', '84', '83', '10', '12', '126', '57', '58', '468', '118', '117', '70', '71', '254', '122', '88', '119', '61', '62', '147', '64', '65', '72', '111', '77', '78', '374', '67', '76', '75', '123', '144', '401', '112', '108', '109', '251', '107', '139', '215', '216', '106', '105', '183', '182', '171', '140', '138', '211', '489', '277', '223', '154', '234', '313', '305', '158', '236', '238', '245', '352', '294', '430', '249', '371', '472', '266', '293', '299', '267', '268', '269', '302', '272', '289', '292', '345', '344', '600', '487', '488', '686', '758', '744', '794', '759', '760', '761', '752', '753', '756', '754', '755', '774', '863', '865', '806', '798', '800', '803', '804', '846', '858', '904', '853', '854', '849', '850', '937', '945', '936', '979', '884', '901', '867', '868', '916', '882', '924', '970', '870', '872', '874', '876', '877', '910', '905', '879', '880', '898', '892', '923', '976', '968', '928', '933', '929', '930', '931', '917', '919']
val_speaker_list = ['127', '128', '26', '43', '45', '314', '51', '54', '97', '169', '98', '167', '110', '124', '146', '116', '184', '451', '679', '149', '210', '471', '350', '235', '237', '270', '288', '287', '315', '335', '316', '743', '733', '749', '750', '762', '764', '765', '772', '766', '135']
test_speaker_list = ['30', '39', '2', '33', '32', '34', '35', '36', '42', '5', '4', '8', '148', '7', '9', '23', '22', '132', '53', '133', '16', '17', '18', '59', '63', '60', '19', '20', '94', '81', '80', '143', '27', '28', '29', '95', '96', '170', '141', '150', '164', '134', '130', '131', '160', '159', '151', '152', '162', '155']

f = open('./data/MSP-Podcast/per/label_distribution.txt', 'r')
label_distribution = json.load(f)
# f = open('./data/MSP-Podcast/per/speaker_rank_unsupervised_{}.txt'.format('base'), 'r')
f = open('./data/MSP-Podcast/per/speaker_rank.txt', 'r')
speaker_rank = json.load(f)

def test_model(model, data_loader, test_speaker, device, opts):
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
            gt_list[i] = np.array(gt_list[i])
            pred_list[i] = np.array(pred_list[i])

            similar_train_speakers = speaker_rank[test_speaker][i]
            pred_mean, pred_std = 0., 0.
            for train_speaker in similar_train_speakers[:opts.top_k]:
                pred_mean += label_distribution[train_speaker][i]
                pred_std += label_distribution[train_speaker][i+3]
            pred_mean /= opts.top_k
            pred_std /= opts.top_k

            pred_list[i] = (pred_list[i] - np.mean(pred_list[i])) / np.std(pred_list[i])
            pred_list[i] = pred_list[i] * pred_std + pred_mean

            # if opts.p != 0:
            #     if opts.p < 1:
            #         sample_gt = np.random.choice(gt_list[i], int(opts.p*len(gt_list[i])), replace=False)
            #     else:
            #         sample_gt = gt_list[i]
            #     pred_list[i] = (pred_list[i] - np.mean(pred_list[i])) / np.std(pred_list[i]) * np.std(sample_gt) + np.mean(sample_gt)

            ccc_score.append(CCC(gt_list[i], pred_list[i]))

            global_gt_list[i].append(gt_list[i])
            global_pred_list[i].append(pred_list[i])

        return ccc_score


def main(opts):
    # load checkpoint
    device = torch.device('cuda:0')
    model = FairseqAudioModel(opts, device).to(device)
    ckpt_path = os.path.join(opts.ckpt_path, opts.data, 'hubert_{}_multitask_{}.pt'.format(opts.backbone, opts.seed))
    model.load_state_dict(torch.load(ckpt_path))

    # test model
    test_speaker_list = ['134', '150', '23', '7', '133', '94', '170', '148', '80', '4', '19', '20', '2', '143', '5', '130', '60', '28', '131', '132', '36', '141', '8', '39', '33', '16', '162', '164', '30', '9', '18', '27', '22', '155', '17', '32', '160', '42', '59', '159', '152', '151', '81', '63', '35', '53', '95', '34', '29', '96']
    test_ccc_list = [[], [], []]
    print('Speaker, Arousal, Valence, Dominance')
    for test_speaker in test_speaker_list:    
        test_loader = get_data_loaders(opts, test_speaker)
        test_ccc = test_model(model, test_loader, test_speaker, device, opts)
        for i in range(3):
            test_ccc_list[i].append(test_ccc[i])
        print('{}, {:.3f}, {:.3f}, {:.3f}'.format(test_speaker, test_ccc[0], test_ccc[1], test_ccc[2]))

    test_ccc_list = np.array(test_ccc_list)
    print('='*20)
    print('CCC Mean Arousal: {:.3f} Valence: {:.3f} Dominance: {:.3f}'.format(np.mean(test_ccc_list[0]), np.mean(test_ccc_list[1]), np.mean(test_ccc_list[2])))
    print('CCC Std Arousal: {:.3f} Valence: {:.3f} Dominance: {:.3f}'.format(np.std(test_ccc_list[0]), np.std(test_ccc_list[1]), np.std(test_ccc_list[2])))

    total_test_ccc = []
    for i in range(3):
        global_gt_list[i] = np.concatenate(global_gt_list[i])
        global_pred_list[i] = np.concatenate(global_pred_list[i])
        test_ccc = CCC(global_gt_list[i], global_pred_list[i])
        total_test_ccc.append(test_ccc)
    print('='*20)
    print('Total CCC Arousal: {:.3f} Valence: {:.3f} Dominance: {:.3f}'.format(total_test_ccc[0], total_test_ccc[1], total_test_ccc[2]))


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
    parser.add_argument('--p', type=float, default=0)
    parser.add_argument('--top_k', type=int, default=5)

    opts = parser.parse_args()

    if opts.backbone == 'base':
        opts.fairseq_ckpt_path = './checkpoints/hubert_base_ls960.pt'
    else:
        opts.fairseq_ckpt_path = './checkpoints/hubert_large_ll60k.pt'

    print(opts)
    set_seed(opts.seed)
    main(opts)
