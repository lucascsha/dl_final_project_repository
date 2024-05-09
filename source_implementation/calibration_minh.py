import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model import FairseqAudioModel
from data import MspPodcast
from utils import set_seed, CCC


def get_speaker_data_loader(test_speaker, opts):
    test_csv_path = os.path.join(opts.data_root, opts.data, 'per_labels/per_speaker_b', test_speaker+'.csv')
    test_set = MspPodcast(test_csv_path, opts)
    test_loader = DataLoader(test_set, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)

    return test_loader


def get_speaker_feature(model, opts):
    print('Extracting speaker features ...')

    model.eval()
    with torch.no_grad():
        opts.speaker_feature_dict = {}
        for speaker in tqdm(opts.speaker_list):
            os.makedirs('/shares/perception-temp/yufeng/per_er/data/MSP-Podcast/{}'.format(opts.features), exist_ok=True)
            speaker_feature_path = '/shares/perception-temp/yufeng/per_er/data/MSP-Podcast/{}/{}.npy'.format(opts.features, speaker)
            if not os.path.exists(speaker_feature_path):
                audio_paths = opts.speaker_dict[speaker]
                feature_list = []
                for i in range(0, len(audio_paths), 64):
                    feature = model.extract_features(audio_paths[i:i+64]).detach().cpu()
                    feature_list.append(feature)
                feature_list = torch.cat(feature_list, dim=0)
                speaker_feature = torch.mean(feature_list, dim=0)
                np.save(speaker_feature_path, speaker_feature.numpy())
            else:
                speaker_feature = np.load(speaker_feature_path)
                speaker_feature = torch.from_numpy(speaker_feature)
            opts.speaker_feature_dict[speaker] = speaker_feature.reshape(1024)


def test_model(model, test_speaker_list, device, opts):
    print('Testing ...')

    ccc_score_list = []
    opts.sim_dict = {}
    for test_speaker in tqdm(test_speaker_list):
        test_loader = get_speaker_data_loader(test_speaker, opts)
        ccc_score = test_model_per_speaker(model, test_loader, test_speaker, device, opts)
        ccc_score_list.append(ccc_score)
    ccc_score_list = np.array(ccc_score_list)

    print('Per speaker evaluation')
    print('Test {} CCC: {:.3f}'.format(opts.label, np.mean(ccc_score_list, axis=0)))
    print('Test {} CCC std: {:.3f}'.format(opts.label, np.std(ccc_score_list, axis=0)))

    opts.total_gt = np.array(opts.total_gt)
    opts.total_pred = np.array(opts.total_pred)
    ccc_score_list = CCC(opts.total_gt, opts.total_pred)
    print('Overall evaluation')
    print('Test {} CCC: {:.3f}'.format(opts.label, ccc_score_list))

    # file = open('rank_{}.txt'.format(opts.features), 'w')
    # json.dump(opts.sim_dict, file)


def test_model_per_speaker(model, test_loader, test_speaker, device, opts):
    gt_list, pred_list = [], []
    model.eval()
    with torch.no_grad():
        for (audio_paths, labels) in test_loader:
            labels = labels.float().to(device)
            preds = []
            for audio_path in audio_paths:
                preds.append(float(opts.predictions[audio_path]))
            gt_list += labels.tolist()
            pred_list += preds

        test_speaker_embedding = opts.speaker_feature_dict[test_speaker]
        sim_dict = {}
        for train_speaker in opts.train_speaker_list:
            train_speaker_embedding = opts.speaker_feature_dict[train_speaker]
            sim_dict[train_speaker] = F.cosine_similarity(test_speaker_embedding, train_speaker_embedding, dim=0)
        sorted_sim_dict = sorted(sim_dict, key=lambda x: sim_dict[x], reverse=True)
        opts.sim_dict[test_speaker] = sorted_sim_dict

        ccc_score = []
        gt_list = np.array(gt_list)
        pred_list = np.array(pred_list)

        confidence = 0.
        train_labels = []
        for train_speaker in sorted_sim_dict[:opts.top_k]:
            confidence += sim_dict[train_speaker]
            train_csv_path = os.path.join(opts.data_root, opts.data, 'per_labels/per_speaker', train_speaker+'.csv')
            train_list = pd.read_csv(train_csv_path)
            train_labels += train_list[opts.label].values.tolist()
        confidence /= opts.top_k

        pred_mean, pred_std = np.mean(np.array(train_labels)), np.std(np.array(train_labels))

        ori_mean, ori_std = np.mean(pred_list), np.std(pred_list)
        ori_pred_list = pred_list

        # if confidence < opts.threshold:
        #     pred_list = (pred_list - ori_mean) / ori_std
        #     pred_list = pred_list * pred_std + pred_mean
        #     # pred_list = pred_list - ori_mean + pred_mean
        #     # pred_list = (pred_list - ori_mean) / ori_std
        #     # pred_list = pred_list * pred_std + ori_mean

        opts.total_gt += gt_list.tolist()
        opts.total_pred += pred_list.tolist()

        ori_ccc_score = CCC(gt_list, ori_pred_list)
        ccc_score = CCC(gt_list, pred_list)

        # print('{}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(test_speaker, len(pred_list), confidence, ori_ccc_score, ccc_score, np.mean(gt_list), ori_mean, pred_mean, np.std(gt_list), ori_std, pred_std))

        return ccc_score


def main(opts):
    # prediction
    predictions = open('./checkpoints_minh/ptapt_{}_output.csv'.format(opts.label), 'r').readlines()
    opts.predictions = {}
    for row in predictions:
        row = row[:-1].split(',')
        name, _, pred = row[0], row[1], row[2]
        name = name.replace('/home/ICT2000/yin', '/shares/perception-temp/yufeng')
        opts.predictions[name] = pred

    # data
    label_distribution = open('/shares/perception-temp/yufeng/per_er/data/MSP-Podcast/per_labels/label_distribution.txt', 'r')
    opts.label_distribution = json.load(label_distribution)

    opts.train_speaker_list = ['3', '85', '415', '1', '115', '163', '31', '40', '41', '25', '48', '101', '100', '285', '286', '273', '479', '227', '228', '15', '89', '275', '274', '38', '135', '82', '355', '137', '13', '284', '74', '84', '83', '10', '12', '126', '57', '58', '468', '118', '117', '70', '71', '254', '122', '88', '119', '61', '62', '147', '64', '65', '72', '111', '77', '78', '374', '67', '76', '75', '123', '144', '401', '112', '108', '109', '251', '107', '139', '215', '216', '106', '105', '183', '182', '171', '140', '138', '211', '489', '277', '223', '154', '234', '313', '305', '158', '236', '238', '245', '352', '294', '430', '249', '371', '472', '266', '293', '299', '267', '268', '269', '302', '272', '289', '292', '345', '344', '600', '487', '488', '686', '758', '744', '794', '759', '760', '761', '752', '753', '756', '754', '755', '774', '863', '865', '806', '798', '800', '803', '804', '846', '858', '904', '853', '854', '849', '850', '937', '945', '936', '979', '884', '901', '867', '868', '916', '882', '924', '970', '870', '872', '874', '876', '877', '910', '905', '879', '880', '898', '892', '923', '976', '968', '928', '933', '929', '930', '931', '917', '919']
    opts.val_speaker_list = ['127', '128', '26', '43', '45', '314', '51', '54', '97', '169', '98', '167', '110', '124', '146', '116', '184', '451', '679', '149', '210', '471', '350', '235', '237', '270', '288', '287', '315', '335', '316', '743', '733', '749', '750', '762', '764', '765', '772', '766', '135']
    opts.test_speaker_list = ['134', '150', '23', '7', '133', '94', '170', '148', '80', '4', '19', '20', '2', '143', '5', '130', '60', '28', '131', '132', '36', '141', '8', '39', '33', '16', '162', '164', '30', '9', '18', '27', '22', '155', '17', '32', '160', '42', '59', '159', '152', '151', '81', '63', '35', '53', '95', '34', '29', '96']
    # opts.test_speaker_list = ['747', '505', '504', '1106', '1107', '1105', '1108', '1110', '1109', '1114', '1112', '1113', '1115', '1121', '1122', '1129', '1123', '1126', '1125', '1127', '1130', '1116', '1169', '1168', '1170', '1172', '1171', '1173', '1117', '1118', '1133', '1132', '1131', '1242', '1136', '1135', '1134', '1138', '1137', '1087', '1088', '1089', '1092', '1091', '1090', '746', '1177', '1176', '1141', '1140', '1155', '1154', '1156', '1139', '1259', '1158', '1157', '1159', '1174', '1200', '1198', '1196', '1197']
    opts.speaker_list = opts.train_speaker_list + opts.val_speaker_list + opts.test_speaker_list

    opts.speaker_dict = {}
    csv_file = open('/shares/perception-temp/yufeng/per_er/data/MSP-Podcast/per_labels/train.csv').readlines()[1:]
    csv_file += open('/shares/perception-temp/yufeng/per_er/data/MSP-Podcast/per_labels/val.csv').readlines()[1:]
    csv_file += open('/shares/perception-temp/yufeng/per_er/data/MSP-Podcast/per_labels/test_b.csv').readlines()[1:]
    for row in csv_file:
        speaker = row.split(',')[-2]
        if speaker in opts.speaker_list:
            if not speaker in opts.speaker_dict:
                opts.speaker_dict[speaker] = []
            opts.speaker_dict[speaker].append('./data/MSP-Podcast/audios/{}'.format(row.split(',')[0]))

    # model
    if opts.backbone == 'base':
        opts.fairseq_ckpt_path = '/shares/perception-temp/yufeng/per_er/checkpoints/hubert_base_ls960.pt'
    else:
        if opts.features == 'tapt':
            opts.fairseq_ckpt_path = '/home/ICT2000/mtran/hubert_large_tapt_podcast_10K/checkpoints/checkpoint_best.pt'
        else:
            opts.fairseq_ckpt_path = '/shares/perception-temp/yufeng/per_er/checkpoints/hubert_large_ll60k.pt'

    device = torch.device('cuda:0')
    model = FairseqAudioModel(opts, device).to(device)

    if opts.features == 'tapt_valence':
        ckpt_path = os.path.join('/shares/perception-temp/yufeng/per_er/checkpoints_minh/valence.pt')
        model.load_state_dict(torch.load(ckpt_path), strict=False)
    elif opts.features == 'tapt_arousal':
        ckpt_path = os.path.join('/shares/perception-temp/yufeng/per_er/checkpoints_minh/arousal.pt')
        model.load_state_dict(torch.load(ckpt_path), strict=False)
    elif opts.features == 'tapt_dominance':
        ckpt_path = os.path.join('/shares/perception-temp/yufeng/per_er/checkpoints_minh/dominance.pt')
        model.load_state_dict(torch.load(ckpt_path), strict=False)
    elif opts.features == 'finetune_hubert_large':
        ckpt_path = os.path.join('/shares/perception-temp/yufeng/per_er/checkpoints/MSP-Podcast/valence/hubert_large_100.pt')
        model.load_state_dict(torch.load(ckpt_path), strict=False)

    # extract speaker features
    get_speaker_feature(model, opts)

    # test model
    opts.total_gt, opts.total_pred = [], []
    test_model(model, opts.test_speaker_list, device, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=100)

    # storage
    parser.add_argument('--data_root', type=str, default='/shares/perception-temp/yufeng/per_er/data/')
    parser.add_argument('--data', type=str, default='MSP-Podcast')
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--label', type=str, default='valence', choices=['arousal', 'valence', 'dominance'])

    # data
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--max_audio_len', type=int, default=250000)
    parser.add_argument('--sample_rate', type=int, default=16000)

    # training
    parser.add_argument('--backbone', type=str, default='large')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=15)
    parser.add_argument('--threshold', type=float, default=0.985)
    parser.add_argument('--features', type=str, default='hubert_large')

    opts = parser.parse_args()

    print(opts)
    set_seed(opts.seed)
    main(opts)
