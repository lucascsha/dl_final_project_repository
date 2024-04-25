#---------------------------
import os
import matplotlib.pyplot as plt; import seaborn as sns
import csv
import json
import numpy as np
import tensorflow as tf; 
from transformers import Wav2Vec2FeatureExtractor, HubertModel, Wav2Vec2Model
from librosa import resample; from librosa.feature import mfcc; from librosa.util import frame
import argparse

#---------------------------
#system configuration

def get_args():
    parser = argparse.ArgumentParser(description = 'Process audio files, organize data')
    parser.add_argument('--code_dir', type=str, default='/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/self_implementation', help='Directory where code implementation is stored')
    parser.add_argument('--root_dir', type = str, default='/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/cs1470_ser_fp_data', help='Directory where (raw, processed) data are stored')
    args, _ = parser.parse_known_args()

    parser.add_argument('--crema_dir', type = str, default = os.path.join(args.root_dir, 'crema-d_data/AudioWAV'), help = 'Directory where raw CREMA-D dataset WAV audio stored')
    parser.add_argument('--npy_dir', type=str, default= os.path.join(args.root_dir,'processed_data/npy_dir'), help= 'Directory to save .npy audio features')

    parser.add_argument('--csv_path', type=str, default= os.path.join(args.root_dir,'processed_data/dataset.csv'), help= 'Path for output CSV file of dataset')
    parser.add_argument('--json_path', type=str, default= os.path.join(args.root_dir,'processed_data/speaker_dict.json'), help='Path for JSON file mapping speaker ids to .npy audio feature paths')

    parser.add_argument('--featurizer_type', type = str, default = 'mfcc', help = 'Audio feature extraction method for calculating speaker features')
    parser.add_argument('--max_feature_frames', type=int, default=216, help='Maximum length of audio files after padding.')
    parser.add_argument('--sample_rate', type=int, default= 22050, help='Sample rate to which audio files will be resampled.')

    return parser.parse_args()

def get_featurizer(featurizer_type):
    if featurizer_type.lower() == 'mfcc':
        return mfcc
    else:
        print('unidentified featurizer type')
        return None
    

#---------------------------
#audio processing
def pad(mfcc_features, max_feature_frames):
    #pad all mfcc features to 216 frames
    mfcc_features = tf.convert_to_tensor(mfcc_features)
    pad_left = (max_feature_frames - len(mfcc_features))//2
    pad_right = max_feature_frames - len(mfcc_features) - pad_left
    mfcc_features = tf.pad(mfcc_features, [[pad_left, pad_right], [0,0]])
    return mfcc_features.numpy()


def process_audio_sample(path, args):
    max_feature_frames = args.max_feature_frames
    featurizer = get_featurizer(args.featurizer_type)
    sample_rate = args.sample_rate
    npy_dir = args.npy_dir

    #file operations
    npy_path = path.split('.')
    npy_path.remove('wav')
    npy_path = '.'.join(npy_path) +'.npy'
    npy_path = os.path.join(npy_dir,npy_path)

    #audio operations
    if not os.path.exists(npy_path):
        n_mfcc = 13
        n_ftt = 2048
        hop_length = 512
        
        raw_vec, s_r = tf.audio.decode_wav(tf.io.read_file(path))
        s_r = s_r.numpy()
        raw_vec = raw_vec.numpy(); raw_vec = np.reshape(raw_vec, (-1, ))
        raw_vec = resample(raw_vec, orig_sr = s_r, target_sr = sample_rate)
        
        mfcc_features = featurizer(y=raw_vec, sr=sample_rate, n_mfcc=n_mfcc, n_fft = n_ftt, hop_length = hop_length).T
        mfcc_features = pad(mfcc_features, max_feature_frames)

        np.save(file = npy_path, arr = mfcc_features)

    return npy_path



#---------------------------
#variable setup
def return_paths(crema_dir):
    return [file.name for file in os.scandir(crema_dir)]

def return_header():
    header = ['Path', 'SpeakerID', 'Sentence', 'Emotion_Happy', 'Emotion_Angry', 'Emotion_Fearful','Emotion_Disgusted' ,'Emotion_Neutral', 'Emotion_Sadness', 'Intensity']
    return header

def return_speaker_ids():
    speaker_lst = [i for i in range(1001, 1092)]
    return speaker_lst

#---------------------------
#label extraction

def ohe_helper(path):
    ohe_vec = ['0', '0', '0', '0', '0', '0']
    if 'HAP' in path:
        ohe_vec[0] = '1'
    if 'ANG' in path:
        ohe_vec[1] = '1'
    if 'FEA' in path:
        ohe_vec[2] = '1'
    if 'DIS' in path:
        ohe_vec[3] = '1'
    if 'NEU' in path:
        ohe_vec[4] = '1'
    if 'SAD' in path:
        ohe_vec[5] = '1'
    return ohe_vec

def intensity_to_num(path):
    if 'XX' in path:
        intensity = '<unk>'
    if 'LO' in path:
        intensity = '0'
    if 'MD' in path:
        intensity = '1'
    if 'HI' in path: 
        intensity = '2'
    return intensity


#---------------------------
#make files

def make_row(path): 
    row = path.split('_')
    row.remove(row[-1])

    id_and_sent = row[0:2]
    emotions = ohe_helper(path)
    intensity = intensity_to_num(path)

    return [path] + id_and_sent + emotions + [intensity]

def create_csv(args):
    if not os.path.exists(args.csv_path):
        header = return_header()
        paths = return_paths(args.crema_dir)

        with open(args.csv_path, mode = 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for path in paths[0:10]:
                writer.writerow(make_row(path))

def build_speaker_dict(args):
    speaker_lst = return_speaker_ids()
    speaker_dict = {str(speaker_id) : [] for speaker_id in speaker_lst}
    with open(args.csv_path, mode = 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_data = [row[key] for key in row if key != 'SpeakerID']
            speaker_dict[row['SpeakerID']].append(row_data)
            (speaker_dict[row['SpeakerID']])[0][0] = process_audio_sample(row['Path'], args)
    return speaker_dict

def create_json(args, speaker_dict):
    if not os.path.exists(args.json_path):
        with open(args.json_path, 'w') as f:
            json.dump(speaker_dict, f, indent = 4)
       
#---------------------------
#preprocess
            
def main():
    args = get_args()
    os.chdir(args.crema_dir)
    
    create_csv(args)
    speaker_dict = build_speaker_dict(args)
    create_json(args, speaker_dict)

if __name__ == '__main__': 
    main()
   
    


    

    

                





