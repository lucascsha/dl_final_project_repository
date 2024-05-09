# The purpose of this file is to split the training pipeline into processing with HuBERT and the training of the fine-tuning head.
# This file encodes batches as dictionaries from file.name ==> audio_encoding and dumps them into a folder containing multiple JSONS to store them for later loading efficiently.
#----------------------------
from model import HubertWrapper
import argparse
import os
import tensorflow as tf
import json

#system configuration
#----------------------------
def get_args():
    parser = argparse.ArgumentParser(description = 'Process audio files, organize data')
    #directory and file paths
    parser.add_argument('--code_dir', type=str, default='/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/self_implementation', help='Directory where code implementation is stored')
    parser.add_argument('--root_dir', type = str, default='/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/cs1470_ser_fp_data', help='Directory where (raw, processed) data are stored')

    args, _ = parser.parse_known_args()
    parser.add_argument('--crema_dir', type = str, default = os.path.join(args.root_dir, 'crema-d_data/AudioWAV'), help = 'Directory where raw CREMA-D dataset WAV audio stored')
    parser.add_argument('--audio_dir', type=str, default= os.path.join(args.root_dir,'processed_data/audio_dir'), help= 'Directory where .npy audio vectors saved')
    parser.add_argument('--hubert_dir', type = str, default = os.path.join(args.root_dir, 'processed_data/hubert_dir_repadded'), help = 'Directory to store json files containing batches of path: audio encoding dictionaries')

    return parser.parse_args()

#set file paths 
#----------------------------
def save_sorted_file_paths(args):
    #file to store a consistent order of paths
    output_json = os.path.join(args.root_dir, 'processed_data/sorted_file_paths.json')
    paths = sorted([file.name for file in os.scandir(args.audio_dir)])
    if not os.path.exists(output_json):
        with open(output_json, 'w') as f:
            json.dump(paths, f, indent = 4)
    return output_json

#embed audio
#----------------------------
def embed_audio_batch(args, paths, start_slice, end_slice):
    model = HubertWrapper()

    #slice path:
    paths = paths[start_slice: end_slice]
    
    #determine path to store json file from start_slice/end_slice
    hubert_path = os.path.join(args.hubert_dir, f'path_encoding_{start_slice}-{end_slice-1}.json')
    
    #create and dump data
    with tf.device('/gpu:0'):
        path_encoding_dict = model(paths, dict())
    with open(hubert_path, 'a') as f:
        json.dump(path_encoding_dict, f, indent = 4)

def determine_new_slice(current_slices):
    current_slices = [json_file.split('-') for json_file in current_slices]
    last_indices = [last_index[-1] for last_index in current_slices]
    last_indices = [int(last_index.split('.')[0]) for last_index in last_indices]
    last_index = max(last_indices)

    start_index = last_index+1
    end_index = start_index+100
    return start_index, end_index

#testing:
#----------------------------

#test slicing algorithm:
def test_determine_new_slice():
    current_slices = ['path_encoding_0-99.json', 'path_encoding_100-199.json']
    print(determine_new_slice(current_slices))

#tests for dictionaries
def test_one_dictionary(dict_path):
    args = get_args()
    os.chdir(args.hubert_dir)
    with open(dict_path, 'r') as f:
        path_encoding_dict = json.load(f)
    assert len(path_encoding_dict.keys()) == 100

def test_two_dictionaries(path1, path2):
    args = get_args()
    os.chdir(args.hubert_dir)
    with open(path1, 'r') as f:
        path_encoding_dict1 = json.load(f)
    with open(path2, 'r') as f:
        path_encoding_dict2 = json.load(f)
    x = set(path_encoding_dict1.keys())
    y = set(path_encoding_dict2.keys())
    x.update(y)
    assert len(x) == 200

def test_n_dictionaries():
    args = get_args()
    os.chdir(args.hubert_dir)
    dicts = [file.name for file in os.scandir(args.hubert_dir)]
    dicts.remove('.DS_Store')
    n = len(dicts)

    total_keys = set()
    for dict in dicts:
        with open(dict, 'r') as f:
            dict = json.load(f)
        total_keys.update(set(dict.keys()))
    assert len(total_keys) == n*100

def test_all_dictionaries():
    args = get_args()
    os.chdir(args.hubert_dir)
    dicts = [file.name for file in os.scandir(args.hubert_dir)]
    dicts.remove('.DS_Store')
    n = len(dicts)

    total_keys = set()
    for dict in dicts:
         with open(dict, 'r') as f:
            dict = json.load(f)
            total_keys.update(set(dict.keys()))
    assert len(total_keys) == 7442


#main
#----------------------------
def main():
    args = get_args()
    os.chdir(args.audio_dir)

    paths_json = save_sorted_file_paths(args)
    with open(paths_json, 'r') as f:
        file_paths = json.load(f)

    if not os.path.exists(os.path.join(args.hubert_dir, 'path_encoding_0-99.json')):
        embed_audio_batch(args, file_paths, 0, 100)
    else:
        current_slices = [file.name for file in os.scandir(args.hubert_dir)]
        if '.DS_Store' in current_slices:
            current_slices.remove('.DS_Store') 
        start, end = determine_new_slice(current_slices)
        if end > 7442:
            end = 7442
        embed_audio_batch(args, file_paths, start, end)

if __name__ == '__main__':
    #main()
    #test_all_dictionaries()    
    print("audio encoding completed")