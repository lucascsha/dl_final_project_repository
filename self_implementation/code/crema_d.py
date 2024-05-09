#---------------------------
import tensorflow as tf
import pandas as pd
import numpy as np
import os 
from preprocess import ohe_helper, intensity_to_num
import json

#modify path for colab 
#--------------------------
def modify_path(path, notebook_dir = '/content/drive/MyDrive/CS1470/cs1470_ser_fp/data/processed_data/audio_dir'):
  path_elts = path.split('/')
  file_name = path_elts[-1]
  notebook_path = os.path.join(notebook_dir, file_name)
  return notebook_path

#construct dataset wrapper
#---------------------------
class CremaD_HubertEncoded(object):
    def __init__(self, root_dir):
        df_dict = {'Original Path' : [], 'SpeakerID': [], 'Sentence Label': [],  'Emotional Label':[], 'Emotional Intensity' : [], 'HuBERT Embedding' : []}
        path_encoding_dicts = [file.name for file in os.scandir(root_dir)]
        if '.DS_Store' in path_encoding_dicts:
            path_encoding_dicts.remove('.DS_Store')


        for path_encoding_dict in path_encoding_dicts:
            with open(path_encoding_dict, 'r') as f:
                path_encoding_dict = json.load(f)
                paths = set(path_encoding_dict.keys())
                for path in paths:
                    path_split = path.split('_')
                    df_dict['Original Path'].append(path)
                    df_dict['SpeakerID'].append(int(path_split[0]))
                    df_dict['Sentence Label'].append(path_split[1])
                    df_dict['Emotional Label'].append(ohe_helper(path))
                    df_dict['Emotional Intensity'].append(intensity_to_num(path))
                    df_dict['HuBERT Embedding'].append(path_encoding_dict[path])
        
        self.path = df_dict['Original Path']
        self.speaker = df_dict['SpeakerID']
        self.sentence = df_dict['Sentence Label']
        self.emo_label = df_dict['Emotional Label']
        self.emo_intensity = df_dict['Emotional Intensity']
        self.audio_encoding = df_dict['HuBERT Embedding']
        
        self.data = tf.data.Dataset.from_tensor_slices((self.path, self.speaker, self.sentence, self.emo_label, self.emo_intensity, self.audio_encoding))

#test dataset iteration
#--------------------------- 
def test_dataset():
    root_dir = '/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/cs1470_ser_fp_data/processed_data/hubert_dir'
    os.chdir(root_dir)
    dataset = CremaD_HubertEncoded(root_dir).data

    for path, speaker, sentence, emo_label, emo_intensities, audio_data in dataset.take(1):
        tf.print(path, "File Path")
        tf.print(speaker, "Speaker ID")
        tf.print(sentence, "Sentence Label")
        tf.print(emo_label, "Emotional OHE Label")
        tf.print(emo_intensities, "Emotional Intensity")
        tf.print(audio_data.shape, 'HuBERT Embedding Shape')

    length = tf.data.experimental.cardinality(dataset).numpy()
    print(length) #should be 7442


