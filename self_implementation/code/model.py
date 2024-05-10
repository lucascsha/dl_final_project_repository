#Model implementations
#---------------------------
import tensorflow as tf
import tensorflow.keras as keras
from transformers import AutoProcessor, TFHubertModel
import json
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

#models
#---------------------------
#HuBERT processing wrapper
class HubertWrapper(keras.Model):
    def __init__(self):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.encoder = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

    def call(self, paths, path_encoding_dict):
        #encode audio
        max_audio_len = 48000
        raw_audio_batch = []
        for path in paths:
            if len(np.load(path)) <= max_audio_len:
                raw_audio_batch.append(np.load(path))
            else:
                raw_audio_batch.append(np.load(path)[0:max_audio_len])
        
        raw_audio_features = self.processor(raw_audio_batch, sampling_rate = 16000, return_tensors = 'tf', padding = True)['input_values']

        audio_encoding = self.encoder(raw_audio_features).last_hidden_state[:,0,:]
        audio_encoding = audio_encoding.numpy().tolist()

        #update dictionary
        for i in range(len(paths)):
            path_encoding_dict[paths[i]] = audio_encoding[i]

        return path_encoding_dict
    
#fine tuning head with learnable MFCC speaker feature embedding
class AudioModel_learnable(keras.Model):
    def __init__(self, scaler_type):
        super().__init__()
        self.speaker_embedding = tf.keras.layers.Embedding(92, 64)
        self.embed_dense = tf.keras.layers.Dense(1024)

        #classifier
        self.dense1 = tf.keras.layers.Dense(1024, activation = 'relu')
        self.drop1 = tf.keras.layers.Dropout(0.1)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.dense2 = tf.keras.layers.Dense(256, activation = 'relu')
        self.drop2 = tf.keras.layers.Dropout(0.1)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
        self.drop3 = tf.keras.layers.Dropout(0.1)
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        self.dense4 = tf.keras.layers.Dense(64, activation='relu')
        self.drop4 = tf.keras.layers.Dropout(0.1)
        self.batchnorm4 = tf.keras.layers.BatchNormalization()
        self.final_dense = tf.keras.layers.Dense(6, activation = 'softmax')

        if scaler_type.lower() == 'standardscaler':
            self.scaler = StandardScaler()

    def call(self, speaker_dict, encoded_audio, speaker_ids):
        #speaker_features = tf.convert_to_tensor([speaker_dict[str(speaker_id)][0] for speaker_id in speaker_ids.numpy().tolist()]) #(batch_size, 13)

        personalized_encoding = encoded_audio + self.embed_dense(self.speaker_embedding(speaker_ids- 1001))

        #extra layers
        preds = self.dense1(personalized_encoding)
        preds = self.drop1(preds); preds = self.batchnorm1(preds)
        #1
        preds = self.dense2(preds) 
        preds = self.drop2(preds); preds = self.batchnorm2(preds)

        #2
        preds = self.dense3(preds)
        preds = self.drop3(preds); preds = self.batchnorm3(preds)
        #3 
        preds = self.dense4(preds)
        preds = self.drop4(preds); preds = self.batchnorm4(preds)

        #final 
        preds = self.final_dense(preds)
        return preds
    
    def accuracy(self, preds, labels):
        correct_predictions = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    


