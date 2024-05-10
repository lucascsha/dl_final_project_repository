#----------------------------
from crema_d import CremaD_HubertEncoded
from model import AudioModel_learnable
import argparse
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from visualize_training import organize_losses, organize_accs
import json
import numpy as np
#system configuration
model = None
#----------------------------
def get_args():
    parser = argparse.ArgumentParser(description = 'Process audio files, organize data')
    #directory and file paths
    parser.add_argument('--code_dir', type=str, default='/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/self_implementation', help='Directory where code implementation is stored')
    parser.add_argument('--root_dir', type = str, default='/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/cs1470_ser_fp_data', help='Directory where (raw, processed) data are stored')

    args, _ = parser.parse_known_args()
    parser.add_argument('--crema_dir', type = str, default = os.path.join(args.root_dir, 'crema-d_data/AudioWAV'), help = 'Directory where raw CREMA-D dataset WAV audio stored')
    parser.add_argument('--audio_dir', type=str, default= os.path.join(args.root_dir,'processed_data/audio_dir'), help= 'Directory where .npy audio vectors saved')
    parser.add_argument('--loss_dir', type = str, default = '/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/model/losses')
    parser.add_argument('--acc_dir', type = str, default = '/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/model/accuracies')
    parser.add_argument('--json_path', type=str, default= os.path.join(args.root_dir,'processed_data/speaker_dict.json'), help='Path for JSON file mapping speaker ids to .npy audio feature paths')

    parser.add_argument('--hubert_dir', type = str, default = os.path.join(args.root_dir, 'processed_data/hubert_dir_repadded'))
    #model selection
    parser.add_argument('--model_type', type=str, default= 'audiomodel_learnable', help= 'Model used for training and testing')

    #hyperparameters
    parser.add_argument('--batch_size', type=int, default= 150, help= 'Batch size of data') #Use batch_size = 30 and l_r = 1e-4 FOR BASE MODEL WO PERSONALIZATION
    parser.add_argument('--optimizer', type=str, default= 'adam', help= 'Optimizer used to update gradients')
    parser.add_argument('--learning_rate', type=float, default= 1.5e-3, help= 'Model learning rate')
    parser.add_argument('--epochs', type = int, default = 20, help = 'Number of epochs to train for')

    return parser.parse_args()

#load data
#----------------------------
def load_data(args):
    os.chdir(args.hubert_dir)
    cremaD =  CremaD_HubertEncoded(args.hubert_dir)

    #shuffle
    dataset = cremaD.data.shuffle(buffer_size = 10000)

    #split
    size = tf.data.experimental.cardinality(dataset).numpy()
    train_split = int(0.6*size)
    test_split = int(0.2 * size)
    val_split = size - train_split - test_split

    
    train_data = dataset.take(train_split)
    test_data = dataset.take(test_split)
    val_data = dataset.take(val_split)

    #batch
    train_data = train_data.batch(args.batch_size)
    test_data = test_data.batch(test_split)
    val_data = val_data.batch(val_split)


    #prefetch
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    test_data = test_data.prefetch(tf.data.AUTOTUNE)
    val_data = val_data.prefetch(tf.data.AUTOTUNE)

    return train_data, test_data, val_data

#set model
#----------------------------
def get_model(args):
    if args.model_type.lower() == 'audiomodel_learnable':
        return AudioModel_learnable('standardscaler')
    #if args.model_type.lower() == 'audiomodel_fixed':
       # return AudioModel_fixed()
    
def get_optimizer(args):
    if args.optimizer.lower() == 'adam':
        return tf.keras.optimizers.legacy.Adam(learning_rate = args.learning_rate)
    
def set_checkpoint(args):
    checkpoint_path = os.path.join(args.code_dir, "training_1/cp.ckpt")
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only = False, verbose=1)
    return cp_callback

#train
#----------------------------
def train(args):
    global model 
    #load data
    train_data, _, val_data = load_data(args)
    with open(args.json_path, 'r') as f:
        speaker_dict = json.load(f)
    

    #set up model
    model = get_model(args)
    optimizer = get_optimizer(args)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits = False)

    #track metrics
    train_losses = []
    train_accs = []

    val_losses = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
    #training
        for batch in train_data:
            _, speaker_ids, _, emo_labels, _, encoded_audio_data = batch
            with tf.GradientTape() as tape:
                preds = model(speaker_dict, encoded_audio_data, speaker_ids)
                loss = loss_fn(emo_labels, preds)    

            #update metrics
            train_losses.append(loss.numpy())
            acc = model.accuracy(preds, emo_labels)
            train_accs.append(acc.numpy())
            
            #update weights
            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))
       
        print('=============')
        print(f'Training loss at epoch {epoch}/{args.epochs}: {loss}')
        print(f'Training accuracy at epoch {epoch}/{args.epochs}: {acc}')

    
    #validation 
        for data in val_data:
            _,val_speaker_ids, _, val_emo_labels, _, val_audio_data = data
        val_preds = model(speaker_dict, val_audio_data, val_speaker_ids)

        #metrics
        val_loss = loss_fn(val_emo_labels, val_preds)
        val_losses.append(val_loss)
        val_acc = model.accuracy(val_preds, val_emo_labels)
        val_accs.append(val_acc)

        print('=============')
        print(f'Validation loss at epoch {epoch}/{args.epochs}: {val_loss}')
        print(f'Validation accuracy at epoch {epoch}/{args.epochs}: {val_acc}')

    return train_losses, train_accs, val_losses, val_accs 

#test
#----------------------------
def test(args):
    #load data
    _, test_data, _ = load_data(args)
    with open(args.json_path, 'r') as f:
        speaker_dict = json.load(f)

    #model setup
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits = False)
 
    #evaluate
    for data in test_data:
        _, speaker_ids, _, emo_labels, _, audio_data = data
        preds = model(speaker_dict, audio_data, speaker_ids) 
        #metrics

        test_loss = loss_fn(emo_labels, preds)
        test_acc = model.accuracy(preds, emo_labels)
        print('=============')
        print(f'Test loss: {test_loss}')
        print(f'Test accuracy: {test_acc}')
    





def main():
    args = get_args()
    train_losses, train_accs, val_losses, val_accs = train(args)
    x = val_accs
    print(np.mean(x[10:]), np.max(x))

    organize_losses(train_losses, args.loss_dir)
    organize_accs(train_accs, args.acc_dir)

    test(args)

if __name__ == '__main__':
    main()
    #args = get_args()
    #load_data(args)
