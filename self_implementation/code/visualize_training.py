#Some functions for visualizing the training process of the model.
#----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os

#organize metrics
#----------------------------
def organize_losses(losses, dir):
    num_batches = len(losses) 
    batches = [i for i in range(1, num_batches+1)]


    loss_path = os.path.join(dir, 'training_loss.csv')

    if not os.path.exists(loss_path):
        with open(loss_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Batch', 'Loss'])
            for i in range(num_batches):
                writer.writerow([batches[i], losses[i]])
                

def organize_accs(accs, dir, type = 'training'):
    num_batches = len(accs) 
    batches = [i for i in range(1, num_batches+1)]

    loss_path = os.path.join(dir, 'training_acc.csv')

    if not os.path.exists(loss_path):
        with open(loss_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Batch', 'Accuracy'])
            for i in range(num_batches):
                writer.writerow([batches[i], accs[i]])

def visualize_losses():
    losses_path = '/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/model/losses/training_loss.csv'
    batch_loss_frame = pd.read_csv(losses_path)
    sns.lineplot(x = 'Batch', y = 'Loss', data = batch_loss_frame, palette = 'viridis')

    plt.title('Loss over Batches')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

def visualize_accs():
    acc_path = '/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/model/accuracies/training_acc.csv'
    batch_loss_frame = pd.read_csv(acc_path)
    sns.lineplot(x = 'Batch', y = 'Accuracy', data = batch_loss_frame, palette = 'viridis')

    plt.title('Accuracy over Batches')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.show()


