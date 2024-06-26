import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os 
import numpy as np
import librosa

root_dir = '/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/cs1470_ser_fp_data'
crema_dir = os.path.join(root_dir, 'crema-d_data/AudioWAV')
audio_dir = os.path.join(root_dir, 'processed_data/audio_dir')

#----------------------------
#speakers
def return_speakers():
    return [n for n in range(1001, 1092)]


#----------------------------
#labels
def return_labels():
    return ['happiness', 'anger', 'disgust', 'fear', 'neutrality', 'sadness']

def get_label_freqs(data_dir):
    freq_ANG = 0; freq_HAP = 0; freq_DIS = 0; freq_FEA = 0; freq_NEU = 0; freq_SAD = 0

    for file in os.scandir(data_dir):
        if 'ANG' in file.name:
            freq_ANG += 1
        if 'HAP' in file.name:
            freq_HAP += 1
        if 'DIS' in file.name:
            freq_DIS += 1
        if 'FEA' in file.name:
            freq_FEA += 1
        if 'NEU' in file.name:
            freq_NEU += 1
        if 'SAD' in file.name:
            freq_SAD += 1
    return [freq_HAP, freq_ANG, freq_DIS, freq_FEA, freq_NEU, freq_SAD]


#----------------------------
#plot emotion frequencies
 
def plot_emo_freq():
    emos = return_labels()
    freqs = get_label_freqs(crema_dir)

    emo_fq_frame = pd.DataFrame({'Emotion': emos, 'Frequency': freqs})
    bars  = sns.barplot(x='Emotion', y = 'Frequency', data = emo_fq_frame, palette = 'deep')
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.title('Emotion Frequencies')
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


#demographics
def plot_demo_freq():
    demographics = os.path.join(root_dir, 'crema-d_data/VideoDemographics.csv')
    caucasians = 0; african_americans = 0; asians = 0; cauc_hisp = 0; afr_hisp = 0

    with open(demographics, newline = '') as csv_file:
        demo_data = csv.reader(csv_file, delimiter = ',')
        for row in demo_data:
            if 'Caucasian' in row:
                if 'Hispanic' in row:
                    cauc_hisp += 1
                else:
                    caucasians += 1
            if 'African American' in row:
                if 'Hispanic' in row:
                    afr_hisp += 1
                else:
                    african_americans += 1
            if 'Asian' in row:
                asians += 1
    ethnicities = ['Caucasian', 'African American', 'Asian', 'Caucasian Hispanic', 'Afro-Hispanic']
    freqs = [caucasians, african_americans, asians, cauc_hisp, afr_hisp]

    demo_fq_frame = pd.DataFrame({'Ethnicity': ethnicities, 'Count': freqs})
    bars = sns.barplot(x = 'Ethnicity', y = 'Count', data = demo_fq_frame, palette = 'viridis')
    plt.xlabel('Ethnicity')
    plt.ylabel('Count')
    plt.title('Demographics')
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


#plot audio lengths
#----------------------------
def visualize_audio_lens():
    audio_files = [file.path for file in os.scandir(audio_dir)]
    audio_lengths = [len(np.load(audio_file)) for audio_file in audio_files]
    data = pd.DataFrame({'Audio Lengths': audio_lengths})
    hist = sns.histplot(data['Audio Lengths'], kde = True, palette = 'bright')

    mean_val = np.mean(data['Audio Lengths'])
    std_val = np.std(data['Audio Lengths'])

    plt.annotate(f'Mean: {mean_val:.2f}\nSD: {std_val:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                 horizontalalignment='left', verticalalignment='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', edgecolor='black', alpha=0.5))
    plt.xlabel('Audio Sample Length (Frames)')
    plt.ylabel('Count')
    plt.title('Au')
    plt.show()

def find_sr():
    paths = [file.path for file in os.scandir(crema_dir)][0:10]
    sample_rates = [librosa.load(path)[1] for path in paths]
    print(sample_rates)
    return sample_rates


plot_demo_freq()

