#----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
from crema_d import CremaD_HubertEncoded
import numpy as np
import json
from sklearn.decomposition import PCA

#----------------------------
root_dir = '/Users/lucascsha/Desktop/Classes/CS1470/cs1470_ser_fp/cs1470_ser_fp_data'
hubert_dir = os.path.join(root_dir, 'processed_data/hubert_dir')

json_path = os.path.join(root_dir, 'processed_data/speaker_dict.json')


#make colors
def get_colors(data, cmap_name='viridis'):
    norm = plt.Normalize(np.min(data), np.max(data))
    cmap = plt.get_cmap(cmap_name)
    return cmap(norm(data))

#plot data features
#----------------------------

#plot hubert output audio encodings
def plot_hubert_features():
    os.chdir(hubert_dir)
    dataset = CremaD_HubertEncoded(hubert_dir)
    audio_encoding = np.asarray(dataset.audio_encoding)
    feature1 = audio_encoding[:, 0]
    feature92 = audio_encoding[:,91]
    feature184 = audio_encoding[:,183]
    feature276 = audio_encoding[:, 275]
    feature385 = audio_encoding[:, 384]
    feature477 = audio_encoding[:, 476]
    feature569 = audio_encoding[:, 568]
    feature661 = audio_encoding[:,660]
    feature768 = audio_encoding[:, -1]

    features_to_plot = ['Feature 1', 'Feature 92', 'Feature 184', 'Feature 276', 'Feature 385', 'Feature 477','Feature 569', 'Feature 661', 'Feature 768']
    feature_data = pd.DataFrame({'Feature 1' : feature1, 'Feature 92': feature92, 'Feature 184': feature184, 'Feature 276': feature276, 
                             'Feature 385': feature385, 'Feature 477': feature477, 'Feature 569': feature569, 'Feature 661': feature661, 'Feature 768': feature768})

    colors = get_colors(feature_data)
    plt.figure(figsize=(9, 6))
    for i, feature in enumerate(features_to_plot, 1):
        plt.subplot(3, 3, i)
        sns.histplot(feature_data[feature], kde=True, palette = colors)
        #plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(np.min(feature_data), np.max(feature_data)), cmap='viridis'))
        plt.title(f'Distribution of {feature}')

        mean_val = np.mean(feature_data[feature])
        std_val = np.std(feature_data[feature])

        plt.annotate(f'Mean: {mean_val:.2f}\nSD: {std_val:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                 horizontalalignment='left', verticalalignment='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', edgecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'processed_data/hubert_features_plt.pdf'))
    plt.show()


#plot MFCC features
def plot_MFCC_speaker_features():
    os.chdir(os.path.join(root_dir, 'processed_data'))

    with open(json_path, 'r') as f:
        speaker_dict = json.load(f)

    all_features = np.asarray([speaker_dict[speaker][0] for speaker in speaker_dict.keys()])
    
    feature1 = all_features[:,0]
    feature2 = all_features[:,1]
    feature3 = all_features[:,2]
    feature4 = all_features[:,3]
    feature6 = all_features[:, 5]
    feature7 = all_features[:,6]
    feature9 = all_features[:,8]
    feature11 = all_features[:, 10]
    feature13 = all_features[:, -1]

    features_to_plot = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 6', 'Feature 7', 'Feature 9', 'Feature 11', 'Feature 13']
    feature_data = pd.DataFrame({'Feature 1' : feature1, 'Feature 2': feature2, 'Feature 3': feature3, 'Feature 4': feature4, 
                             'Feature 6': feature6, 'Feature 7': feature7, 'Feature 9': feature9, 'Feature 11': feature11, 'Feature 13': feature13})

    colors = get_colors(feature_data)
    plt.figure(figsize=(15, 12))
    for i, feature in enumerate(features_to_plot, 1):
        plt.subplot(3, 3, i)
        sns.histplot(feature_data[feature], kde=True, palette = colors)
        #plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(np.min(feature_data), np.max(feature_data)), cmap='viridis'))

        mean_val = np.mean(feature_data[feature])
        std_val = np.std(feature_data[feature])

        plt.annotate(f'Mean: {mean_val:.2f}\nSD: {std_val:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                 horizontalalignment='left', verticalalignment='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', edgecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'processed_data/mfcc_feature_dist_plot.pdf'))
    plt.show()


#get demographics
def track_demographics():
    demographics = os.path.join(root_dir, 'crema-d_data/VideoDemographics.csv')

    demographics_dict = {'Caucasian Male' : [], 'Caucasian Female': [], 'African American Male': [], 'African American Female':[],'Asian Male' :[],
                          'Asian Female' :[], 'Caucasian-Hispanic Male': [], 'Caucasian-Hispanic Female': [], 'Afro-Hispanic Male' : [], 'Afro-Hispanic Female': []}

    with open(demographics, newline = '') as csv_file:
        demo_data = csv.reader(csv_file, delimiter = ',')
        for row in demo_data:
            if 'Caucasian' in row and 'Male' in row:
                if 'Hispanic' in row:
                    demographics_dict['Caucasian-Hispanic Male'].append(row[0])
                else:
                    demographics_dict['Caucasian Male'].append(row[0])

            if 'Caucasian' in row and 'Female' in row:
                if 'Hispanic' in row:
                    demographics_dict['Caucasian-Hispanic Female'].append(row[0])
                else:
                    demographics_dict['Caucasian Female'].append(row[0])

            if 'African American' in row and 'Male' in row:
                if 'Hispanic' in row:
                    demographics_dict['Afro-Hispanic Male'].append(row[0])
                else:
                    demographics_dict['African American Male'].append(row[0])

            if 'African American' in row and 'Female' in row:
                if 'Hispanic' in row:
                    demographics_dict['Afro-Hispanic Female'].append(row[0])
                else:
                    demographics_dict['African American Female'].append(row[0])

            if 'Asian' in row and 'Male' in row:
                demographics_dict['Asian Male'].append(row[0])
            if 'Asian' in row and 'Female' in row:
                demographics_dict['Asian Female'].append(row[0])

    return demographics_dict
    

#plot mfcc clusters
def plot_MFCC_speaker_feature_clusters():
    #load speaker features
    with open(json_path, 'r') as f:
        speaker_dict = json.load(f)

    #track demographic mappings
    demographics_dict = track_demographics()
    features = []
    labels = []
    for demo_group, actor_ids in demographics_dict.items():
        for actor_id in actor_ids:
            features.append(speaker_dict[actor_id][0])
            labels.append(demo_group)

    features = np.asarray(features)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    #set up dataframe
    plot_data = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'Demographic Group': labels
    })

    #plot
    plt.figure(figsize=(12, 8))  
    scatter_plot = sns.scatterplot(data=plot_data, x='PCA1', y='PCA2', hue='Demographic Group',
                                   style='Demographic Group', palette='bright', s=100)  
    plt.title('PCA of MFCC Features by Demographic Group')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.legend(title='Demographic Group', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the rect to fit the legend outside the plot
    plt.show()
    
plot_MFCC_speaker_feature_clusters()





    




    
            




    
    




    















