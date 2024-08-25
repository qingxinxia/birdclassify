# install choco: https://www.liquidweb.com/blog/how-to-install-chocolatey-on-windows/
# add ffmpeg to root path:https://stackoverflow.com/questions/55669182/how-to-fix-filenotfounderror-winerror-2-the-system-cannot-find-the-file-speci

import os
import torch
import librosa
import numpy as np
from pydub import AudioSegment
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pydub.utils import which

# AudioSegment.converter = which("ffmpeg")


#---------------------------------------------
import pickle
def list_subfolders(directory):
    try:
        # Get a list of all sub-folder names in the directory
        subfolders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        return subfolders
    except FileNotFoundError:
        return f"The directory '{directory}' does not exist."
    except Exception as e:
        return f"An error occurred: {e}"



def list_wav_files(directory):
    try:
        # Get a list of all mp3 files in the directory
        mp3_files = [file for file in os.listdir(directory) if file.endswith('.wav')]
        return mp3_files
    except FileNotFoundError:
        return f"The directory '{directory}' does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

def load_labels(txt_path):
    segments = []
    # Load annotations
    with open(txt_path, 'r') as f:
        for line in f:
            start, end, label = line.strip().split('\t')
            segments.append((float(start), float(end), label))
    return segments


def transform_audio_to_mel(y, sr=22050, n_mels=224, n_fft=2048, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def columns_with_values(row):
    return row.dropna().index.tolist()

root_path = r'D:\code\bird_data\multilabel-bird-species-classification-nips2013\NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV'
root_train_path = os.path.join(root_path, 'train')


label_file = r'D:\code\bird_data\multilabel-bird-species-classification-nips2013\NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS\nips4b_birdchallenge_train_labels.csv'
def read_data():

    with open(label_file, 'rb') as f:
        label_df = pd.read_csv(f)
    df = label_df.drop('Filename', axis=1)
    result = df.apply(columns_with_values, axis=1)

    # 重组result，把文件名和标签对应
    filenames = label_df.Filename.values
    file_label = {}
    for i in range(len(filenames)):
        file_label[filenames[i]] = result[i]

    features, subjects, classes = [], [], []

    wav_files = list_wav_files(root_train_path)
    for subjectid, wav_file in enumerate(wav_files):
        wav_path = os.path.join(root_train_path, wav_file)
        # txt_path = wav_path.replace('.wav', '.txt')
        # segments = load_labels(txt_path)  # 用于获得单个MP3文件中有效时间
        if file_label[wav_file] == []:
            continue
        # Load and convert audio using pydub
        audio = AudioSegment.from_mp3(wav_path)

        # Export segment to raw audio data
        y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

        # Resample the segment if necessary
        y = librosa.resample(y, orig_sr=audio.frame_rate, target_sr=22050)

        # Transform audio
        mel_spectrogram_db = transform_audio_to_mel(y)

        features.append(mel_spectrogram_db)
        # subjects.append(wavid)
        classes.append(file_label[wav_file])

        # print('')


    with open('features_nps4b.pkl', 'wb') as f:
        pickle.dump(features, f)
    # with open('subjects_nps4b.pkl', 'wb') as f:
    #     pickle.dump(subjects, f)
    with open('classes_nps4b.pkl', 'wb') as f:
        pickle.dump(classes, f)


def load_data():
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('subjects.pkl', 'rb') as f:
        subjects = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)

    return features, subjects, classes
#---------------------------------------------


class BirdAudioSegmentDataset(Dataset):
    def __init__(self, audio, subject, classlabel):
        self.audio = audio
        self.subject = subject
        self.classlabel = classlabel

    def __len__(self):
        return len(self.subject)

    def __getitem__(self, idx):
        return self.audio[idx], self.classlabel[idx]

    # def _label_to_int(self, label):
    #     # Example label conversion
    #     label_dict = {'song': 0}  # Add more labels as needed
    #     return label_dict.get(label, -1)

#
#
def main():

    read_data()

    root_path = r'D:\code\bird_data\wmwb\audio_files'
    subfolders = list_subfolders(root_path)

    folderIDs = np.arange(len(subfolders))
    label_dict = dict(zip(subfolders, folderIDs))

    audios, subjects, classlabels = load_data()

    audios = np.concatenate(audios, axis=1).transpose()
    subjects = np.concatenate(subjects)
    classlabels = np.concatenate(classlabels)

    classlabelIDs = [label_dict[i] for i in classlabels]

    # Instantiate the dataset
    dataset = BirdAudioSegmentDataset(audios, subjects, classlabelIDs)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=254, shuffle=True, num_workers=0)

    # Example usage of the dataloader
    for batch in dataloader:
        inputs, labels = batch  # inputs=batch, feature dim; labels=batch
        print(inputs.shape, labels.shape)
        # Your training code here

if __name__ == '__main__':
    main()