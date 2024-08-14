import pandas as pd
import torch
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import torchaudio
import os
import librosa
import numpy as np
from pydub import AudioSegment
import pickle

# # Read the labeling file into a DataFrame
# file_path = r'D:\code\bird_data\chronister\annotation_Files\Recording_1\Recording_1_Segment_04.Table.1.selections.txt'
# df = pd.read_csv(file_path, sep='\t')
#
# # Display the first few rows of the DataFrame to check the contents
# print(df.head())
#

#--------------------------------------------------


class BirdSongDataset(Dataset):
    def __init__(self, audio, classlabel):
        self.audio = audio
        self.classlabel = classlabel

    def __len__(self):
        return len(self.classlabel)

    def __getitem__(self, idx):
        return self.audio[idx], self.classlabel[idx]


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
    data = pd.read_csv(txt_path, sep='\t')
    for line in range(len(data)):
        row = data.iloc[line]

        # Extract the necessary information
        begin_time = row['Begin Time (s)'] * 1000
        end_time = row['End Time (s)'] * 1000
        label = row['Species']

        segments.append((float(begin_time), float(end_time), label))
    return segments

def transform_audio_to_mel(y, sr=22050, n_mels=224, n_fft=2048, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

#===========================================

def save_data():
    txt_root_path = r'D:\code\bird_data\chronister\annotation_Files'
    wav_root_path = r'D:\code\bird_data\chronister\wav_formed'
    recording_list = ['Recording_1', 'Recording_2', 'Recording_3', 'Recording_4']

    features, classes = [], []
    for record in recording_list:
        wav_files = list_wav_files(os.path.join(wav_root_path, record))
        for wav_file in wav_files:
            txt_path = os.path.join(txt_root_path, record, wav_file.replace('.wav', '.Table.1.selections.txt'))
            wav_path = os.path.join(wav_root_path, record, wav_file)
            segments = load_labels(txt_path)  # 用于获得单个MP3文件中有效时间
            audio = AudioSegment.from_mp3(wav_path)

            # get all segments from a wav file
            audio_segments, labels = [], []
            for segment in segments:
                start_time, end_time, label = segment
                audio_seg = audio[start_time:end_time]

                # Export segment to raw audio data
                y = np.array(audio_seg.get_array_of_samples(), dtype=np.float32) / 32768.0

                # Resample the segment if necessary
                y = librosa.resample(y, orig_sr=audio_seg.frame_rate, target_sr=22050)

                # Transform audio
                mel_spectrogram_db = transform_audio_to_mel(y)

                audio_segments.append(mel_spectrogram_db)
                labels.append([label] * mel_spectrogram_db.shape[1])

            audio_array = np.concatenate(audio_segments, axis=1)
            labels = np.concatenate(labels)

            features.append(audio_array)
            classes.append(labels)


    savep = r'D:\code\bird_data\chronister'
    with open(os.path.join(savep,'features.pkl'), 'wb') as f:
        pickle.dump(features, f)
    with open(os.path.join(savep,'classes.pkl'), 'wb') as f:
        pickle.dump(classes, f)
    print('')
#===========================================

def load_data():
    savep = r'D:\code\bird_data\chronister'
    with open(os.path.join(savep,'features.pkl'), 'rb') as f:
        features = pickle.load(f)

    with open(os.path.join(savep,'classes.pkl'), 'rb') as f:
        classes = pickle.load(f)

    return features, classes

# Example usage
def main():

    audios, classlabels = load_data()

    audios = np.concatenate(audios, axis=1).transpose()
    classlabels = np.concatenate(classlabels)

    label_uniq = list(set(classlabels))
    folderIDs = np.arange(len(label_uniq))
    label_dict = dict(zip(label_uniq, folderIDs))
    classlabelIDs = [label_dict[i] for i in classlabels]

    # Instantiate the dataset
    dataset = BirdSongDataset(audios, classlabelIDs)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Iterate through the DataLoader
    for audio_segments, labels in dataloader:
        print(f"Batch Audio Shape: {audio_segments.shape}, Labels: {labels}")


if __name__ == '__main__':
    main()