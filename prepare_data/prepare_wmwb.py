# install choco: https://www.liquidweb.com/blog/how-to-install-chocolatey-on-windows/
# add ffmpeg to root path:https://stackoverflow.com/questions/55669182/how-to-fix-filenotfounderror-winerror-2-the-system-cannot-find-the-file-speci

import os
import torch
import librosa
import numpy as np
from pydub import AudioSegment
from torch.utils.data import Dataset, DataLoader

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


root_path = r'D:\code\bird_data\wmwb\audio_files'
subfolders = list_subfolders(root_path)
def read_data():
    features, subjects, classes = [], [], []
    for wav_folder in subfolders:
        wav_files = list_wav_files(os.path.join(root_path, wav_folder))
        for subjectid, wav_file in enumerate(wav_files):
            wav_path = os.path.join(root_path, wav_folder, wav_file)
            txt_path = wav_path.replace('.wav', '.txt')
            segments = load_labels(txt_path)  # 用于获得单个MP3文件中有效时间

            # Load and convert audio using pydub
            audio = AudioSegment.from_mp3(wav_path)
            audio_segments = []
            for segment in segments:
                start_time, end_time, label = segment
                # Extract segment in milliseconds
                start_ms = start_time * 1000
                end_ms = end_time * 1000
                audio_seg = audio[start_ms:end_ms]

                # Export segment to raw audio data
                y = np.array(audio_seg.get_array_of_samples(), dtype=np.float32) / 32768.0

                # Resample the segment if necessary
                y = librosa.resample(y, orig_sr=audio_seg.frame_rate, target_sr=22050)

                # Transform audio
                mel_spectrogram_db = transform_audio_to_mel(y)

                audio_segments.append(mel_spectrogram_db)

            audio_array = np.concatenate(audio_segments, axis=1)
            # audio_array.shape
            # 接下来添加 fileID和classID
            mel_len = audio_array.shape[1]  # 和时间长度对应
            wavid = np.ones(mel_len) * subjectid
            classstr = [wav_folder] * mel_len

            features.append(audio_array)
            subjects.append(wavid)
            classes.append(classstr)

            # print('')

    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)
    with open('subjects.pkl', 'wb') as f:
        pickle.dump(subjects, f)
    with open('classes.pkl', 'wb') as f:
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


def sliding_window_split(sample, subject, classlabels, window_size=45, step=35):
    len_data, dim = sample.shape
    new_len = (len_data - window_size) // step + 1
    batch_samples = []
    batch_subjects = []
    batch_classlabels = []

    for start in range(0, len_data - window_size + 1, step):
        end = start + window_size

        # Extracting the window from sample
        batch_samples.append(sample[start:end])

        # Extracting the window from subject
        batch_subjects.append(subject[start:end])

        # Extracting the window from classlabels
        batch_classlabels.append(classlabels[start:end])

    # Converting lists to numpy arrays
    batch_samples = np.array(batch_samples)
    batch_subjects = np.array(batch_subjects)
    batch_classlabels = np.array(batch_classlabels)

    return batch_samples, batch_subjects, batch_classlabels
#
def main():
    # read_data()

    root_path = r'D:\code\bird_data\wmwb\audio_files'
    subfolders = list_subfolders(root_path)

    folderIDs = np.arange(len(subfolders))
    label_dict = dict(zip(subfolders, folderIDs))

    audios, subjects, classlabels = load_data()

    audios = np.concatenate(audios, axis=1).transpose()
    subjects = np.concatenate(subjects)
    classlabels = np.concatenate(classlabels)

    classlabelIDs = [label_dict[i] for i in classlabels]

    batch_samples, batch_subjects, batch_classlabels = sliding_window_split(audios, subjects, classlabelIDs)

    # Instantiate the dataset
    # dataset = BirdAudioSegmentDataset(audios, subjects, classlabelIDs)
    dataset = BirdAudioSegmentDataset(batch_samples, batch_subjects, batch_classlabels)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=254, shuffle=True, num_workers=0)

    # Example usage of the dataloader
    for batch in dataloader:
        inputs, labels = batch  # inputs=batch, feature dim; labels=batch
        print(inputs.shape, labels.shape)
        # Your training code here

if __name__ == '__main__':
    main()