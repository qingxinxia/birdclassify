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

class BirdAudioSegmentDataset(Dataset):
    def __init__(self, audio_path, annotation_path, transform=None, sample_rate=22050):
        self.audio_path = audio_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.segments = []

        # Load annotations
        with open(annotation_path, 'r') as f:
            for line in f:
                start, end, label = line.strip().split('\t')
                self.segments.append((float(start), float(end), label))

        # Load and convert audio using pydub
        self.audio = AudioSegment.from_mp3(audio_path)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        start_time, end_time, label = self.segments[idx]

        # Extract segment in milliseconds
        start_ms = start_time * 1000
        end_ms = end_time * 1000
        segment = self.audio[start_ms:end_ms]

        # Export segment to raw audio data
        y = np.array(segment.get_array_of_samples(), dtype=np.float32) / 32768.0

        # Resample the segment if necessary
        y = librosa.resample(y, orig_sr=segment.frame_rate, target_sr=self.sample_rate)

        # Transform audio
        if self.transform:
            y = self.transform(y)

        # # 比较每秒钟梅尔光谱的特征
        # melp = r'D:\code\bird_data\wmwb\spectrograms\Acrocephalus_arundinaceus\XC417157_0.npy'
        # mel_array = np.load(melp)
        # # Convert label to a numerical format if needed
        # label = self._label_to_int(label)

        return y, label

    def _label_to_int(self, label):
        # Example label conversion
        label_dict = {'song': 0}  # Add more labels as needed
        return label_dict.get(label, -1)


# Example transform function for generating Mel spectrogram
def transform_audio_to_mel(y, sr=22050, n_mels=224, n_fft=2048, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

# Convert MP3 to WAV using pydub
def convert_mp3_to_wav(mp3_path, wav_path):
    print(AudioSegment.ffmpeg)
    AudioSegment.converter = which("ffmpeg")
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format='wav')

def main():
    # Define paths to the audio file and annotation file
    root_p = r'D:\code\bird_data\wmwb\audio_files\Acrocephalus_arundinaceus'
    audio_path = os.path.join(root_p, 'XC417157.wav')
    annotation_path = os.path.join(root_p, 'XC417157.txt')

    # Instantiate the dataset
    dataset = BirdAudioSegmentDataset(mp3_path, annotation_path, transform=transform_audio_to_mel)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Example usage of the dataloader
    for batch in dataloader:
        inputs, labels = batch
        print(inputs.shape, labels)
        # Your training code here

if __name__ == '__main__':
    main()