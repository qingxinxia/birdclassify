import os
import pandas as pd
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt 


UPPER_QUARTILE = 80000  # 需要自己定义数据长度

def read_data_label(folders):
    '''
    folders = [Recording_1, Recording_2, 3, 4]
    '''
    data_root_path = os.path.join(os.getcwd(), 'datasets', 'zenodo_chronister')
    annotation_root_path = os.path.join(data_root_path, 'annotation_Files')
    wav_root_path = os.path.join(data_root_path, 'wav_Files')
    max_wav_lens = []
    for folder in folders:
        # read path
        annotation_path = os.path.join(annotation_root_path, folder)
        txt_files = [os.path.join(annotation_path, f) for f in os.listdir(annotation_path) if f.endswith('.txt')]
        sorted_txt_files = sorted(txt_files)

        wav_path = os.path.join(wav_root_path, folder)
        wav_files = [os.path.join(wav_path, f) for f in os.listdir(wav_path) if f.endswith('.WAV')]
        sorted_wav_files = sorted(wav_files)

        # read data
        samples = []
        for txt_f, wav_f in zip(sorted_txt_files, sorted_wav_files):
            annotations = pd.read_csv(txt_f, delimiter='\t')  # 读取每个txt文件中的所有标签
            # 加载音频文件并提取指定的时间段
            waveform, sample_rate = torchaudio.load(wav_f)
            # plt.plot(waveform.t().numpy()); plt.show()
            for index, annotation in annotations.iterrows():  # 获取txt文件中的每一行标签
                # file_name = annotation['Selection']
                start_time = annotation['Begin Time (s)']
                end_time = annotation['End Time (s)']
                label = annotation['Species']

                # 转换开始时间和结束时间为样本索引
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                waveform_seg = waveform[:, start_sample:end_sample]  

                # 存储max waveform length，之后pad所有数据到相同长度
                max_wav_lens.append(waveform_seg.size()[1])

                # 存储一个sample和label
                sample = {'waveform': waveform_seg, 'sample_rate': sample_rate, 'label': label}
                samples.append(sample)
    # # 绘制柱状图
    # plt.figure(figsize=(10, 6))
    # plt.hist(max_wav_lens, bins='auto', alpha=0.7, rwidth=0.85)

    # # 添加标题和标签
    # plt.title('Data Distribution')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')

    # # 显示图表
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()
    # 统一sample长度  
    upper_quartile = int(np.percentile(max_wav_lens, 75))  # 计算上四分位数

    for sidx in range(len(samples)):
        sample = samples[sidx]
        padding_size = UPPER_QUARTILE - waveform_seg.size()[1]
        if padding_size > 0:
            waveform_seg_pad = F.pad(waveform_seg, (0, padding_size), "constant", 0)
        else:
            waveform_seg_pad = waveform_seg[:, : UPPER_QUARTILE]
        nsample = {'waveform': waveform_seg_pad, 'sample_rate': sample_rate, 'label': label}
        samples[sidx] = nsample

    return samples

class AudioSegmentDataset(Dataset):
    def __init__(self, folders, transform=None):
        """
        Args:
            annotation_file (string): Path to the annotation file.
            audio_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
            在这里将不同folder内的txt和wav文件一起读取
        """
        self.transform = transform
        self.samples = read_data_label(folders)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.samples[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToMelSpectrogram(object):
    def __init__(self, n_mels=64):
        self.n_mels = n_mels
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)

    def __call__(self, sample):
        waveform, sample_rate, label = sample['waveform'], sample['sample_rate'], sample['label']
        mel_spec = self.mel_spectrogram(waveform)
        return {'waveform': waveform, 'mel_spectrogram': mel_spec, 'label': label}

class Normalize(object):
    def __call__(self, sample):
        waveform, mel_spectrogram, label = sample['waveform'], sample['mel_spectrogram'], sample['label']
        mel_spec_normalized = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()
        return {'waveform': waveform, 'mel_spectrogram': mel_spec_normalized, 'label': label}



if __name__ == '__main__':
    # 定义数据转换
    transform = transforms.Compose([
        ToMelSpectrogram(n_mels=64),
        Normalize()
    ])

    # 创建数据集
    folders = ['Recording_1']  # 通过这里拆分训练集和测试集，只能按照folder拆分
    # audio_dataset = AudioSegmentDataset(folders, transform=None)
    audio_dataset = AudioSegmentDataset(folders, transform=transform)


    # 创建数据加载器
    dataloader = DataLoader(audio_dataset, batch_size=16, shuffle=True, num_workers=0)

    # 示例: 遍历数据加载器中的数据
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['mel_spectrogram'].size(), len(sample_batched['label']))
        # print(i_batch, sample_batched['waveform'].size(), len(sample_batched['label']))  # 没有使用transform时

        # plt.figure(1)
        # plt.plot(sample_batched['waveform'][0,:].numpy())
        # plt.show()
        # plt.figure(2)
        # plt.imshow(sample_batched['mel_spectrogram'].log2()[0,0,:,:].numpy(), cmap='gray')
        # plt.show()
