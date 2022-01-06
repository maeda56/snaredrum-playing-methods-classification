import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import glob

class_dict = {
    'Strike':0,
    'Rim':1,
    'Cross Stick':2,
    'Buzz':3
}

class SnareDataset(Dataset):

    def __init__(self,
                 audio_path,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.audio_path = audio_path
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.audio_path)

    def __getitem__(self, index):
        audio_sample_path = self.audio_path[index]
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = signal.to(self.device)   # この下でも上でもうまくいかない。ここで。
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_label(self, index):
        target_path = self.audio_path[index]
        label = os.path.basename(os.path.dirname(target_path))
        label_index = class_dict[label]
        return label_index


def get_datapath_list():
    rootpath = "../../data/MDLib2.2/MDLib2.2/Sorted/Snare"
    target_path = os.path.join(rootpath+'/**/*.wav')

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    print(f"path_list length : {len(path_list)}")
    return path_list


    



if __name__ == "__main__":
    audio_path = get_datapath_list()
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    sd = SnareDataset(audio_path,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(f"There are {len(sd)} samples in the dataset.")
    signal, label = sd[1000]
