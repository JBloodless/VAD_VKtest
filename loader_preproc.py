import os
import random

import numpy as np
import soundfile as sf
import json

from config import config
import torch
from scipy.signal import convolve
import matplotlib.pyplot as plt

from python_speech_features import mfcc, delta


def split_frames(data):
    frames = np.array(np.array_split(data, config.frame_size))
    # print(frames.shape)
    return frames.T


def to_db(ratio):
    assert ratio >= 0

    ratio_db = 10. * np.log10(ratio + 1e-8)
    return ratio_db


def from_db(ratio_db):
    ratio = 10 ** (ratio_db / 10.) - 1e-8

    return ratio


def coef_by_snr(src_audio, ns_audio, snr):
    src_audio = torch.from_numpy(src_audio)
    ns_audio = torch.from_numpy(ns_audio)
    # try:
    target_snr_n = from_db(snr)
    ns_target_sq = torch.mean(src_audio ** 2) / target_snr_n
    ns_mult = torch.sqrt(ns_target_sq / torch.mean(ns_audio ** 2))
    # except Exception:
    #     print('Failed!')
    #     ns_mult = 1.
    abs_max = ns_mult * torch.abs(ns_audio).max().item()
    if abs_max > 1.:
        ns_mult /= abs_max
    return ns_mult


class DatasetLoader:
    def __init__(self, speech_path: list, noise_path, noise_prob, rir_path=None, rir_prob=None):
        self.speech_path = speech_path
        self.noise_path = noise_path
        self.rir_path = rir_path
        self.ns_prob = noise_prob
        self.rir_prob = rir_prob
        self.all_speech, self.all_noise, self.all_rir, self.all_json = [], [], [], []
        self.isns = random.choices([0, 1], [1 - self.ns_prob, self.ns_prob])[0]
        self.isrir = None

        print('building data')
        for subset in speech_path:
            for path, subdirs, files in os.walk(subset):
                for name in files:
                    # print(name)
                    if name.endswith('.wav') or name.endswith('.flac') and os.path.exists(
                            os.path.join(subset + '_labels', name.split('.')[0] + '.json')):
                        self.all_speech.append(
                            [os.path.join(path, name), os.path.join(subset + '_labels', name.split('.')[0] + '.json')])
        for path, subdirs, files in os.walk(self.noise_path):
            for name in files:
                # print(name)
                if name.endswith('.wav') or name.endswith('.flac'):
                    self.all_noise.append(os.path.join(path, name))
        if rir_path:
            self.isrir = random.choices([0, 1], [1 - self.rir_prob, self.rir_prob])[0]
            for path, subdirs, files in os.walk(self.rir_path):
                for name in files:
                    # print(name)
                    if name.endswith('.wav') or name.endswith('.flac'):
                        self.all_rir.append(os.path.join(path, name))
        print('data is built')

    def get_random(self):
        chosen_sp = random.choice(self.all_speech)
        chosen_ns = random.choice(self.all_noise)
        sp_data, _ = sf.read(chosen_sp[0])
        ns_data, _ = sf.read(chosen_ns)  # TODO: force resample
        rir_data = _
        # print(chosen_sp)

        with open(chosen_sp[1]) as json_file:
            json_label = json.load(json_file)
        label = np.zeros(len(sp_data))
        for speech_segment in json_label['speech_segments']:
            label[speech_segment['start_time']:speech_segment['end_time']] = [1] * (
                    speech_segment['end_time'] - speech_segment['start_time'])

        if self.rir_path:
            chosen_rir = random.choice(self.all_rir)
            rir_data, _ = sf.read(chosen_rir)

        # print(sp_data_bytes.shape)

        random_len_fr = min(random.randint(config.slice_min, config.slice_max),
                            len(sp_data) // config.frame_size,
                            len(ns_data) // config.frame_size)

        random_start_sp = random.randint(0, len(sp_data) // config.frame_size - random_len_fr)
        random_start_ns = random.randint(0, len(ns_data) // config.frame_size - random_len_fr)

        sp_chunk = sp_data[
                   random_start_sp * config.frame_size:random_start_sp * config.frame_size + random_len_fr * config.frame_size]
        ns_chunk = ns_data[
                   random_start_ns * config.frame_size:random_start_ns * config.frame_size + random_len_fr * config.frame_size]

        label_chunk = label[
                      random_start_sp * config.frame_size:random_start_sp * config.frame_size + random_len_fr * config.frame_size]
        assert len(sp_chunk) == len(label_chunk)

        return sp_chunk, label_chunk, ns_chunk, rir_data,

    def mix_generator(self, snr=0):
        sp, lb, ns, rir = self.get_random()
        # print(self.isrir, self.isns)
        if self.isrir and self.rir_path:
            mult = np.abs(rir).max() + 1e-3
            mult = 1 / mult
            rir *= mult
            sp = convolve(sp, rir)
        if self.isns:
            mix_data = np.add(sp, coef_by_snr(sp, ns, snr))
        else:
            mix_data = sp
        mix_mfcc = mfcc(mix_data, samplerate=config.sr, winlen=2 * config.frame_len / 1000,
                        winstep=config.frame_len / 1000,
                        nfft=2048)
        mix_mfcc = mix_mfcc[:, 1:]
        mix_delta = delta(mix_mfcc, 1)
        return mix_mfcc, mix_delta, lb, mix_data


if __name__ == '__main__':
    dataset = DatasetLoader([r'D:\Datasets\LibriSpeech\train-clean-360', r'D:\Datasets\LibriSpeech\train-other-500'],
                             r'D:\Datasets\noises', noise_prob=0.7)
    mfcc, delta, label, mix = dataset.mix_generator(0)
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(mix)
    ax2.plot(label)
    plt.show()
