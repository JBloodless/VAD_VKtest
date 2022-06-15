import os
import random

import numpy as np
import soundfile as sf
import webrtcvad
from pydub import AudioSegment

from config import config
import torch
from scipy.signal import convolve


from python_speech_features import mfcc, delta


class dataset_loader:
    def __init__(self, speech_path: list, noise_path, noise_prob, rir_path=None,  rir_prob=None):
        self.speech_path = speech_path
        self.noise_path = noise_path
        self.rir_path=rir_path
        self.ns_prob = noise_prob
        self.rir_prob = rir_prob
        self.all_speech, self.all_noise, self.all_rir = [], [], []
        self.isns = random.choices([0, 1], [1-self.ns_prob, self.ns_prob])[0]
        self.isrir = None

        print('building data')
        for subset in speech_path:
            for path, subdirs, files in os.walk(subset):
                for name in files:
                    # print(name)
                    if name.endswith('.wav') or name.endswith('.flac'):
                        self.all_speech.append(os.path.join(path, name))
        for path, subdirs, files in os.walk(self.noise_path):
            for name in files:
                # print(name)
                if name.endswith('.wav') or name.endswith('.flac'):
                    self.all_noise.append(os.path.join(path, name))
        if rir_path:
            self.isrir = random.choices([0,1], [1-self.rir_prob, self.rir_prob])[0]
            for path, subdirs, files in os.walk(self.rir_path):
                for name in files:
                    # print(name)
                    if name.endswith('.wav') or name.endswith('.flac'):
                        self.all_rir.append(os.path.join(path, name))
        print('data is built')

    def to_db(self, ratio):
        assert ratio >= 0

        ratio_db = 10. * np.log10(ratio + 1e-8)
        return ratio_db

    def from_db(self, ratio_db):
        ratio = 10 ** (ratio_db / 10.) - 1e-8

        return ratio

    def coef_by_snr(self, src_audio, ns_audio, snr):
        src_audio = torch.from_numpy(src_audio)
        ns_audio = torch.from_numpy(ns_audio)
        # try:
        target_snr_n = self.from_db(snr)
        ns_target_sq = torch.mean(src_audio ** 2) / target_snr_n
        ns_mult = torch.sqrt(ns_target_sq / torch.mean(ns_audio ** 2))
        # except Exception:
        #     print('Failed!')
        #     ns_mult = 1.
        abs_max = ns_mult * torch.abs(ns_audio).max().item()
        if abs_max > 1.:
            ns_mult /= abs_max
        # ns_mult = ns_mult
        return ns_mult

    def split_frames(self, data):
        frames = np.array(np.array_split(data, config.frame_size))
        print(frames.shape)
        return frames.T

    def get_random(self):
        chosen_sp = random.choice(self.all_speech)
        chosen_ns = random.choice(self.all_noise)
        sp_data, _ = sf.read(chosen_sp)
        ns_data, _ = sf.read(chosen_ns)          # TODO: force resample
        rir_data = _

        if self.rir_path:
            chosen_rir = random.choice(self.all_rir)
            rir_data, _ = sf.read(chosen_rir)

        sp_data_bytes = (AudioSegment.from_file(chosen_sp)
                         .set_frame_rate(config.sr)
                         .set_sample_width(2)
                         .set_channels(1))
        sp_data_bytes = np.array(sp_data_bytes.get_array_of_samples(), dtype=np.int16)
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

        sp_chunk_bytes = sp_data_bytes[
                         random_start_sp * config.frame_size:random_start_sp * config.frame_size + random_len_fr * config.frame_size]

        vad = webrtcvad.Vad(0)

        label_chunk = [1 if vad.is_speech(f.tobytes(), sample_rate=config.sr) else 0 for f in
                       self.split_frames(sp_chunk_bytes)]
        print(label_chunk)

        return sp_chunk, label_chunk, ns_chunk, rir_data, sp_chunk_bytes

    def mix_generator(self, snr=0):
        sp, lb, ns, rir, spb = self.get_random()
        print(self.isrir, self.isns)
        if self.isrir and self.rir_path:
            mult = np.abs(rir).max() + 1e-3
            mult = 1 / mult
            rir *= mult
            sp = convolve(sp, rir)
        if self.isns:
            mix_data = np.add(sp, self.coef_by_snr(sp, ns, snr))
        else:
            mix_data = sp
        mix_mfcc = mfcc(mix_data, samplerate=config.sr, winlen=2*config.frame_len / 1000, winstep=config.frame_len / 1000,
                        nfft=2048)
        mix_mfcc = mix_mfcc[:, 1:]
        mix_delta = delta(mix_mfcc, 1)
        print(mix_mfcc.shape, mix_delta.shape, len(lb[:mix_mfcc.shape[0]]))
        return mix_mfcc, mix_delta, lb[:mix_mfcc.shape[0]], mix_data, spb


if __name__ == '__main__':
    dataset = dataset_loader([r'D:\Datasets\LibriSpeech\train-clean-360', r'D:\Datasets\LibriSpeech\train-other-500'], r'D:\Datasets\noises', noise_prob=0.7)
    mfcc, delta, label, mix, spb = dataset.mix_generator(0)