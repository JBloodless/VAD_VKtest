import numpy as np
import matplotlib.pyplot as plt
import webrtcvad
from config import config


def split_frames(data):
    frames = np.array(np.array_split(data, config.frame_size))
    print(frames.shape)
    return frames.T


class Visualize:

    @staticmethod
    def _norm_raw(raw):
        '''
        Private function.
        Normalize the raw signal into a [0..1] range.
        '''
        return raw / np.max(np.abs(raw), axis=0)

    @staticmethod
    def _time_axis(raw, labels):
        '''
        Private function.
        Generates time axis for a raw signal and its labels.
        '''
        time = np.linspace(0, len(raw) / config.sr, num=len(raw))
        time_labels = np.linspace(0, len(raw) / config.sr, num=len(labels))
        return time, time_labels

    @staticmethod
    def _plot_waveform(audio, labels, title='Sample'):
        '''
        Private function.
        Plot a raw signal as waveform and its corresponding labels.
        '''
        # raw = Visualize._norm_raw(frames.flatten())
        time, time_labels = Visualize._time_axis(audio, labels)

        plt.figure(1, figsize=(16, 3))
        plt.title(title)
        plt.plot(time, audio)
        plt.plot(time_labels, labels)
        plt.show()

    @staticmethod
    def plot_sample(audio, labels, title='Sample', show_distribution=True):
        '''
        Plot a sample with its original labels
        (before noise is applied to sample).
        '''
        Visualize._plot_waveform(audio, labels, title)

        # Print label distribution if enabled.
        if show_distribution:
            voice = (labels.count(1) * 100) / len(labels)
            silence = (labels.count(0) * 100) / len(labels)
            print('{0:.0f} % voice {1:.0f} % silence'.format(voice, silence))

    @staticmethod
    def plot_sample_webrtc(audio_b, sensitivity=0):
        '''
        Plot a sample labeled with WebRTC VAD
        (after noise is applied to sample).
        Sensitivity is an integer from 0 to 2,
        with 0 being the most sensitive.
        '''
        vad = webrtcvad.Vad(sensitivity)
        labels = np.array([1 if vad.is_speech(f.tobytes(), sample_rate=config.sr) else 0 for f in split_frames(audio_b)])
        Visualize._plot_waveform(audio_b, labels, title='Sample (WebRTC)')

    @staticmethod
    def plot_features(mfcc=None, delta=None):
        '''
        Plots the MFCC and delta-features
        for a given sample.
        '''
        if mfcc is not None:
            plt.figure(1, figsize=(16, 3))
            plt.plot(mfcc)
            plt.title('MFCC ({0} features)'.format(mfcc.shape[1]))
            plt.show()

        if delta is not None:
            plt.figure(1, figsize=(16, 3))
            plt.plot(delta)
            plt.title('Deltas ({0} features)'.format(mfcc.shape[1]))
            plt.show()