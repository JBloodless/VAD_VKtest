import numpy as np
from matplotlib import pyplot as plt

from config import config


class DataGenerator:

    def __init__(self, mix_gen, batch_count=0):
        self.batch_count = batch_count
        self.generator = mix_gen
        self.noise_level = None
        self.batch_size = config.batch_size

    def set_noise_level_db(self, level):
        if level not in config.noise_snrs:
            raise Exception(f'Noise level "{level}" not supported!')

        self.noise_level = level

    def get_data(self, n_frames):
        mfcc, delta, labels, mix = self.generator(n_frames, self.noise_level)
        print(mfcc.shape, delta.shape, labels.shape, mix.shape)
        return mfcc, delta, labels, mix

    def get_batch(self, batch_size, n_frames):
        self.batch_size = batch_size

        x, y, i = [], [], 0

        # Get batches
        while len(y) < self.batch_size:
            mfcc, delta, labels, mix = self.get_data(n_frames)
            X = np.hstack((mfcc, delta))
            x.append(X)
            y_range = labels
            y.append(int(y_range[int(len(labels) / 2)]))

        return np.array(x), np.array(y)



if __name__ == '__main__':
    from loader_preproc import DatasetLoader

    dataset = DatasetLoader([r'D:\Datasets\LibriSpeech\train-clean-360', r'D:\Datasets\LibriSpeech\train-other-500'],
                             r'D:\Datasets\noises', noise_prob=0.7)
    generator = DataGenerator(dataset.mix_generator, batch_count=10000)
    generator.set_noise_level_db(0)
    X, y = generator.get_batch(batch_size=32, n_frames=5)
    print(X.shape, y.shape)
