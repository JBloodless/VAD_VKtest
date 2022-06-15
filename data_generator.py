import numpy as np


class DataGenerator:

    def __init__(self, mix_gen, size_limit=0):
        self.size = size_limit
        self.data_mode = 0  # Default to training data
        self.generator = mix_gen
        self.noise_level = None
        self.batch_size = None

    def set_noise_level_db(self, level):

        if level not in [-5, 0, 5, 10]:
            raise Exception(f'Noise level "{level}" not supported!')

        self.noise_level = level

    def get_data(self):
        mfcc, delta, labels, mix, spb = self.generator(self.noise_level)
        return mfcc, delta, labels, mix, spb

    def get_batch(self, batch_size):
        self.batch_size = batch_size
        mfcc, delta, labels, mix, spb = self.get_data()

        x, y, i = [], [], 0

        # Get batches
        while len(y) < self.batch_size:
            X = np.hstack((mfcc, delta))
            x.append(X)
            y_range = labels
            y.append(int(y_range[int(len(labels) / 2)]))

        return np.array(x), np.array(y)

    def plot_data(self):
        from visualizer import Visualize as V
        import soundfile as sf

        mfcc, delta, label, mix, spb = self.get_data()

        V.plot_sample(mix, label)
        V.plot_sample_webrtc(spb)
        V.plot_features(mfcc, delta)
        sf.write('sample.wav', mix, 16000)



if __name__ == '__main__':
    from loader import dataset_loader
    dataset = dataset_loader(r'D:\Datasets\LibriSpeech', r'D:\Datasets\noises')
    generator = DataGenerator(dataset.mix_generator, size_limit=10000)
    generator.set_noise_level_db(0)
    X, y = generator.get_batch(32)
    print(X.shape, y.shape)