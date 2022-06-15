from models import ConvGRU, SelfAttentiveVAD
from loader import dataset_loader
from data_generator import DataGenerator
import numpy as np

import torch

from config import config

dataset_tr = dataset_loader(r'D:\Datasets\LibriSpeech\train-other-500', r'D:\Datasets\noises')
generator_tr = DataGenerator(dataset_tr.mix_generator, size_limit=10000)
generator_tr.set_noise_level_db(0)

# dataset_val = dataset_loader(r'D:\Datasets\LibriSpeech\train-other-500', r'D:\Datasets\noises')
# generator_val = DataGenerator(dataset_val.mix_generator, size_limit=10000)
# generator_val.set_noise_level_db(0)

def accuracy(out, y):
    '''
    Calculate accuracy of model where
    out.shape = (64, 2) and y.shape = (64)
    '''
    out = torch.max(out, 1)[1].float()
    eq = torch.eq(out, y.float()).float()
    return torch.mean(eq)

if __name__ == '__main__':
    net = ConvGRU(large=False)
    for i in range(3):

    # Get batch
        X, y = generator_tr.get_batch(config.batch_size)

        X = torch.from_numpy(np.array(X)).float().cpu()
        y = torch.from_numpy(np.array(y)).cpu()

        # Run through network
        out = net(X)
        print(out.shape, y.shape)
        acc = accuracy(out, y).data.numpy()

print('Successfully ran the network!\n\nExample output:', out.data.numpy()[0])