import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.autograd import Variable

from loader_preproc import DatasetLoader
from data_generator import DataGenerator
from config import config


def test_predict(model, noise_level):
    '''
    Computes predictions on test data using given network.
    '''

    # Set up an instance of data generator using default partitions
    dataset_test = DatasetLoader(
        [r'D:\Datasets\LibriSpeech\dev_clean'], r'D:\Datasets\noises',
        noise_prob=0.7)
    generator_test = DataGenerator(dataset_test.mix_generator, batch_count=100)

    if noise_level not in config.noise_snrs:
        print('Error: invalid noise level!')
        return

    model.eval()
    generator_test.set_noise_level_db(noise_level)

    y_true, y_score = [], []

    for i in range(generator_test.batch_count):

        X, y = generator_test.get_batch(config.batch_size)
        X = Variable(torch.from_numpy(np.array(X)).float())
        y = Variable(torch.from_numpy(np.array(y))).long()

        X.to(config.device)

        out = model(X)

        out = out.cpu()
        y = y.cpu()

        # Add true labels.
        y_true.extend(y.data.numpy())

        # Add probabilities for positive labels.
        y_score.extend(out.data.numpy()[:, 1])

    return y_true, y_score


def roc_auc(nets, noise_lvl):
    '''
    Generates a ROC curve for the given network and data for each noise level.
    '''
    plt.figure(1, figsize=(16, 10))
    plt.title('Receiver Operating Characteristic (%s)' % noise_lvl, fontsize=16)

    # For each noise level
    for key in nets:
        net = nets[key]

        # Make predictions
        y_true, y_score = test_predict(net, noise_lvl)

        # Compute ROC metrics and AUC
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc_res = metrics.auc(fpr, tpr)

        # Plots the ROC curve and show area.
        plt.plot(fpr, tpr, label='%s (AUC = %0.3f)' % (key, auc_res))

    plt.xlim([0, 0.2])
    plt.ylim([0.6, 1])
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(loc='lower right', prop={'size': 16})
    plt.show()


def vad(model, noise_level='-3', init_pos=50, length=700, only_plot_net=False):
    '''
    Generates a sample of specified length and runs it through
    the given network. By default, the network output is plotted
    alongside the original labels and WebRTC output for comparison.
    '''

    # Set up an instance of data generator using default partitions
    dataset_test = DatasetLoader(
        [r'D:\Datasets\LibriSpeech\dev_clean'], r'D:\Datasets\noises',
        noise_prob=0.7)
    generator_test = DataGenerator(dataset_test.mix_generator, batch_count=100)

    if noise_level not in config.noise_snrs:
        print('Error: invalid noise level!')
        return

    model.eval()
    generator_test.set_noise_level_db(noise_level)

    mfcc, delta, labels, mix_data = generator_test.get_data()




    # Plot results
    print('Displaying results for noise level:', noise_level)
    if not only_plot_net:
        Vis.plot_sample(raw_frames, labels, show_distribution=False)
        Vis.plot_sample_webrtc(raw_frames, sensitivity=0)
    Vis.plot_sample(raw_frames, accum_out, title='Sample (Neural Net)', show_distribution=False)