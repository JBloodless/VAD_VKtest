import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable

from config import config
from data_generator import DataGenerator
from loader_preproc import DatasetLoader
from train import set_seed


def test_predict(model, noise_level):
    '''
    Computes predictions on test data using given network.
    '''

    # Set up an instance of data generator using default partitions
    dataset_test = DatasetLoader(
        [r'D:\Datasets\LibriSpeech\dev_clean'], r'D:\Datasets\noises',
        noise_prob=0.7)
    generator_test = DataGenerator(dataset_test.mix_generator, batch_count=config.test_size)

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


def roc_auc(model, noise_lvl):
    '''
    Generates a ROC curve for the given network and data for each noise level.
    '''
    plt.figure(1, figsize=(16, 10))
    plt.title('Receiver Operating Characteristic (%s)' % noise_lvl, fontsize=16)

    # Make predictions
    y_true, y_score = test_predict(model, noise_lvl)

    # Compute ROC metrics and AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc_res = metrics.auc(fpr, tpr)

    # Plots the ROC curve and show area.
    plt.plot(fpr, tpr, label='AUC = %0.3f' % auc_res)

    plt.xlim([0, 0.2])
    plt.ylim([0.6, 1])
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(loc='lower right', prop={'size': 16})
    plt.show()


def reject_metrics(model, frr=1, far=1):
    '''
    Computes the confusion matrix for a given network.
    '''

    # Evaluate predictions using threshold

    def apply_threshold(y_score, t=0.5):
        return [1 if y >= t else 0 for idx, y in enumerate(y_score)]

    def search(y_true, y_score, frr_target, far_target):
        cond = {}

        # Quick hack for initial threshold level to hit 1% FRR a bit faster.
        t = 1e-9

        # Compute FAR for a fixed FRR
        while t < 1.0:

            tn, fp, fn, tp = confusion_matrix(y_true, apply_threshold(y_score, t)).ravel()

            far = (fp * 100) / (fp + tn)
            frr = (fn * 100) / (fn + tp)

            if frr >= frr_target:
                cond['fixfrr'] = far, frr
            elif far >= far_target:
                cond['fixfar'] = far, frr
            elif far == frr:
                cond['farfrr'] = far, frr

            t *= 1.1

        # Return closest result if no good match found.
        return cond

    print('Network metrics:')

    # For each noise level
    for lvl in config.noise_snrs:
        # Make predictions
        y_true, y_score = test_predict(model, lvl)
        cond = search(y_true, y_score, frr, far)
        print('FAR: %0.2f%% for fixed FRR at %0.2f%% and noise level' % cond['fixfrr'], lvl)
        print('FRR: %0.2f%% for fixed FAR at %0.2f%% and noise level' % cond['fixfar'], lvl)
        print('FAR: %0.2f%% == FRR at %0.2f%% and noise level' % cond['farfrr'], lvl)


if __name__ == '__main__':
    set_seed()
    model = torch.load(r'D:\vk_test\pretrained\gru_large_epoch014.net')
    roc_auc(model, 0)
    reject_metrics(model, frr=1, far=1)