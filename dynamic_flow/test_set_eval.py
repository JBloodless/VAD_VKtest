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

    dataset_test = DatasetLoader(
        config.test_sets, config.noise_sets,
        noise_prob=0.7)
    generator_test = DataGenerator(dataset_test.mix_generator, batch_count=config.test_size)

    if noise_level not in config.noise_snrs:
        print('Error: invalid noise level!')
        return

    model.eval()
    generator_test.set_noise_level_db(noise_level)

    y_true, y_score = [], []

    for i in range(generator_test.batch_count):
        X, y = generator_test.get_batch(config.batch_size, n_frames=config.train_frames)
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


if __name__ == '__main__':
    set_seed()
    model = torch.load(r'models/cGRU_epoch029.net')
    roc_auc(model, 0)