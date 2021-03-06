import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from config import config
from data_generator import DataGenerator
from loader_preproc import DatasetLoader
from tqdm import tqdm


def accuracy(out, y):
    '''
    Calculate accuracy of model where
    out.shape = (64, 2) and y.shape = (64)
    '''
    out = torch.max(out, 1)[1].float()
    eq = torch.eq(out, y.float()).float()
    return torch.mean(eq)

def set_seed(seed = 666):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=0)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def model_path(epoch, title):
    part = os.path.join(os.getcwd(), 'models', title)
    if epoch >= 0:
        return part + '_epoch' + str(epoch).zfill(3) + '.net'
    else:
        return part + '.net'


def save_model(model, epoch, title):
    if not os.path.exists(os.getcwd() + '/models'):
        os.makedirs(os.getcwd() + '/models')
    torch.save(model, model_path(epoch, title))


# def load_model(title, epoch=None):
#     if torch.cuda.is_available():
#         return torch.load(model_path(epoch, title))
#     else:
#         return torch.load(model_path(epoch, title), map_location='cpu')


def train_model(model, fig, noise_level='None', epochs=config.epochs, lr=1e-3, use_adam=True,
                weight_decay=1e-5, momentum=0.9, use_focal_loss=True, gamma=0.0,
                early_stopping=False, patience=25,
                auto_save=True, title='cGRU', verbose=True):
    '''
    Full-featured training of a given neural network.
    A number of training parameters are optionally adjusted.
    If verbose is True, the training progress is continously
    plotted at the end of each epoch.
    If auto_save is True, the model will be saved every epoch.
    '''

    # Set up an instance of data generator using default partitions
    dataset_tr = DatasetLoader(
       config.train_sets, config.noise_sets,
        noise_prob=0.7)
    generator_tr = DataGenerator(dataset_tr.mix_generator, batch_count=config.train_size)

    dataset_val = DatasetLoader(
        config.val_sets, config.noise_sets,
        noise_prob=0.7)
    generator_val = DataGenerator(dataset_val.mix_generator, batch_count=config.val_size)

    # Instantiate the chosen loss function
    if use_focal_loss:
        criterion = FocalLoss(gamma)
        levels = config.noise_snrs
    else:
        criterion = nn.CrossEntropyLoss()
        levels = [noise_level]

    model.to(config.device)
    criterion.to(config.device)

    # Instantiate the chosen optimizer with the parameters specified
    if use_adam:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # If verbose, print starting conditions
    if verbose:
        print(f'Initiating training of {title}...\n\nLearning rate: {lr}')
        _trsz = generator_tr.batch_count * len(config.noise_snrs) if use_focal_loss else generator_tr.batch_count
        _vlsz = generator_val.batch_count * len(config.noise_snrs) if use_focal_loss else generator_val.batch_count
        print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
        print(f'Train/val partitions: {_trsz} | {_vlsz}')
        _critstr = f'Focal Loss (?? = {gamma})' if use_focal_loss else f'Cross-Entropy ({noise_level} dB)'
        _optmstr = f'Adam (decay = {weight_decay})' if use_adam else f'SGD (momentum = {momentum})'
        _earlstr = f'Early Stopping (patience = {patience})' if early_stopping else str(epochs)
        _autostr = 'Enabled' if auto_save else 'DISABLED'
        print(f'Criterion: {_critstr}\nOptimizer: {_optmstr}')
        print(f'Max epochs: {_earlstr}\nAuto-save: {_autostr}')

    model.train()
    stalecount, maxacc = 0, 0

    def plot(losses, accs, val_losses, val_accs, fig=fig):
        '''
        Continously plots the training/validation loss and accuracy
        of the model being trained. This functions is only called if
        verbose is True for the training session.
        '''
        # plt.ion()
        plt.clf()
        e = [i for i in range(len(losses))]
        # fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(e, losses, label='Loss (Training)')

        plt.plot(e, val_losses, label='Loss (Validation)')

        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(e, accs, label='Accuracy (Training)')

        plt.plot(e, val_accs, label='Accuracy (Validation)')

        plt.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def run(model, generator, optimize=False):

        epoch_loss, epoch_acc, level_acc = 0, [], []
        batches = generator.batch_count
        print('batches in run ', batches)
        num_batches = batches * len(config.noise_snrs)
        print('num_batches in run ', num_batches)

        # In case we apply focal loss, we want to include all noise levels

        # Helper function responsible for running a batch
        def run_batch(X, y, epoch_loss, epoch_acc):

            X = Variable(torch.from_numpy(np.array(X)).float())
            y = Variable(torch.from_numpy(np.array(y))).long()

            X = X.to(config.device)
            y = y.to(config.device)

            out = model(X)
            print('net_out in run_batch', out.shape)

            # Compute loss and accuracy for batch
            batch_loss = criterion(out, y)
            batch_acc = accuracy(out, y)

            # If training session, initiate backpropagation and optimization
            if optimize == True:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            if torch.cuda.is_available():
                batch_acc = batch_acc.cpu()
                batch_loss = batch_loss.cpu()

            # Accumulate loss and accuracy for epoch metrics
            epoch_loss += batch_loss.data.numpy() / float(config.batch_size)
            epoch_acc.append(batch_acc.data.numpy())

            return epoch_loss, epoch_acc

        # For each noise level scheduled
        for lvl in levels:
            print('noise SNR ', lvl)

            # Set up generator for iteration
            generator.set_noise_level_db(lvl)

            # For each batch in noise level
            for i in tqdm(range(batches)):
                # Get a new batch and run it
                X, y = generator.get_batch(config.batch_size, config.train_frames)
                print('get_batch_train/val ', np.array(X).shape, np.array(y).shape)
                temp_loss, temp_acc = run_batch(X, y, epoch_loss, epoch_acc)
                epoch_loss += temp_loss / float(num_batches)
                level_acc.append(np.mean(temp_acc))

        return epoch_loss, np.mean(level_acc)

    losses, accs, val_losses, val_accs = [], [], [], []

    if verbose:
        start_time = time.time()

    # Iterate over training epochs
    for epoch in range(epochs):
        print('Epoch ', epoch)

        # Calculate loss and accuracy for that epoch and optimize
        loss, acc = run(model, generator=generator_tr, optimize=True)
        losses.append(loss)
        accs.append(acc)

        # If validation data is available, calculate validation metrics
        model.eval()
        val_loss, val_acc = run(model, generator=generator_val)
        # print(val_loss, val_acc)
        # return
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        model.train()

        # Early stopping algorithm.
        # If validation accuracy does not improve for
        # a set amount of epochs, abort training and retrieve
        # the best model (according to validation accuracy)
        if epoch > 0 and val_accs[-1] <= maxacc:
            stalecount += 1
            if stalecount > patience and early_stopping:
                return
        else:
            stalecount = 0
            maxacc = val_accs[-1]

        if auto_save:
            save_model(model, epoch, title)

        # Optionally plot performance metrics continously
        if verbose:

            plot(losses, accs, val_losses, val_accs)


if __name__ == '__main__':
    from models import ConvGRU
    print(config.device)

    set_seed()
    plt.ion()
    fig = plt.figure()
    cgru = ConvGRU(large=True)
    train_model(cgru, fig, gamma=2)
