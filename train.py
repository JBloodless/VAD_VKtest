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


def accuracy(out, y):
    '''
    Calculate accuracy of model where
    out.shape = (64, 2) and y.shape = (64)
    '''
    out = torch.max(out, 1)[1].float()
    eq = torch.eq(out, y.float()).float()
    return torch.mean(eq)


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


def net_path(epoch, title):
    part = os.path.join(os.getcwd(), 'models', title)
    if epoch >= 0:
        return part + '_epoch' + str(epoch).zfill(3) + '.net'
    else:
        return part + '.net'


def save_model(net, epoch, title):
    if not os.path.exists(os.getcwd() + '/models'):
        os.makedirs(os.getcwd() + '/models')
    torch.save(net, net_path(epoch, title))


def load_model(title, epoch=None):
    if torch.cuda.is_available():
        return torch.load(net_path(epoch, title))
    else:
        return torch.load(net_path(epoch, title), map_location='cpu')


def train_model(model, noise_level='None', epochs=30, lr=1e-3, use_adam=True,
                weight_decay=1e-5, momentum=0.9, use_focal_loss=True, gamma=0.0,
                early_stopping=False, patience=25,
                auto_save=True, title='net', verbose=True):
    '''
    Full-featured training of a given neural network.
    A number of training parameters are optionally adjusted.
    If verbose is True, the training progress is continously
    plotted at the end of each epoch.
    If auto_save is True, the model will be saved every epoch.
    '''

    # Set up an instance of data generator using default partitions
    dataset_tr = DatasetLoader(
        [r'D:\Datasets\LibriSpeech\train-clean-360', r'D:\Datasets\LibriSpeech\train-other-500'], r'D:\Datasets\noises',
        noise_prob=0.7)
    generator_tr = DataGenerator(dataset_tr.mix_generator, batch_count=10000)

    dataset_val = DatasetLoader(
        [r'D:\Datasets\LibriSpeech\train-clean-100'], r'D:\Datasets\noises',
        noise_prob=0.7)
    generator_val = DataGenerator(dataset_val.mix_generator, batch_count=1000)

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
        _trsz = generator_tr.batch_size * len(config.noise_snrs) if use_focal_loss else generator_tr.batch_size
        _vlsz = generator_val.batch_size * len(config.noise_snrs) if use_focal_loss else generator_val.batch_size
        print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
        print(f'Train/val partitions: {_trsz} | {_vlsz}')
        _critstr = f'Focal Loss (γ = {gamma})' if use_focal_loss else f'Cross-Entropy ({noise_level} dB)'
        _optmstr = f'Adam (decay = {weight_decay})' if use_adam else f'SGD (momentum = {momentum})'
        _earlstr = f'Early Stopping (patience = {patience})' if early_stopping else str(epochs)
        _autostr = 'Enabled' if auto_save else 'DISABLED'
        print(f'Criterion: {_critstr}\nOptimizer: {_optmstr}')
        print(f'Max epochs: {_earlstr}\nAuto-save: {_autostr}')

    model.train()
    stalecount, maxacc = 0, 0

    def plot(losses, accs, val_losses, val_accs):
        '''
        Continously plots the training/validation loss and accuracy
        of the model being trained. This functions is only called if
        verbose is True for the training session.
        '''
        plt.ion()
        e = [i for i in range(len(losses))]
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(e, losses, label='Loss (Training)')

        if generator_val.size != 0:
            plt.plot(e, val_losses, label='Loss (Validation)')

        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(e, accs, label='Accuracy (Training)')

        if generator_val.size != 0:
            plt.plot(e, val_accs, label='Accuracy (Validation)')

        plt.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def run(model, generator, optimize=False):
        '''
        This function constitutes a single epoch.
        Snippets are loaded into memory and their associated
        frames are loaded as generators. As training progresses
        and new frames are needed, they are generated by the iterator,
        and are thus not stored in memory when not used.
        If optimize is True, the associated optimizer will backpropagate
        and adjust network weights.
        Returns the average sample loss and accuracy for that epoch.
        '''
        epoch_loss, epoch_acc, level_acc = 0, [], []
        batches = generator.batch_count
        num_batches = batches * len(config.noise_snrs)

        # In case we apply focal loss, we want to include all noise levels

        # Helper function responsible for running a batch
        def run_batch(X, y, epoch_loss, epoch_acc):

            X = Variable(torch.from_numpy(np.array(X)).float())
            y = Variable(torch.from_numpy(np.array(y))).long()

            X = X.to(config.device)
            y = y.to(config.device)

            out = model(X)

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

            # Set up generator for iteration
            generator.set_noise_level_db(lvl)

            # For each batch in noise level
            for i in range(batches):
                # Get a new batch and run it
                X, y = generator.get_batch(config.batch_size)
                temp_loss, temp_acc = run_batch(X, y, epoch_loss, epoch_acc)
                epoch_loss += temp_loss / float(num_batches)
                level_acc.append(np.mean(temp_acc))

        return epoch_loss, np.mean(level_acc)

    losses, accs, val_losses, val_accs = [], [], [], []

    if verbose:
        start_time = time.time()

    # Iterate over training epochs
    for epoch in range(epochs):

        # Calculate loss and accuracy for that epoch and optimize
        loss, acc = run(model, generator=generator_tr, optimize=True)
        losses.append(loss)
        accs.append(acc)

        # If validation data is available, calculate validation metrics
        if generator_val.batch_size != 0:
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

            # Print measured wall-time of first epoch
            if epoch == 0:
                dur = str(int((time.time() - start_time) / 60))
                print(f'\nEpoch wall-time: {dur} min')

            plot(losses, accs, val_losses, val_accs)
