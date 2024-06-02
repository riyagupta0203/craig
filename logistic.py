import glob
import argparse

import numpy as np
import time
from os import path
from warnings import simplefilter

# Ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
np.seterr(all='ignore')

import random


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.sum(e, axis=1, keepdims=True)  # ndim = 2


class LogisticRegression(object):
    def __init__(self, dim, num_class):
        self.binary = num_class == 1
        self.W = np.zeros((dim, num_class))  # Initialize weights
        self.b = np.zeros((1, num_class))  # Initialize bias

    def activation(self, input, params=None):
        W, b = params if params is not None else (self.W, self.b)
        if self.binary:
            return sigmoid(np.dot(input, W) + b)
        else:
            return softmax(np.dot(input, W) + b)

    def loss(self, input, label, l2_reg=0.00, params=None):
        activation = self.activation(input, params)
        if self.binary:
            cross_entropy = - np.mean(label * np.log(activation) + (1 - label) * np.log(1 - activation))
        else:
            cross_entropy = - np.mean(np.sum(label * np.log(activation), axis=1))
        return cross_entropy + l2_reg * np.linalg.norm(self.W) ** 2 / 2

    def predict(self, input, params=None):
        return self.activation(input, params)

    def accuracy(self, input, label, params=None):
        if self.binary:
            return np.mean(np.isclose(np.rint(self.predict(input, params)), label))
        else:
            return np.mean(np.argmax(self.predict(input, params), axis=1) == np.argmax(label, axis=1))

    def gradient(self, input, label, l2_reg=0.00, params=None):
        W, b = params if params is not None else (self.W, self.b)
        p_y_given_x = self.activation(input, (W, b))
        d_y = label - p_y_given_x
        d_W = -np.dot(input.T, d_y) - l2_reg * W
        d_b = -np.mean(d_y, axis=0, keepdims=True)
        return np.array([d_W, d_b])


class Optimizer(object):

    @staticmethod
    def order_elements(shuffle, n, seed=1234):
        if shuffle == 0:
            indices = np.arange(n)
        elif shuffle == 1:
            indices = np.random.permutation(n)
        elif shuffle == 2:
            indices = np.random.randint(0, n, n)
        else:  # fixed permutation
            np.random.seed(seed)
            indices = np.random.permutation(n)
        return indices

    def optimize(self, method, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        if method == 'sgd':
            return self.sgd(model, data, labels, weights, num_epochs, shuffle, lr, l2_reg)
        elif method == 'saga':
            return self.saga(model, data, labels, weights, num_epochs, shuffle, lr, l2_reg)
        elif method == 'svrg':
            return self.svrg(model, data, labels, weights, num_epochs, shuffle, lr, l2_reg)
        else:
            print('Optimizer is not defined!')

    def sgd(self, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        n = len(data)
        W = [[] for _ in range(num_epochs)]
        T = np.empty(num_epochs)

        time.sleep(.1)
        start_epoch = time.process_time()

        for epoch in range(num_epochs):
            indices = self.order_elements(shuffle, n)
            for i in indices:
                grads = model.gradient(data[i].reshape(1, -1), labels[i].reshape(1, -1), l2_reg / n) * weights[i]
                model.W -= lr[epoch] * grads[0]
                model.b -= lr[epoch] * grads[1]
            W[epoch] = (model.W.copy(), model.b.copy())
            T[epoch] = (time.process_time() - start_epoch)
        return W, T

    def saga(self, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        n = len(data)
        W = [[] for _ in range(num_epochs)]
        T = np.empty(num_epochs)

        time.sleep(.1)
        start_epoch = time.process_time()

        saved_grads = np.array([model.gradient(data[i].reshape(1, -1), labels[i].reshape(1, -1), l2_reg / n) * weights[i] for i in range(n)])
        avg_saved_grads = saved_grads.mean(axis=0)

        for epoch in range(num_epochs):
            indices = self.order_elements(shuffle, n)
            for i in indices:
                grads = model.gradient(data[i].reshape(1, -1), labels[i].reshape(1, -1), l2_reg / n) * weights[i]
                model.W -= lr[epoch] * (grads[0] - saved_grads[i][0] + avg_saved_grads[0])
                model.b -= lr[epoch] * (grads[1] - saved_grads[i][1] + avg_saved_grads[1])
                avg_saved_grads += (grads - saved_grads[i]) / n
                saved_grads[i] = grads

            W[epoch] = (model.W.copy(), model.b.copy())
            T[epoch] = (time.process_time() - start_epoch)
        return W, T

    def svrg(self, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        n = len(data)
        W = [[] for _ in range(num_epochs)]
        T = np.empty(num_epochs)

        time.sleep(.1)
        start_epoch = time.process_time()

        for epoch in range(num_epochs):
            init_grads = np.array([model.gradient(data[i].reshape(1, -1), labels[i].reshape(1, -1), l2_reg / n) * weights[i] for i in range(n)])
            avg_init_grads = np.mean(init_grads, axis=0)

            indices = self.order_elements(shuffle, n)
            for i in indices:
                grads = model.gradient(data[i].reshape(1, -1), labels[i].reshape(1, -1), l2_reg / n) * weights[i]
                model.W -= lr[epoch] * (grads[0] - init_grads[i][0] + avg_init_grads[0])
                model.b -= lr[epoch] * (grads[1] - init_grads[i][1] + avg_init_grads[1])

            W[epoch] = (model.W.copy(), model.b.copy())
            T[epoch] = (time.process_time() - start_epoch)
        return W, T


def load_dataset(dataset, normalize=False):
    DATASET_DIR = '/tmp/data/'
    if dataset == 'covtype':
        print(f'Loading {dataset}')
        from ucimlrepo import fetch_ucirepo 
        covertype = fetch_ucirepo(id=31) 
        X = covertype.data.features.to_numpy()
        y = covertype.data.targets.to_numpy()

        N = len(X)
        NUM_TRAINING = N // 2  # Using integer division
        NUM_VALIDATION = NUM_TRAINING + N // 4
        NUM_VALIDATION = min(NUM_VALIDATION, N)
        sample = np.arange(N)
        np.random.seed(0)
        np.random.shuffle(sample)
        train_sample = sample[:NUM_TRAINING]
        val_sample = sample[NUM_TRAINING:NUM_VALIDATION]
        test_sample = sample[NUM_VALIDATION:]

        X_train, y_train = X[train_sample, :], y[train_sample]
        X_val, y_val = X[val_sample, :], y[val_sample]
        X_test, y_test = X[test_sample, :], y[test_sample]

    elif dataset == 'ijcnn1':
        print(f'Loading {dataset}')
        X_train, y_train = util.load_dataset('ijcnn1.tr', DATASET_DIR)
        X_test, y_test = util.load_dataset('ijcnn1.t', DATASET_DIR)
        X_val, y_val = X_test, y_test

    elif dataset == 'combined':
        print(f'Loading {dataset}')
        X_train, y_train = util.load_dataset('combined_scale', DATASET_DIR)
        X_test, y_test = util.load_dataset('combined_scale.t', DATASET_DIR)
        X_0, y_0 = X_train[y_train == 0], y_train[y_train == 0]
        X_1, y_1 = X_train[y_train == 1], y_train[y_train == 1]
        X_2, y_2 = X_train[y_train == 2], y_train[y_train == 2]

        X_1, y_1 = X_1[:len(X_2)], y_1[:len(X_2)]
        X_train = np.vstack([X_1, X_2])
        y_train = np.hstack([y_1, y_2])
        y_train[y_train == 2] = 0

        X_0, y_0 = X_test[y_test == 0], y_test[y_test == 0]
        X_1, y_1 = X_test[y_test == 1], y_test[y_test == 1]
        X_2, y_2 = X_test[y_test == 2], y_test[y_test == 2]

        X_1, y_1 = X_1[:len(X_2)], y_1[:len(X_2)]
        X_test = np.vstack([X_1, X_2])
        y_test = np.hstack([y_1, y_2])
        y_test[y_test == 2] = 0
        X_val, y_val = X_test, y_test

    else:
        print('Dataset does not exist')
        return

    if normalize:
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    print(f'Loading completed')
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    import util

    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', type=int, default=0, help='Shuffle type')
    parser.add_argument('--dataset', type=str, default='covtype', help='Dataset type')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--reg', type=float, default=0.0, help='L2 regularization term')
    parser.add_argument('--method', type=str, default='sgd', help='Optimization method')
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.dataset, normalize=True)
    print(f'Dataset: {args.dataset}\nMethod: {args.method}\nShuffle: {args.shuffle}\nEpoch: {args.epoch}\n')

    y_train_one_hot = np.zeros((len(y_train), 2))
    y_train_one_hot[np.arange(len(y_train)), y_train] = 1

    y_val_one_hot = np.zeros((len(y_val), 2))
    y_val_one_hot[np.arange(len(y_val)), y_val] = 1

    y_test_one_hot = np.zeros((len(y_test)), 2)
    y_test_one_hot[np.arange(len(y_test)), y_test] = 1

    num_class = y_train_one_hot.shape[1]
    num_feature = X_train.shape[1]

    print(f'num_class: {num_class}\nnum_feature: {num_feature}\n')

    model = LogisticRegression(num_feature, num_class)
    optimizer = Optimizer()

    weights = np.ones(len(X_train))
    step = 1e-1
    lr = [step / np.sqrt(i + 1) for i in range(args.epoch)]
    params = optimizer.optimize(args.method, model, X_train, y_train_one_hot, weights, args.epoch, args.shuffle, lr, args.reg)
    W, T = params

    acc = []
    for i in range(args.epoch):
        acc.append(model.accuracy(X_val, y_val_one_hot, W[i]))
        print(f'accuracy of epoch {i} is: {acc[i]:.3f}')
    print(f'best validation accuracy is {np.max(acc):.3f}')

