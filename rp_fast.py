import numpy as np
import random
from time import time
from copy import deepcopy
import pickle
import argparse
import torch

def binarize(arr):
    return np.where(arr < 0, -1, 1)

def encoding_rp(X_data, base_matrix, binary=False):
    enc_hvs = np.matmul(base_matrix, X_data.T)
    if binary:
        enc_hvs = binarize(enc_hvs)
    return enc_hvs.T

def max_match(enc_hv, class_hvs, norms=None):
    if norms != None:
        predicts = np.matmul(enc_hv, (class_hvs.T / norms.T))
    else:
        predicts = np.matmul(enc_hv, class_hvs.T)
    return predicts.argmax(axis=enc_hv.ndim - 1)

def train_hd(enc_train, enc_test, EPOCH=100, shuffle=True, log=False):
    D = len(enc_train[0])
    class_hvs = np.zeros((max(y_train)+1, D))
    n = 0
    for i in range(EPOCH):
        pickList = np.arange(0, len(enc_train))
        if shuffle: np.random.shuffle(pickList)
        correct = 0
        for j in pickList:
            predict = max_match(enc_train[j], class_hvs, None)
            if predict != y_train[j]:
                class_hvs[predict] -= enc_train[j]
                class_hvs[y_train[j]] += enc_train[j]
            else:
                correct += 1
        acc_train = correct/len(enc_train)
        if log: print(i+1, 'acc_train %.4f' %acc_train)
        if i == 0:
            predict = max_match(enc_test, class_hvs, None)
            acc_test1 = sum(predict == y_test)/len(y_test)
        if acc_train == 1 or i == EPOCH - 1:
            predict = max_match(enc_test, class_hvs, None)
            acc_test = sum(predict == y_test)/len(y_test)
            break
    return acc_train, acc_test1, acc_test, class_hvs


parser = argparse.ArgumentParser()
parser.add_argument('-D', default=8192)
parser.add_argument('-dataset', required=True)
args = parser.parse_args()
D = int(args.D)
benchmark = args.dataset

def get_dataset(benchmark, normalize=True, folder='/home/eagle/research/datasets/'):
    path = folder + '{}.pickle'.format(benchmark)
    with open(path, 'rb') as f:
        dataset = pickle.load(f, encoding='latin1')
    X_train, y_train, X_test, y_test = dataset
    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)
    assert X_train.ndim in [2, 3]
    if X_train.ndim == 3: #2D images, e.g. (60000, 32, 32)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    if np.max(X_train) > 1 and normalize:
        X_train = X_train/255.
        X_test = X_test/255.
    del dataset
    return X_train, y_train, X_test, y_test

assert benchmark in ['mnist', 'fmnist', 'cifar10', 'mnist_letter', 'face_caltech_cifar32', 'animal_faces', 'face_mask', 'intel', 'sign_language', 'waste']
if benchmark in ['mnist', 'fmnist', 'cifar10', 'mnist_letter', 'face_caltech_cifar32']:
    folder = '/home/eagle/research/datasets/'
else:
    folder = '/home/eagle/research/datasets_vision/{}/'.format(benchmark)

X_train, y_train, X_test, y_test = get_dataset(benchmark=benchmark, folder=folder)

n_class = np.unique(y_train).size
d = len(X_train[0])

B = np.random.uniform(-1, 1, (D, d))
B = np.where(B >= 0, 1, -1)
print('encoding started')
enc_train = encoding_rp(X_train, B, binary=True)
enc_test = encoding_rp(X_test, B, binary=True)
print('training started')
acc_train, acc_test1, acc_test, class_hvs = train_hd(enc_train, enc_test, EPOCH=100, shuffle=True, log=True)
print(acc_train, '\t', acc_test1, '\t', acc_test)