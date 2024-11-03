import numpy as np
import random
import sys
from time import time
from copy import deepcopy
import pickle
import argparse
import multiprocessing

def binarize(arr):
    return np.where(arr < 0, -1, 1)

def max_match(enc_hv, class_hvs, norms=None):
    if norms != None:
        predicts = np.matmul(enc_hv, (class_hvs.T / norms.T))
    else:
        predicts = np.matmul(enc_hv, class_hvs.T)
    return predicts.argmax(axis=enc_hv.ndim - 1)

parser = argparse.ArgumentParser()
parser.add_argument('-D', default=8192)
parser.add_argument('-dataset', required=True)
parser.add_argument('-binarize', action='store_true')
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

L = 64
lvl_hvs = []
temp = [-1]*(D//2) + [1]*(D//2)
np.random.shuffle(temp)
lvl_hvs.append(temp)
change_list = np.arange(0, D)
np.random.shuffle(change_list)
cnt_toChange = int(D/2 / (L-1))
for i in range(1, L):
    temp = np.array(lvl_hvs[i-1])
    temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]] = -temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]]
    lvl_hvs.append(list(temp))
lvl_hvs = np.array(lvl_hvs, dtype=np.int8)

id_hvs = []
for i in range(d):
    id_hvs.append(np.random.choice([-1, 1], size=D))
id_hvs = np.array(id_hvs, dtype=np.int8)

x_min = np.min(X_train)
x_max = np.max(X_train)
bin_len = (x_max - x_min)/float(L)


def encoding_idlv(data_sub):
    enc_hvs = []
    for i in range(len(data_sub)):
        sum_ = np.array([0] * D)
        for j in range(len(data_sub[i])):
            bin_ = min( np.round((data_sub[i][j] - x_min)/bin_len), L-1)
            bin_ = int(bin_)
            sum_ += lvl_hvs[bin_]*id_hvs[j]
        enc_hvs.append(sum_)
    return enc_hvs


def encode(X_data):
    n_core = 10
    X_data_splits = np.array_split(X_data, n_core)
    pool = multiprocessing.Pool(processes=n_core)
    results = pool.map(encoding_idlv, X_data_splits)
    pool.close()
    pool.join()
    results = [item for sublist in results for item in sublist]
    return results
    

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


print('encoding started')
enc_train = encode(X_train)
enc_test = encode(X_test)
enc_train = np.array(enc_train)
enc_test = np.array(enc_test)
if args.binarize:
    enc_train = binarize(enc_train)
    enc_test = binarize(enc_test)

print('training started')
acc_train, acc_test1, acc_test, class_hvs = train_hd(enc_train, enc_test, EPOCH=100, shuffle=True, log=True)
print(acc_train, '\t', acc_test1, '\t', acc_test)