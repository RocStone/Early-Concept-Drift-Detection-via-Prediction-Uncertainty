import numpy as np
import sys
from river.datasets import synth
from river import datasets as rd
import numpy as np
import torch
class Interest():
    def __init__(self):
        self.current = [0, 1, 2]
        self.markov = self.generate_markov_matrix(10)

    def generate_markov_matrix(self, size=10):
        matrix = np.random.rand(size, size)
        np.fill_diagonal(matrix, matrix.diagonal() + 5)
        matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]
        return matrix

    def one_hot_encode(self, index, size=10):
        vector = np.zeros(size)
        vector[index] = 1
        return vector

    def sample_new_interests(self):
        new_interests = []
        for interest in self.current:
            transition_probabilities = np.dot(self.one_hot_encode(interest), self.markov)
            new_interest = np.random.choice(range(self.markov.shape[1]), p=transition_probabilities)
            new_interests.append(new_interest)
        self.current = new_interests
        return new_interests

def get_numpy_from_dict(d):
    num_feature = max([max(d[i][0].keys()) for i in range(len(d))]) + 1
    size = len(d)
    x = np.zeros((size, num_feature))
    y = np.zeros(size)
    for i in range(size):
        for j in d[i][0].keys():
            x[i][j] = d[i][0][j]
        y[i] = d[i][1]
    return x, y

def get_numpy_from_dict2(d):
    s = set()
    for i in range(len(d)):
        s.update(d[i][0].keys())
    
    str_int_map = {k: v for v, k in enumerate(s)}
    num_feature = len(s)
    size = len(d)
    x = np.zeros((size, num_feature))
    y = np.zeros(size)
    for i in range(size):
        for k in d[i][0].keys():
            j = str_int_map[k]
            x[i][j] = d[i][0][k]
        y[i] = d[i][1]
    return x, y

def get_numpy_from_dict3(d):
    s = set()
    s2 = set()
    for i in range(len(d)):
        s.update(d[i][0].keys())
        s2.update([d[i][1]])
    
    str_int_map = {k: v for v, k in enumerate(s)}
    str_int_map2 = {k:v for v, k in enumerate(s2)}
    num_feature = len(s)
    size = len(d)
    x = np.zeros((size, num_feature))
    y = np.zeros(size)
    for i in range(size):
        for k in d[i][0].keys():
            j = str_int_map[k]
            x[i][j] = d[i][0][k]
        y[i] = str_int_map2[d[i][1]]
    return x, y

def get_local_data(name):
    def airline(num2=58100, num1=0):
        # n = 5810462
        n = num2 - num1 # read a partial of this dataset
        d = 679 #there are only 13 features in the original dataset. One-hot encoding will results in having dimension of 679

        drift_times = [31, 67]

        X = []
        Y = []
        with open('data/airline.data') as file:
            i = 0
            for line in file:
                if  num1 <= i < num2:
                    fields = line.strip().split(',')
                    label = int(fields[len(fields)-1])

                    features = {0:1}
                    for j in range(len(fields)-1):
                        (index, val) = fields[j].split(':')
                        features[int(index)] = float(val)
                    X.append(features)
                    Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times
    
    def powersupply():
        n = 29928
        d = 2 + 1

        drift_times = [17, 47, 76]

        X = []
        Y = []

        with open('data/powersupply.arff') as file:
            i = 0
            for line in file:
                fields = line.strip().split(',')
                label = 1 if int(fields[2]) < 12 else -1
                features = {0: 1}
                for j in range(2):
                    features[j+1] = float(fields[j])
                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        max_0 = max(X[i][1] for i in range(len(X)))
        max_1 = max(X[i][2] for i in range(len(X)))
        for i in range(len(X)):
            X[i][1] = X[i][1]/max_0
            X[i][2] = X[i][2]/max_1
        return X, Y, n, d, drift_times

    if name == 'powersupply':
        X, Y, n, d, drift_times = powersupply()
    elif name == 'airline':
        X, Y, n, d, drift_times = airline()

    Y = np.array(Y)
    Y[Y==-1] = 0

    max_feature_idx = 0
    for i in range(len(X)):
        for k in X[i]:
            if k > max_feature_idx:
                max_feature_idx = k

    xx = np.zeros((len(X), max_feature_idx + 1))
    for i in range(len(X)):
        for k, v in X[i].items():
            xx[i][k] = v

    return xx, Y

def get_chunk(name, episode_length, chunk_size):
    j = -1
    v = -1
    d = []
    if name.startswith('sea'):
        noise_rate = int(name[3:]) / 100
        values = [0, 1, 2, 3]
        vlist = np.random.choice(values, chunk_size)
        for i in range(episode_length):
            if i % 10 == 0:
                j += 1
                while v == vlist[j]:
                    j +=1
                v = vlist[j]
            dataset = synth.SEA(variant=v, noise=noise_rate)
            d.extend([[x, y] for x, y in dataset.take(chunk_size)])
    elif name.startswith('sine'):
        values = [0, 1, 2, 3]
        vlist = np.random.choice(values, chunk_size)
        for i in range(episode_length):
            if i % 10 == 0:
                j += 1
                while v == vlist[j]:
                    j +=1
                v = vlist[j]
            dataset = synth.Sine(classification_function=v)
            d.extend([[x, y] for x, y in dataset.take(chunk_size)])
    elif name.startswith('mixed'):
        values = [0, 1]
        vlist = np.random.choice(values, chunk_size)
        for i in range(episode_length):
            if i % 10 == 0:
                j += 1
                while v == vlist[j]:
                    j +=1
                v = vlist[j]
            dataset = synth.Mixed(classification_function=v)
            d.extend([[x, y] for x, y in dataset.take(chunk_size)])
    elif name.startswith('elec2'):
        dataset = rd.Elec2()
        d = ([[x, y] for x, y in dataset.take(episode_length * chunk_size)])
    elif 'powersupply' in name:
        x, y = get_local_data('powersupply')
    elif 'airline' in name:
        x, y = get_local_data('airline')
    elif 'cifar10' in name:
        # x has been applied data aug
        path = './data/cifar10/x.pth'
        x_chunks = torch.load(path)
        path = './data/cifar10/y.pth'
        new_y = torch.load(path)

        return x_chunks, new_y, len(x_chunks)
    
    if name.startswith('elec2'):
        x, y = get_numpy_from_dict2(d)
        x = x[:45000]
        y = y[:45000]
    elif name == 'powersupply':
        x = x[:29000]
        y = y[:29000]
    elif name == 'airline':
        x = x[:58000]
        y = y[:58000]
    else:
        x, y = get_numpy_from_dict(d)

    episode_length = len(x) // 1000
    x = x.reshape(episode_length, -1, x.shape[1])
    y = y.reshape(episode_length, -1)
    return x, y, episode_length


def load_dataset_with_drift(name: str, chunk_size=1000, n_chunks=100, seed=42, drift_every_k_chunks=10):
    X, y, episode_length = get_chunk(name, n_chunks, chunk_size)
    return X, y, episode_length
