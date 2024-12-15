import os, random, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt, seaborn as sns
import numpy as np
from tqdm import trange
import sys
from ei_detector import EIKMEANS
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
os.makedirs('results', exist_ok=True)
from collections import defaultdict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--RL_model', type=str, default='None')
parser.add_argument('--dataset_name', type=str, default='airline')
parser.add_argument('--classifier_name', type=str, default='DNN')
parser.add_argument('--continual', type=str, default='True')
parser.add_argument('--optimizer_type', type=str, default='Adam')
parser.add_argument('--threshold', type=float, default=-5)
parser.add_argument('--model_num', type=float, default=5)
args = parser.parse_args()
args.continual = args.continual == 'True'

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from classifiers_ensemble import MyClassifier
from load_dataset import load_dataset_with_drift


class RealEnv():
    def __init__(self, classifier_name, optimizer_type, dataset_name, chunk_size, n_chunks, seed, drift_every_k_chunks) -> None:
        super().__init__()
        self.classifier_name = classifier_name
        self.optimizer_type = optimizer_type
        self.dataset_name = dataset_name
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.seed = seed
        self.drift_every_k_chunks = drift_every_k_chunks
        self.X, self.y, self.n_chunks = load_dataset_with_drift(self.dataset_name, self.chunk_size, self.n_chunks, self.seed, self.drift_every_k_chunks)
        self.classifier = MyClassifier(classifier_type=self.classifier_name, optimizer_type=self.optimizer_type, dataset_name=self.dataset_name, first_x=self.X[0], first_y=self.y[0], totaly=self.y, model_num=args.model_num)
        self.timestep = 0
        self.accs = []
        self.ei_detector = EIKMEANS(k=5)

    def step(self, action):
        if self.timestep == 0:
            self.classifier.train(self.X[self.timestep], self.y[self.timestep], self.y)
            min_pvalue = 1
            acc = 0.5
            reward = acc * 100
            state = torch.FloatTensor([acc, np.log10(min_pvalue + 1e-100)]).to(device)
            self.accs.append(acc)
            self.timestep += 1
            return np.log10(min_pvalue + 1e-100), reward, False, {}, state


        if action == 1:
            min_pvalue = 1
            self.ei_detector.residuals_chunks = []
            self.ei_detector.accs = []
            self.ei_detector.correct_pred = []
            # self.classifier.train(self.X[self.timestep-1], self.y[self.timestep-1])
            if self.dataset_name == 'cifar10':
                recent_x = self.X[self.timestep-self.ei_detector.ei_split_point:self.timestep].reshape((-1, 3, 32, 32))
                recent_y = self.y[self.timestep-self.ei_detector.ei_split_point:self.timestep].reshape((-1))
            else:
                recent_x = self.X[self.timestep-self.ei_detector.ei_split_point:self.timestep].reshape((-1, self.X.shape[-1]))
                recent_y = self.y[self.timestep-self.ei_detector.ei_split_point:self.timestep].reshape((-1))

            # recent_x, recent_y = self.X[self.timestep-1], self.y[self.timestep-1]
            if self.dataset_name != 'cifar10':
                self.classifier.reset_model(classifier_type=self.classifier_name, x=recent_x, y=recent_y)
                self.classifier.train(recent_x, recent_y, self.y, epochs=100)
                if self.dataset_name == 'airline':
                    self.classifier.reset_lr()
            else:
                # self.classifier.classifier.reset()
                self.classifier.train(recent_x, recent_y, self.y)
                self.classifier.finetune(recent_x, recent_y, self.y)


        _, acc, residuals, correct, _ = self.classifier.predict(self.X[self.timestep], self.y[self.timestep])

        reward = acc * 100
        self.accs.append(acc)
        self.ei_detector.accs.append(acc)
        self.ei_detector.residuals_chunks.append(residuals)
        self.ei_detector.correct_pred.append(correct)        
        if acc >= 0.01:
            min_pvalue  = self.ei_detector.compute_pvalue(self.dataset_name)
        else:
            min_pvalue = 0
        if args.continual:
            self.classifier.train(self.X[self.timestep], self.y[self.timestep], self.y)
        self.timestep += 1  # increment timeste

        if self.timestep >= self.n_chunks:
            done = True
        else:
            done = False

        if self.timestep >= self.n_chunks:
            done = True
        else:
            done = False

        state = torch.FloatTensor([acc, np.log10(min_pvalue + 1e-100)]).to(device)
        if state[1].item() > 0:
            print('strange')

        return np.log10(min_pvalue + 1e-100), reward, done, {}, state

    
class Agent:
    def __init__(self):
        self.accs = []

    def determine_action(self, pvalue, threshold, random_action):
        if pvalue < threshold:
            return 1, random_action, threshold
        else:
            return 0, random_action, threshold


    def select_action(self, state, accs):
        self.accs.append(accs[-1])
        value = state[1].item()
        threshold = args.threshold
        return self.determine_action(value, threshold, True)

agent = Agent()

class DataStorage:
    def __init__(self):
        self.timestep_acc = []
        self.episode_acc = []
        self.timestep_reward = []
        self.episode_reward = []
        self.episode_detected_times = []
        self.episode_random_action_times = []
        self.actions_selected_by_agent = []
        self.is_action_output_by_policy_network = []
        self.timestep_pvalue = []
        self.qnet_loss = []
        self.transitions = defaultdict(list)

    def save(self, file_name='results/data_storage.pkl'):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
    
    def load(self, file_name='results/data_storage.pkl'):
        with open(file_name, 'rb') as file:
            loaded_data = pickle.load(file)
        self.__dict__.update(loaded_data.__dict__)

def save_results_to_csv(dataset_name, classifier, optimizer, RL_model, acc, episode_num, continual, threshold, file_path='./result.csv'):
    data = {
        'dataset_name': [dataset_name],
        'classifier': [classifier],
        'optimizer': [optimizer],
        'RL_model': [RL_model],
        'acc': [acc],
        'episode_num': [episode_num],
        'continual': [continual],
        'threshold': [threshold]
    }
    import pandas as pd
    df = pd.DataFrame(data)
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False, mode='w')
    else:
        df.to_csv(file_path, index=False, mode='a', header=False)


# Usage
record = DataStorage()

num_episodes = 100
classifier_name = args.classifier_name
optimizer_type = args.optimizer_type                                  
dataset_name = args.dataset_name                             
chunk_size = 1000
n_chunks = 100
drift_every_k_chunks = 10

for episode in trange(num_episodes):
    detected_times = 0
    random_action_times = 0
    seed = episode
    np.random.seed(seed)
    env =RealEnv(classifier_name, optimizer_type, dataset_name, chunk_size, n_chunks, seed, drift_every_k_chunks)
    pvalue, reward, done, _, state = env.step(0)
    done = False
    total_reward = []
    n_step_buffer = []

    if episode >= 30:
        epsilon = max(0.01, epsilon * 0.995)

    record.timestep_acc.append([])
    record.timestep_reward.append([])
    record.actions_selected_by_agent.append([])
    record.is_action_output_by_policy_network.append([])
    record.timestep_pvalue.append([])

    while not done:
        action, random_action, threshold = agent.select_action(state, env.accs)

        record.timestep_acc[-1].append(state[0].item())
        record.timestep_pvalue[-1].append(state[1].item())
        record.actions_selected_by_agent[-1].append(action)
        record.is_action_output_by_policy_network[-1].append(not random_action)
        record.timestep_reward[-1].append(reward)
        random_action_times += int(random_action)
        detected_times += int(action == 1)
        if action == 1:
            print('reset', flush=True)

        pvalue, reward, done, _, next_state = env.step(action) 
        print(env.timestep, pvalue, state[0].item())

        record.transitions['acc'].append(state[0].item())
        record.transitions['pvalue'].append(state[1].item())
        record.transitions['action'].append(action)
        record.transitions['reward'].append(reward)
        state = next_state

    record.episode_detected_times.append(detected_times)
    record.episode_random_action_times.append(random_action_times)
    record.episode_reward.append(np.mean(record.timestep_reward[-1]))
    record.episode_acc.append(np.mean(env.accs[1:]))
    print(f'{np.mean(env.accs[1:])}')
        

acc = np.mean(record.episode_acc)

save_results_to_csv(
    dataset_name=args.dataset_name,
    classifier=env.classifier_name,
    optimizer=env.optimizer_type,
    RL_model=args.RL_model,
    acc=acc,
    episode_num=num_episodes,
    continual=args.continual,
    threshold=args.threshold
)

