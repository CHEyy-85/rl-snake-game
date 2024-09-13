import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython import display
import random


E_INIT = 1
E_FINAL = 0.01
TAU = 5e-2
DECAY_RATE = 0.95
MINI_BATCH_SIZE = 64
NUM_STEPS_UPDATE = 4

def plot_performace(score, mean_score_last_10, mean_score):
    plt.ion()
    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf()
    plt.plot(score, label = 'Score')
    plt.plot(mean_score_last_10, label = 'Mean for last 10 games')
    plt.plot(mean_score, label = 'Mean Score')
    plt.legend()
    plt.title('AI Snake Player - Training')
    plt.xlabel('# of Games')
    plt.ylabel('Performance - pts')
    plt.ylim(ymin = 0)
    plt.text(len(score)-1, score[-1], str(score[-1]))
    plt.text(len(mean_score_last_10)-1, mean_score_last_10[-1], str(round(mean_score_last_10[-1],2)))
    plt.text(len(mean_score)-1, mean_score[-1], str(round(mean_score[-1],2)))
    plt.show()
    plt.pause(0.05)

def get_epsilon(num_of_games):
    return max(E_INIT * (DECAY_RATE)**num_of_games, E_FINAL)

def target_qNet_softupdate(q_net, target_q_net):
    for target_param, param in zip(target_q_net.parameters(), q_net.parameters()):
        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)



def check_update_condition(num_of_games, memory):
    if num_of_games % 4 == 0 and len(memory) >= MINI_BATCH_SIZE:
        return True
    else:
        return False

def get_experiences_for_replay(memory, mini_batch_size):
    # Sample a random mini-batch of experiences from memory
    experiences = random.sample(memory, mini_batch_size)
    
    # Extract states, actions, rewards, next_states, and done_values from experiences
    states = torch.tensor([exp.state for exp in experiences if exp is not None], dtype=torch.float32)
    actions = torch.tensor([exp.action for exp in experiences if exp is not None], dtype=torch.long)
    rewards = torch.tensor([exp.reward for exp in experiences if exp is not None], dtype=torch.float32)
    next_states = torch.tensor([exp.next_state for exp in experiences if exp is not None], dtype=torch.float32)
    done_values = torch.tensor([exp.done for exp in experiences if exp is not None], dtype=torch.float32)

    return states, actions, rewards, next_states, done_values
