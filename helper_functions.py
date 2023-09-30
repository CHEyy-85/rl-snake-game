import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display
import random


E_INIT = 1
E_FINAL = 0.01
TAU = 1e-2
DECAY_RATE = 0.995
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

def target_qNet_softupdate(qNet, targetQNet):
    for qNetWs, targetQNetWs in zip(qNet.weights, targetQNet.weights):
        targetQNetWs.assign(TAU * qNetWs + (1-TAU) * targetQNetWs)

def check_update_condition(num_of_games, memory):
    if num_of_games % 4 == 0 and len(memory) >= MINI_BATCH_SIZE:
        return True
    else:
        return False
    
def get_experiences_for_replay(memory):
    # memory is a deque of namedtuples ['state', 'action', 'reward', 'next_state', 'done']
    experiences = random.sample(memory, MINI_BATCH_SIZE)
    states = tf.convert_to_tensor(np.array([exp.state for exp in experiences if exp is not None]), dtype = tf.float32)
    actions = tf.convert_to_tensor(np.array([exp.action for exp in experiences if exp is not None]), dtype = tf.float32)
    rewards = tf.convert_to_tensor(np.array([exp.reward for exp in experiences if exp is not None]), dtype = tf.float32)
    next_states = tf.convert_to_tensor(np.array([exp.next_state for exp in experiences if exp is not None]), dtype = tf.float32)
    done_values = tf.convert_to_tensor(np.array([exp.done for exp in experiences if exp is not None]).astype(np.uint8), dtype = tf.float32)
    return (states, actions, rewards, next_states, done_values)