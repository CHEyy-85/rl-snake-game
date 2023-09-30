import numpy as np
import tensorflow as tf
import random
import helper_functions
from collections import deque, namedtuple
from ENV_snake_game_AI import SnakeGameAI, Points, Directions, GRID_SIZE
import model_trainer
from model_trainer import q_net, target_q_net, optimizer, compute_loss, agent_learn
from statistics import mean

MAX_MEMORY_LENGTH = 100000
GAMMA = 0.995
NUM_OF_GAME_FOR_AVG = 10

Experiences = namedtuple('Experiences', ['state', 'action', 'reward', 'next_state', 'done'])

class Agent:

    def __init__(self) -> None:
        self.num_of_games = 0
        self.memory = deque(maxlen = MAX_MEMORY_LENGTH)
        #TODO: model and trainer

    def get_state(self, game_env):
        head = game_env.snake[0]
        up_point = Points(head.x, head.y - GRID_SIZE)
        right_point = Points(head.x + GRID_SIZE, head.y)
        down_point = Points(head.x, head.y + GRID_SIZE)
        left_point = Points(head.x - GRID_SIZE, head.y)
        
        up_dir = game_env.direction == Directions.UP
        right_dir = game_env.direction == Directions.RIGHT
        down_dir = game_env.direction == Directions.DOWN
        left_dir = game_env.direction == Directions.LEFT
        
        state = [
            # danger straight
            up_dir and game_env.is_over(up_point) or 
            right_dir and game_env.is_over(right_point) or
            down_dir and game_env.is_over(down_point) or
            left_dir and game_env.is_over(left_point),

            # danger right
            up_dir and game_env.is_over(right_point) or 
            right_dir and game_env.is_over(down_point) or
            down_dir and game_env.is_over(left_point) or
            left_dir and game_env.is_over(up_point),

            # danger left
            up_dir and game_env.is_over(left_point) or 
            right_dir and game_env.is_over(up_point) or
            down_dir and game_env.is_over(right_point) or
            left_dir and game_env.is_over(down_point),

            # moving direction
            up_dir,
            right_dir,
            down_dir,
            left_dir,

            # relative food position
            game_env.apple.y < game_env.head.y, # u
            game_env.apple.x > game_env.head.x, # r
            game_env.apple.y > game_env.head.y, # d
            game_env.apple.x < game_env.head.x # l
        ]
        return np.array(state, dtype=int)

    def record_expereince(self, state, action, reward, next_state, done):
        self.memory.append(Experiences(state, action, reward, next_state, done))

    def get_action(self, state):
        epsilon = helper_functions.get_epsilon(self.num_of_games)
        action = [0,0,0]
        if random.random() < epsilon:
            print("Random")
            move_idx = random.randint(0,2)
            action[move_idx] = 1
        else:
            print("Model")
            state = tf.convert_to_tensor(np.expand_dims(state, axis = 0), dtype = tf.float32)
            predicted_q_value = q_net(state)
            move_idx = tf.argmax(predicted_q_value, axis = 1).numpy()[0]
            action[move_idx] = 1
        return action
    
    def train_with_experiences(self):
        experiences = helper_functions.get_experiences_for_replay(self.memory)
        tf.function(agent_learn(experiences, GAMMA))


def train_model():

    target_q_net.set_weights(q_net.get_weights())

    scores = []
    mean_scores_last_10 = []
    mean_scores = []
    score_last_10_games = deque(maxlen=10)
    record_score = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, done_value, score = game.take_a_step(action)
        next_state = agent.get_state(game)
        agent.record_expereince(state, action, reward, next_state, done_value)

        update = helper_functions.check_update_condition(agent.num_of_games, agent.memory)
        if update:
                agent.train_with_experiences()

        if done_value:
            game.reset()
            agent.num_of_games += 1
            if score > record_score:
                record_score = score
                q_net.save('AI_Snake_model.h5')
            
            scores.append(score)
            score_last_10_games.append(score)
            mean_score_last_10 = mean(score_last_10_games)
            mean_scores_last_10.append(mean_score_last_10)
            mean_scores.append(mean(scores))
            helper_functions.plot_performace(scores, mean_scores_last_10, mean_scores)

        if record_score > 60:
            print(f"Achieved 60 points in {agent.num_of_games} games.")
            np.save("scores", scores)
            break


if __name__ == '__main__':
    train_model()

            

