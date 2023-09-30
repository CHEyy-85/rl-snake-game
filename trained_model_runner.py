import numpy as np
import tensorflow as tf
from collections import deque, namedtuple
from ENV_snake_game_AI import SnakeGameAI, Points, Directions, GRID_SIZE
from statistics import mean
import matplotlib.pyplot as plt

trained_model = tf.keras.models.load_model('reached 60.h5', compile = False)


class TrainedModelRunner:

    def __init__(self) -> None:
        pass

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
        return tf.convert_to_tensor(np.expand_dims(np.array(state, dtype=int), axis = 0), dtype = tf.float32)
    
    def get_action(self, state):
        action = [0,0,0]
        q_value = trained_model(state)
        move_idx = tf.argmax(q_value, axis = 1).numpy()[0]
        action[move_idx] = 1
        return action
        
def trained_model_runner():
    runner = TrainedModelRunner()
    game = SnakeGameAI()
    final_scores = []

    for i in range(50):
        while True:
            state = runner.get_state(game)
            action = runner.get_action(state)
            _, done_vals, score = game.take_a_step(action)
            if done_vals:
                game.reset()
                final_scores.append(score)
                break
    plt.plot(final_scores, label = 'scores')
    plt.legend()
    plt.xlabel('# of games')
    plt.ylabel('Performance - pts')
    plt.show()

if __name__ == '__main__':
    trained_model_runner()
