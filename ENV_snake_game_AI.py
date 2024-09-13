import random
import pygame
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('Menlo-Regular.ttf', 20)

Points = namedtuple('Points', ['x', 'y'])
Colors = namedtuple('Colors', ['r','g','b'])

# Game constants
SNAKE_SPEED = 100 # achieved by controling frame rate per second (FPS)
GRID_SIZE = 20

# Colors
WHITE = Colors(255,255,255)
RED = Colors(200,0,0)
GOLD = Colors(255,209,0)
BLUE_IN = Colors(0,0,255)
BLUE = Colors(39,116,174)
BLACK = Colors(0,0,0)

# Directions
class Directions(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

# Game Class

class SnakeGameAI:

    def __init__(self, width = 400, height = 400):
        self.width = width
        self.height = height

        # Initialize game interface
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake - AI')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        self.direction = Directions.UP # heading up
        self.head = Points(self.width / 2, 3*self.height / 4) # start position
        self.snake = [self.head, Points(self.head.x, self.head.y + GRID_SIZE), 
                      Points(self.head.x, self.head.y + (2 * GRID_SIZE))] # initial length = 3
        self.score = 0
        self.apple = None
        self.place_apple()
        self.frame = 0


    def place_apple(self):
        x = random.randint(0, (self.width - GRID_SIZE)//GRID_SIZE) * GRID_SIZE
        y = random.randint(0, (self.height - GRID_SIZE)//GRID_SIZE) * GRID_SIZE
        self.apple = Points(x,y)
        if self.apple in self.snake:
            self.place_apple() # Replace apple when eaten
            

    def is_over(self, point=None):
        if point is None:
            point = self.head
        # hit walls
        if point.x > self.width - GRID_SIZE or point.x < 0 or point.y < 0 or point.y > self.height - GRID_SIZE:
            return True
        # eat itself
        if point in self.snake[1:]:
            return True
        
        return False


    def head_next(self, action):
        # action: a binary list [straight, right, left]
        cycle = [Directions.UP, Directions.RIGHT, Directions.DOWN, Directions.LEFT]
        cur_i = cycle.index(self.direction)

        if np.array_equal(action, [1,0,0]):
            new_direction = cycle[cur_i]
        elif np.array_equal(action, [0,1,0]):
            new_direction = cycle[(cur_i + 1) % 4]
        else:
            new_direction = cycle[cur_i - 1]
        
        self.direction = new_direction

        # from top left
        x = self.head.x
        y = self.head.y
        if self.direction == Directions.UP:
            y -= GRID_SIZE
        elif self.direction == Directions.RIGHT:
            x += GRID_SIZE
        elif self.direction == Directions.DOWN:
            y += GRID_SIZE
        elif self.direction == Directions.LEFT:
            x -= GRID_SIZE
        
        self.head = Points(x,y)

    def manhatten_distance(self, point1, point2):
        return abs(point1.x - point2.x) + abs(point1.y - point2.y)
    

    def update_interface(self):
        # draw the screen
        self.display.fill(WHITE[:])

        # draw the snake
        pygame.draw.rect(self.display, GOLD[:], pygame.Rect(self.head.x, self.head.y, GRID_SIZE, GRID_SIZE))
        for points in self.snake[1:]:
            pygame.draw.rect(self.display, BLUE[:], pygame.Rect(points.x, points.y, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(self.display, BLUE_IN[:], pygame.Rect(points.x + 3.5, points.y + 3.5, GRID_SIZE - 7, GRID_SIZE - 7))
        # draw the apple
        pygame.draw.rect(self.display, RED[:], pygame.Rect(self.apple.x, self.apple.y, GRID_SIZE, GRID_SIZE))

        score = font.render("Score: " + str(self.score), True, BLACK[:])
        self.display.blit(score, [0,0])
        pygame.display.flip()
    

    def take_a_step(self, action):
        self.frame += 1
        # Collect user input - Quit or not
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # pre-move: head - apple distance & head - tail distance
        pre_H_A_distance = self.manhatten_distance(self.head, self.apple)
        pre_H_T_distance = self.manhatten_distance(self.head, self.snake[-1])
        # Move
        self.head_next(action)
        self.snake.insert(0,self.head)

        reward = -0.01
        # penalty for turns
        if np.array_equal(action, [0,1,0]) or np.array_equal(action, [0,0,1]):
            reward -= 1
        # penalty for collision
        over = False
        if self.is_over() or self.frame > 100 * len(self.snake):
            over = True
            reward -= 100
            return reward, over, self.score
        
        # place new apple or move 
        if self.head == self.apple:
            self.score += 1
            #reward for eating an apple
            reward += 10
            self.place_apple()
        else:
            self.snake.pop() # pop the last body part since we insert a new head and move for one grid
            # reward & penalty for closer to apple / tail
            after_H_A_distance = self.manhatten_distance(self.head, self.apple)
            after_H_T_distance = self.manhatten_distance(self.head, self.snake[-1])
            if after_H_A_distance < pre_H_A_distance:
                reward += 0.5
            else:
                reward -= 1
            if after_H_T_distance <= pre_H_T_distance:
                reward -= 1

        
        self.update_interface()
        self.clock.tick(SNAKE_SPEED)

        return reward, over, self.score
    