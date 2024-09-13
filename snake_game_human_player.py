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
SNAKE_SPEED = 10 # achieved by controling frame rate per second (FPS)
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

class SnakeGameHuman:

    def __init__(self, width = 400, height = 400):
        self.width = width
        self.height = height

        # Initialize game interface
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake - Human')
        self.clock = pygame.time.Clock()

        # Initialize game state
        self.direction = Directions.UP # heading up
        self.head = Points(self.width / 2, 3*self.height / 4) # start position
        
        self.snake = [self.head, Points(self.head.x, self.head.y + GRID_SIZE), 
                      Points(self.head.x, self.head.y + (2 * GRID_SIZE))] # initial length = 3
        self.score = 0
        self.apple = None
        self.grid = np.zeros((self.height // GRID_SIZE, self.width // GRID_SIZE), dtype=int)
        self.grid[int(self.head.y // GRID_SIZE), int(self.head.x // GRID_SIZE)] = 1
        self.place_apple()

    def place_apple(self):
        x = random.randint(0, (self.width - GRID_SIZE)//GRID_SIZE) * GRID_SIZE
        y = random.randint(0, (self.height - GRID_SIZE)//GRID_SIZE) * GRID_SIZE
        self.apple = Points(x,y)
        if self.apple in self.snake:
            self.place_apple() # Replace apple when eaten
            

    def is_over(self):
        # hit walls
        if self.head.x > self.width - GRID_SIZE or self.head.x < 0 or self.head.y < 0 or self.head.y > self.height - GRID_SIZE:
            return True
        # eat itself
        if self.head in self.snake[1:]:
            return True
        
        return False

    def head_next(self, direction):
        # from top left
        x = self.head.x
        y = self.head.y
        if direction == Directions.UP:
            y -= GRID_SIZE
        elif direction == Directions.RIGHT:
            x += GRID_SIZE
        elif direction == Directions.DOWN:
            y += GRID_SIZE
        elif direction == Directions.LEFT:
            x -= GRID_SIZE
        
        self.head = Points(x,y)


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
    
    def take_a_step(self):
        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.direction = Directions.UP
                if event.key == pygame.K_RIGHT:
                    self.direction = Directions.RIGHT
                if event.key == pygame.K_DOWN:
                    self.direction = Directions.DOWN
                if event.key == pygame.K_LEFT:
                    self.direction = Directions.LEFT
        # Move
        self.head_next(self.direction) # update the head
        self.snake.insert(0,self.head)

        # check if over
        over = False
        if self.is_over():
            over = True
            return over, self.score
        # place new apple or move 
        self.grid[int(self.head.y // GRID_SIZE), int(self.head.x // GRID_SIZE)] = 1
        if self.head == self.apple:
            self.score += 1
            self.place_apple()
        else:
            self.snake.pop() # pop the last body part since we insert a new head and move for one grid
        
        # update
        self.update_interface()
        self.clock.tick(SNAKE_SPEED)

        return over, self.score
    

if __name__ == '__main__':
    humanGame = SnakeGameHuman()

    while True:
        over, score = humanGame.take_a_step()

        if over == True:
            break

    print(f"Final Score: {score}")
    pygame.quit()