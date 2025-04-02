import pygame
from enum import Enum
import random
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial',25)

class Direction(Enum):
    RIGHT =1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point','x, y')

# rgb colors
WHITE = (255,255,255)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 10

class SnakeGame:

    def __init__(self,w=640,h=480):
        self.w = w
        self.h =h
        self.display=pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.direction = Direction.RIGHT
        self.reset()
    
    def reset(self):
        self.head = Point(self.w/2,self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE,self.head.y)] # head and tail at initial step
        self._place_food()
        self.score =0

    def _place_food(self):
        # extract random point and save as food point,
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)

        # if food and sanke, placed in same location place again
        if self.food in self.snake:
            self._place_food()
    
    def _update_ui(self):
        self.display.fill(BLACK) # put black screen,
        # rendering every defined point, and updated point while playing.
        # snake
        for pt in self.snake:
            pygame.draw.rect(self.display,color=BLUE1,rect=pygame.Rect(pt.x,pt.y,BLOCK_SIZE,BLOCK_SIZE))
            pygame.draw.rect(self.display,color=BLUE2,rect=pygame.Rect(pt.x+4,pt.y+4,12,12))
        # food
        pygame.draw.rect(self.display,color=RED,rect=pygame.Rect(self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))

        text = font.render("Score: "+str(self.score),True,WHITE)
        self.display.blit(text,[0,0]) # display text on the top corner
        # display on screen
        pygame.display.flip()
    
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction== Direction.RIGHT:
            x+=BLOCK_SIZE
        elif direction== Direction.LEFT:
            x-=BLOCK_SIZE
        elif direction== Direction.DOWN:
            y+=BLOCK_SIZE
        elif direction==Direction.UP:
            y-=BLOCK_SIZE
        self.head = Point(x,y) # new head
    
    def is_collision(self,pt=None):
        if pt is None:
            pt = self.head
        if pt.x>self.w-BLOCK_SIZE or pt.x<0 or pt.y>self.h-BLOCK_SIZE or pt.y<0:
            return True
        if pt in self.snake[1:]:
            return True

    def play_step(self,action):
        # collect user input
        game_over = False
        reward = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Quit button clicked")
                game_over = True
                pygame.quit()
                return self.score,game_over
            
        # calculated direction based on action
        clock_wise = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
        idx= clock_wise.index(self.direction)

        if np.array_equal(action,[1,0,0]):
            self.direction=clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            self.direction=clock_wise[(idx+1)%4]
        elif np.array_equal(action,[0,0,1]):
            self.direction=clock_wise[(idx-1)%4]

        self._move(self.direction)
        self.snake.insert(0,self.head)

        if self.is_collision():
            game_over = True
            reward = -10
            return self.score, game_over
        if self.food==self.head:
            self.score+=1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, self.score, game_over


