import random
from game import SnakeGame, Point, Direction
import numpy as np


class GameAgent:

    def __init__(self):
        pass

    def get_state(self,game):
        # 11 values state
        head = game.snake[0]

        # head's next point in space, in all direction 
        point_l = Point(head.x-20,head.y) 
        point_r = Point(head.x+20,head.y)
        point_u = Point(head.x,head.y-20)
        point_d = Point(head.x,head.y+20)

        # check the current dir of snake
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # Danger Left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food Position
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y, # food down
        ]

        return np.array(state,dtype=int)

    def get_action(self,state=None):
        r_idx= np.random.randint(0,2)
        action = np.zeros(3)
        action[r_idx] = 1
        return action

    def play(self,game: SnakeGame):
        # play game on loop
        action = self.get_action()
        score, game_over = game.play_step(action=action)
        return score, game_over

if __name__=='__main__':
    agent = GameAgent()
    game = SnakeGame()

    while True:
        _,game_over = agent.play(game=game)
        if game_over:
            break
        

