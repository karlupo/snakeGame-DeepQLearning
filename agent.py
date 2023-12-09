import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

RANDOM_MOVE_LIMIT = 80 #RANDOM_MOVE_LIMIT - n_games == possibility of a random move; if n_games >= RANDOM_MOVE_LIMIT no random move can happen anymore

class Agent:

    def __int__(self):
        self.n_games = 0
        self.epsilon = 0 #control randomness on prediction
        self.gamma = 0 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #pops left elements if over maxlen thats why using a deque
        self.model = None #TODO
        self.trainer = None #TODO

    def get_state(self, game):
        head = game.snake[0]
        point_left = Point(head.x - 20, head.y) #head pos - blocksize
        point_right = Point(head.x + 20, head.y)
        point_up = Point(head.x, head.y - 20)
        point_down = Point(head.x, head.y + 20)

        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_right and game.is_collision(point_right)) or
            (dir_left and game.is_collision(point_left)) or
            (dir_up and game.is_collision(point_up)) or
            (dir_down and game.is_collision(point_down)),

            # Danger right
            (dir_right and game.is_collision(point_down)) or
            (dir_left and game.is_collision(point_up)) or
            (dir_up and game.is_collision(point_right)) or
            (dir_down and game.is_collision(point_left)),

            # Danger left
            (dir_right and game.is_collision(point_up)) or
            (dir_left and game.is_collision(point_down)) or
            (dir_up and game.is_collision(point_left)) or
            (dir_down and game.is_collision(point_right)),

            dir_left, dir_right, dir_up, dir_down,

            game.food.x < game.head.x, #is food left
            game.food.x > game.head.x, #is food right
            game.food.y < game.head.y, #is food up
            game.food.y > game.head.y # is food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) >= BATCH_SIZE:
            memory_sample = random.sample(self.memory, BATCH_SIZE) #randomly returns 1000 (BATCH_SIZE) elements from our memory
        else:
            memory_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*memory_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        #random moves: the better the model gets we want to do less random moves
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float) #converts state in a tensor (tensor = similar to multidimensional array)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item() #gets the index num of the maximum and converts it to an native int (e.g [5.2, 3.9, 2.4] --> 0)
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
    highscore = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #get cur state of game
        state_cur = agent.get_state(game)

        #get move prediction
        final_move = agent.get_action(state_cur)

        #perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short mem
        agent.train_short_memory(state_cur, final_move, reward, state_new, game_over)

        #remember
        agent.remember(state_cur, final_move, reward, state_new, game_over)

        if game_over:
            #train long memory (replay memory), plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > highscore:
                highscore = score
                #agent.model.save

            print('Game', agent.n_games, 'Score', score, 'HighScore', highscore)


if __name__ == '__main__':
    train()
