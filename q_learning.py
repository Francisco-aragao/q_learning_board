from collections import deque
import heapq
from utils import Utils
import numpy as np
import random

TERRAIN_REWARDS_STANDARD = {
    '.': -0.1,
    ';': -0.3,
    '+': -1.0,
    'x': -10.0,
    'O': 10.0,
    '@': -2e15
}

TERRAIN_REWARDS_POSITIVE = {
    '.': 3.0,
    ';': 1.5,
    '+': 1.0,
    'x': 0.0,
    'O': 10.0,
    '@': -2e15
}

# variables to store the number of expanded nodes and the costs -> not important for the solution, just for analysis
EXPANDED_NODES = 0
costs = []

class Game:
    """
    Class to represent the game map
    """

    def __init__(self, width, height):

        self.row = width
        self.col = height
        self.map = [['' for _ in range(width)] for _ in range(height)]

        # Define possible actions: right, left, down, up
        self.directions = [(0, 1, '>'), (0, -1, '<'), (1, 0, 'v'), (-1, 0, '^')]
        
        # Initialize Q-table with zeros
        self.Q = np.zeros((height, width, len(self.directions)))

    # main function to call each algorithm
    def q_learning(self, alg, initial_x, initial_y, step_number):
        
        if alg == 'stochastic':
            self.q_learning_stochastic(initial_x, initial_y, step_number)
        elif alg == 'positive':
            self.q_learning_positive(initial_x, initial_y, step_number)
        elif alg == 'standard':
            self.q_learning_standard(initial_x, initial_y, step_number)
        else:
            raise Exception(f'Algorithm {alg} not implemented')
        
        return 1

    
    def q_learning_stochastic(self, initial_x, initial_y, step_number):
        """
        Perform Q-Learning for the given number of steps.
        
        :param initial_x: The starting x-coordinate.
        :param initial_y: The starting y-coordinate.
        :param step_number: The number of steps to run the algorithm.
        """
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        epsilon = 0.1  # Exploration rate
        
        for _ in range(step_number):
            x, y = initial_x, initial_y
            while True:
                # Choose an action
                action = self.choose_action((x, y), epsilon)
                dx, dy, _ = action

                if random.uniform(0, 1) < 0.2:
                    # Get perpendicular actions based on the original action
                    if dx == 0:  # Original action is left or right
                        perpendicular_actions = [(-1, 0), (1, 0)]  # Up or down
                    else:  # Original action is up or down
                        perpendicular_actions = [(0, -1), (0, 1)]  # Left or right
                    
                    # Randomly choose one of the perpendicular actions
                    dx, dy = random.choice(perpendicular_actions)
                
                
                # Calculate the new position
                nx, ny = x + dx, y + dy
                
                # Check if the new position is out of bounds or a wall
                if nx < 0 or nx >= self.col or ny < 0 or ny >= self.row or self.map[nx][ny] == '@':
                    nx, ny = x, y  # Stay in the same position
                
                # Get the reward for the new position
                reward = TERRAIN_REWARDS_STANDARD.get(self.map[nx][ny], 0)
                
                # Check if the new state is terminal (fire or goal)
                if self.map[nx][ny] in ['x', 'O']:
                    # Terminal state: update Q-value using the Q-learning formula
                    self.Q[x, y, self.directions.index(action)] += alpha * (
                        reward + gamma * 0 - self.Q[x, y, self.directions.index(action)]
                    )
                    break
                else:
                    # Non-terminal state: update Q-value using the Q-learning formula
                    self.Q[x, y, self.directions.index(action)] += alpha * (
                        reward + gamma * np.max(self.Q[nx, ny]) - self.Q[x, y, self.directions.index(action)]
                    )
                    x, y = nx, ny  # Move to the new position


    def choose_action(self, state, epsilon):
        """
        Choose an action using the ϵ-greedy strategy.
        
        :param state: The current state (x, y).
        :param epsilon: The exploration rate.
        :return: The chosen action (dx, dy, symbol).
        """
        if random.uniform(0, 1) < epsilon:
            # Explore: choose a random action
            return random.choice(self.directions)
        else:
            # Exploit: choose the action with the highest Q-value
            return self.directions[np.argmax(self.Q[state])]
    
    def q_learning_standard(self, initial_x, initial_y, step_number, terrain_rewards=TERRAIN_REWARDS_STANDARD):
        """
        Perform Q-Learning for the given number of steps.
        
        :param initial_x: The starting x-coordinate.
        :param initial_y: The starting y-coordinate.
        :param step_number: The number of steps to run the algorithm.
        """
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        epsilon = 0.1  # Exploration rate
        
        for _ in range(step_number):
            x, y = initial_x, initial_y
            while True:
                # Choose an action
                action = self.choose_action((x, y), epsilon)
                dx, dy, _ = action
                
                # Calculate the new position
                nx, ny = x + dx, y + dy
                
                # Check if the new position is out of bounds or a wall
                if nx < 0 or nx >= self.col or ny < 0 or ny >= self.row or self.map[nx][ny] == '@':
                    nx, ny = x, y  # Stay in the same position
                
                # Get the reward for the new position
                reward = terrain_rewards.get(self.map[nx][ny], 0)
                
                # Check if the new state is terminal (fire or goal)
                if self.map[nx][ny] in ['x', 'O']:
                    # Terminal state: update Q-value using the Q-learning formula
                    self.Q[x, y, self.directions.index(action)] += alpha * (
                        reward + gamma * 0 - self.Q[x, y, self.directions.index(action)]
                    )
                    break
                else:
                    # Non-terminal state: update Q-value using the Q-learning formula
                    self.Q[x, y, self.directions.index(action)] += alpha * (
                        reward + gamma * np.max(self.Q[nx, ny]) - self.Q[x, y, self.directions.index(action)]
                    )
                    x, y = nx, ny  # Move to the new position
    
    def q_learning_positive(self, initial_x, initial_y, step_number):
        self.q_learning_standard(initial_x, initial_y, step_number, terrain_rewards=TERRAIN_REWARDS_POSITIVE)
    
    def get_policy(self):
        """
        Get the optimal policy based on the learned Q-values.
        
        :return: A 2D list representing the policy.
        """
        policy = []
        for i in range(self.col):
            row = []
            for j in range(self.row):
                if self.map[i][j] in ['x', 'O', '@']:
                    # Terminal or impassable states: keep the terrain symbol
                    row.append(self.map[i][j])
                else:
                    # Choose the action with the highest Q-value
                    best_action = self.directions[np.argmax(self.Q[i, j])]
                    row.append(best_action[2])
            policy.append(''.join(row))
        return policy