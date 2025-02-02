import numpy as np
import random

# terrains costs for q-learning standard algorithm
TERRAIN_REWARDS_STANDARD = {
    '.': -0.1,
    ';': -0.3,
    '+': -1.0,
    'x': -10.0,
    'O': 10.0,
    '@': -2e15
}

# terrains costs for q-learning positive algorithm
TERRAIN_REWARDS_POSITIVE = {
    '.': 3.0,
    ';': 1.5,
    '+': 1.0,
    'x': 0.0,
    'O': 10.0,
    '@': -2e15
}

# parameters for q-learning search
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate


class Game:
    """
    Class to represent the game map and perform Q-Learning.
    """

    def __init__(self, width, height):

        self.row = width
        self.col = height
        self.map = [['' for _ in range(width)] for _ in range(height)]

        # possible actions: right (>), left (<), down (v), up (^)
        self.directions = [(0, 1, '>'), (0, -1, '<'), (1, 0, 'v'), (-1, 0, '^')]
        
        # initialize Q-table with zeros
        # Q-table dimensions: (height, width, number of actions)
        # -> it is stored the reward for each action (right, left, down, up) in each position of the map
        self.Q = np.zeros((height, width, len(self.directions)))

    # main function to call each algorithm
    def q_learning(self, alg, initial_x, initial_y, step_number):

        if alg == 'stochastic':
            self.q_learning_stochastic(initial_x, initial_y, step_number, terrain_rewards=TERRAIN_REWARDS_STANDARD, algorithm='stochastic')
        elif alg == 'positive':
            self.q_learning_positive(initial_x, initial_y, step_number, terrain_rewards=TERRAIN_REWARDS_POSITIVE, algorithm='positive')
        elif alg == 'standard':
            self.q_learning_standard(initial_x, initial_y, step_number, terrain_rewards=TERRAIN_REWARDS_STANDARD, algorithm='standard')
        else:
            raise Exception(f'Algorithm {alg} not implemented')        

    


    # chose the action to be performed by the agent -> epsilon greedy uses probability to choose between exploration and exploitation
    def choose_action_e_greedy(self, state):
        
        if random.uniform(0, 1) < EPSILON:
            # Explore: choose a random action
            return random.choice(self.directions)
        else:
            # Exploit: choose the action with the highest Q-value (best action)
            return self.directions[np.argmax(self.Q[state])]
    
    # get the next position to be visited. If the algorithm is stochastic, it will have a 20% chance of moving in a perpendicular direction, else it will move in the direction chosen by the agent by the choose_action_e_greedy function
    def get_next_position(self, x, y, dx, dy, algorithm):

        if algorithm == 'stochastic':
            # decide if the agent will move one of the perpendicular directions
            if random.uniform(0, 1) < 0.2:
                if dx == 0:  # original action is left or right
                    perpendicular_actions = [(-1, 0), (1, 0)]  # up or down
                else:  # original action is up or down
                    perpendicular_actions = [(0, -1), (0, 1)]  # left or right
                
                # two options of movements -> random.choice to choose one of them
                dx, dy = random.choice(perpendicular_actions)
            
            
            nx, ny = x + dx, y + dy

            return nx, ny
        
        # standard or positive algorithms -> just return new position
        return x + dx, y + dy

    
    def q_learning_standard(self, initial_x, initial_y, step_number, terrain_rewards, algorithm):
        """
        Perform Q-Learning for the given number of steps.
        
        - initial_x and initil_y = initial position
        - step_number = number of iterations to prform
        - terrain_rewards = dict with the rewards for each terrain
        - algorithm = algorithm to be used (standard, stochastic, positive)
        
        """
        
        for _ in range(step_number):
            x, y = initial_x, initial_y

            while True: # loop until the agent reaches a terminal state

                # Choose an action
                action = self.choose_action_e_greedy((x, y))
                dx, dy, _ = action

                nx, ny = self.get_next_position(x, y, dx, dy, algorithm) # get the next position based on the algorithm
                
                # check if the new position is out of bounds or a wall
                out_of_bounds = nx < 0 or nx >= self.col or ny < 0 or ny >= self.row
                wall = self.map[nx][ny] == '@' if not out_of_bounds else True # if is out of bounds, i will consider it as a wall
                
                if out_of_bounds or wall:
                    nx, ny = x, y  # stay in the same position
                
                # Get the reward for the new position
                reward = terrain_rewards[self.map[nx][ny]]
                
                # Check if the new state is terminal (fire or goal)
                is_in_terminal_state = self.map[nx][ny] in ['x', 'O']
                if is_in_terminal_state:
                    # update Q-value with the future reward -> 0 (terminal state)
                    future_reward_in_terminal_state= 0
                    self.Q[x, y, self.directions.index(action)] += ALPHA * (
                        reward + GAMMA * future_reward_in_terminal_state - self.Q[x, y, self.directions.index(action)]
                    )
                    break

                # non terminal state: update Q-value using the Q-learning formula
                self.Q[x, y, self.directions.index(action)] += ALPHA * (
                    reward + GAMMA * np.max(self.Q[nx, ny]) - self.Q[x, y, self.directions.index(action)]
                )
                x, y = nx, ny  # Move to the new position
    
    def q_learning_stochastic(self, initial_x, initial_y, step_number, terrain_rewards, algorithm):
        """
        Stochastic version of Q-Learning. It works similarly to the standard version, but with a 20% chance of moving in a perpendicular direction.

        The code just call the standard version with the algorithm parameter set to 'stochastic', so the get_next_position function will be called with the right parameters.
        """
        self.q_learning_standard(initial_x, initial_y, step_number, terrain_rewards, algorithm)

    def q_learning_positive(self, initial_x, initial_y, step_number, terrain_rewards, algorithm):
        """
        Positive version of Q-Learning. It works similarly to the standard version, but with positive rewards for each terrain.

        The code just call the standard version with the algorithm parameter set to 'positive' and the correct terrain_rewards dict, so the get_next_position function will be called with the right parameters.
        """
        self.q_learning_standard(initial_x, initial_y, step_number, terrain_rewards, algorithm)
    
    # function to return the best policy found by the Q-learning algorithm
    def get_policy(self):

        policy = []
        for i in range(self.col):
            row = []
            for j in range(self.row):
                if self.map[i][j] in ['x', 'O', '@']: # the policy stores terminal simbols when facing Fire (X), Objective (O) or Wall (@)
                    row.append(self.map[i][j])
                else:
                    # choose the action with the highest Q-value
                    # the Q-table stores on the position (i, j) the reward for each action (right, left, down, up), so I am choosing the best movement here 
                    _, _, best_movement = self.directions[np.argmax(self.Q[i, j])]
                    row.append(best_movement)
            policy.append(''.join(row))

        return policy