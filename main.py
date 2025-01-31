from q_learning import Game
from utils import Utils
from time import time
import sys

if __name__ == '__main__':

    utils = Utils()

    args = utils.arg_parser()

    # index is based on COLS x ROWS so the width is the height and vice versa
    width, height = utils.extract_map_dimensions(args.map_file)
    game = Game(width=width, height=height)
    game.map = utils.store_map(map_file=args.map_file, width=width, height=height)

    for row in game.map:
        print(row)
    
    print('Initial position:', args.initial_x, args.initial_y)
    print('Number of steps:', args.step_number)

    start = time()
    game.q_learning(args.algorithm, args.initial_x, args.initial_y, args.step_number)

    policy = game.get_policy()

    policy = game.get_policy()
    for row in policy:
        print(row)

    end = time()

    # printing additional information to measure performance
    if args.measure:
        print('Time:', end - start)
        """ print('Expanded nodes:', expanded_nodes) """