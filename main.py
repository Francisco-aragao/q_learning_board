from time import time

from q_learning import Game
from utils import Utils

if __name__ == '__main__':

    utils = Utils()

    args = utils.arg_parser()

    # index is based on COLS x ROWS  (not ROWS x COLS)
    width, height = utils.extract_map_dimensions(args.map_file)
    game = Game(width=width, height=height)
    game.map = utils.store_map(map_file=args.map_file, width=width, height=height)

    start = time()

    # the board is inverted, so I need to invert the initial position
    game.q_learning(alg=args.algorithm, initial_x=args.initial_y, initial_y=args.initial_x, step_number=args.step_number)

    policy = game.get_policy()
    for row in policy:
        print(row)

    end = time()

    # printing additional information to measure performance
    if args.measure:
        print('Time:', end - start)