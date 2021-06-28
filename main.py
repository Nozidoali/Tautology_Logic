# Author: Wang Hanyu
# Date  : 2020.06.27

'''
About:
The function of this script is to autometically generate difficult benchmarks
and try to solve it and collect experience using Reinforcement Learning Agent
'''

from Tautology.util import *
import argparse

if __name__ == '__main__':
    # parser of main function
    parser = argparse.ArgumentParser(description='Tautology Generator')
    parser.add_argument('--size', dest='size',
        help='input number of the circuit',
        type=int,
        default = 6)
    parser.add_argument('--output', dest='output',
        help='blif file directory', 
        default='')
    parser.add_argument('--train', dest='train',
        help='blif file RL agent learn to optimize',
        default='')
    parser.add_argument('--print', dest='print',
        help='print the reward of training process',
        default='')
    args = parser.parse_args()
    # parse the command

    if args.output != '':
        '''
        output mode: store the generated circuit to a blif file
        '''
        input_size  = args.size
        output_file = args.output
        write_tautology_to_blif(
            input_size=input_size,
            file=output_file)
    if args.train != '':
        '''
        train mode: use RL agent to learn to optimize the circuit
        '''
        blif_file = args.train
        write_config_to_drills(
            blif_file=blif_file)
    if args.print != '':
        '''
        print mode: plot the training process in the playground
        '''
        episode_max_num = int(args.print)
        show_output_as_csv(
            episode_max_num=episode_max_num)