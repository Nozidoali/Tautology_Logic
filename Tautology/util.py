import subprocess
import sys
import random
import os
import re
from typing import List

def write_tautology_to_blif(
    input_size: int = 7, 
    file: str = 'output.blif',
    shuffle_product: bool = True,
    shuffle_sum: bool = True,
    seq_sum: List = None
    ) -> List:
    '''
    write_tautology_to_blif:
        1. Generate an tautology logic with <input_size> inputs
        2. Write the result to <file>
        3. Random Shufle the product if <shuffle_product> is True
        4. Random Shufle the sum if <shuffle_sum> is True
    '''
    with open(file,'w') as f:
        f.write('.model test\n')
        f.write('.inputs '+' '.join(['input'+str(index) for index in range(input_size)])+'\n')
        f.write('.outputs out\n')
        # single products in random order
        for combination in range(2**input_size):
            terms = [*range(input_size)]
            if shuffle_product:
                random.shuffle(terms)
            f.write('.names '+' '.join(['input'+str(index) for index in terms])+' node'+str(combination)+'\n')
            f.write(('{0:0'+str(input_size)+'b}').format(combination)+' 1\n')
        # sum all the products in random order
        if seq_sum is None:
            seq = [*range(2**input_size)]
            if shuffle_sum:
                random.shuffle(seq)
        else:
            seq = seq_sum
        f.write('.names '+' '.join(['node'+str(index) for index in seq])+' out\n')
        f.write('0'*(2**input_size)+' 0\n')
        f.write('.end\n')
        return seq

def write_config_to_drills(
    blif_file: str = 'output.blif',
    run_mode: str = 'fpga',
    episode_num: int = 50,
    iteration_num: int = 100):
    model_dir = os.path.abspath('./model/model')
    if os.path.isdir('model') == False:
        os.mkdir('model')
    with open('params.yml', 'w') as file:
        file.write(
            '''
abc_binary: yosys-abc
yosys_binary: yosys
design_file: {}
mapping:
    clock_period: 150   # in pico seconds
    library_file: tech.lib
fpga_mapping:
    levels: 100
    lut_inputs: 6
optimizations:
    - rewrite
    - rewrite -z
    - refactor
    - refactor -z
    - resub
    - resub -z
    - balance
playground_dir: playground
episodes: {}
iterations: {}
model_dir: {}
'''.format(
    blif_file,
    episode_num,
    iteration_num,
    model_dir))

    command = 'python DRiLLS/drills.py train {}'.format(run_mode)
    proc = subprocess.Popen(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    while True:
        line = proc.stdout.readline().decode('utf-8')
        if not line:
            break
        print(line, end='')

def show_output_as_csv(
    episode_max_num: int = 100,
    playground_path: str = 'playground'):
    '''
    show_output_as_csv: retrieve the result in the playground directory, and 
    return the csv of the following columns:
    <iter_num>, <circuit area>, <circuit level>
    '''
    for episode_num in range(episode_max_num):
        episode_dir = os.path.join(
            os.path.abspath(playground_path),
            str(episode_num+1),
            'log.csv')
        with open(episode_dir) as f:
            result_line = f.readlines()[-1]
            result_area = result_line.split(',')[2].strip()
            result_level = result_line.split(',')[3].strip()
            print(','.join([
                str(episode_num),
                str(result_area),
                str(result_level)
            ]))
        