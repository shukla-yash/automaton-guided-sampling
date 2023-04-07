import os
import yaml

# from train import train
from train_single import train


def read_one_block_of_yaml_data(filename):
    with open(f'/home/yshukl01/locally-greedy/automaton-guided-sampling/{filename}.yaml','r') as f:
        output = yaml.safe_load(f)
    return output 
    
params_output = read_one_block_of_yaml_data('params')

train(params_output)
