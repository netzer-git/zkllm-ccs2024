import os, sys
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='LLaMa-2 Skip Connection')
parser.add_argument('--block_input_file', required = True, type=str, help='Input of the block.')
parser.add_argument('--block_output_file', required = True, type=str, help='Output of the block.')
parser.add_argument('--output_file', required = True, type=str, help='Output of the skip connection.')
parser.add_argument('--generator_file', required = True, type=str,
                    help='Pedersen generator file (pp-*.bin) used to commit x, y, z for the soundness proof.')

from transformers import AutoTokenizer, AutoModelForCausalLM
import fileio_utils


if __name__ == '__main__':
    compilation_error = os.system('make skip-connection')
    if compilation_error:
        print("Error compiling skip-connection")
        exit(1)
    args = parser.parse_args()
    
    if not os.path.isfile(args.block_input_file) or not os.path.isfile(args.block_output_file):
        print("Input or output file does not exist.")
        exit(1)
    if not os.path.isfile(args.generator_file):
        print("Generator file does not exist: " + args.generator_file)
        exit(1)

    rc = os.system('./skip-connection {} {} {} {}'.format(
        args.block_input_file, args.block_output_file, args.output_file, args.generator_file))
    if rc != 0:
        exit(rc >> 8 if rc >> 8 else 1)