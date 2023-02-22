"""
@author: pvianna
"""

import argparse
import steatosis_dl   #custom function

parser = argparse.ArgumentParser(description='Script for binary steatosis classification')

# Variables
parse_dataset = parser.add_argument('--dataset', type=str, default = '/data/dataset.csv', help='Dataset file, in .csv format')
parse_architecture = parser.add_argument('--architecture', type=str, default = 'VGG16-dropout', help='Default is VGG16 with dropout. Please refer to arch_builder.py')
parse_transfer_learning = parser.add_argument('--transfer_learning', type=str, default='None', help='Transfer learning weights, in .h5 format, or None')
parse_input_size = parser.add_argument('--input_size', type=int, default =128, help='One dimension for square input size. Default is 128.')
parse_images_dir = parser.add_argument('--images_dir', type=str, default='/data/images/' , help='Path to directory with images')
parse_task = parser.add_argument('--task', type=int, default=0, help='0 for S0 vs >=S1, 1 for <=S1 vs >=S2, 2 for <=S2 vs S3, 9 for all tasks in sequence')

# Parse the arguments
args = parser.parse_args()
dataset = args.dataset
architecture = args.architecture
transfer_learning = args.transfer_learning
input_size = tuple([args.input_size, args.input_size])
images_dir = args.images_dir
task = args.task

#Training and validation
results = steatosis_dl.train_and_val(
            dataset, architecture, transfer_learning, task, images_dir, resize_shape=input_size)
