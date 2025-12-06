import sys
import os
import argparse

# Ensure python can see the package
sys.path.append(os.getcwd())

from wl_gcl.src.trainers import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset to train on')
    args = parser.parse_args()
    
    # Run the training
    train_model(args.dataset)