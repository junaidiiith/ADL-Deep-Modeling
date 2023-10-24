import argparse
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    

    args = parser.parse_args()
    logging.info(args)
    return args
