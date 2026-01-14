# -*- coding: utf-8 -*-
"""
Operator-level policy learning for fuzzing
------------------------------------------
Learn to select mutation operators using a neural network.

Workflow:
1. Parse fuzzing log file:
   (context_vector, operator_id, reward)
2. Train a policy network with operator embeddings
3. Use the network to guide next operator selection
"""

import os
import sys
import time
import random
import argparse
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =====================
# Global configuration
# =====================





# =====================
# Main
# =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="fuzzing log file")
    parser.add_argument("--out-dir", default="./out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()