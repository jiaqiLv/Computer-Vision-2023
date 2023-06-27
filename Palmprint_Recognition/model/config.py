import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# BATCH_SIZE = 256
BATCH_SIZE = 256
SAVE_FREQ = 1   # save the model every _ epoch
TEST_FREQ = 5
TOTAL_EPOCH = 100

RESUME = ''
SAVE_DIR = './saved_models'

GPU = 0, 1

# Model parameters
image_w = 224
image_h = 224
channel = 3
emb_size = 512

# Training parameters
num_workers = 4  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 120  # print training/validation state every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
num_classes = 640
num_samples = 12800

pickle_file = 'data/faces_ms1m_112x112.pickle'