import argparse
import gym
import numpy as np
from itertools import count
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description="PyTorch Reinforcement example")
parser.add_argument(

