import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model


parser = argparse.ArgumentParser(description='PyTorch wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=20, help='number of layers')
parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='the gradient clipping')
parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=35, help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers, 0 means no dropouts')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=2019, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='', help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2, help='the number of heads in the encoder/decoder of the transformer model')

args = parser.parse_args()

# set random seed for reproducibility
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, you should run it with --cuda")

device = torch.device('cuda' if args.cuda else 'cpu')

# load data
corpus = data.Corpus(args.data)

def batchify(data, bsz):
    # work out how cleanly we can divide the dataset into bsz parts
    nbatch = data.size(0) // bsz # floor operator
    # trim off any extra elements that wouldn't cleanly fit (trim off remainders)
    data = data.narrow(0, 0, nbatch * bsz)
    # evenly divide the data across the bsz batches
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(c
