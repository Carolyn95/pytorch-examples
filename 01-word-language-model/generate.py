"""
here generates new sentences sampled from the language model

"""


import argparse
import torch
import data

parser = argparse.ArgumentParser(description='Pytorch Wikitext-2 Language Model')

# model parameters
parser.add_argument('--data', type=str, default='./data/', help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt', help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt', help='output file for generated text')
parser.add_argument('--words', type=int, default=1000, help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0, help='higher temporature will increase diversity')
parser.add_argument('--log-interval', type=int, default=100, help='reporting interval')
args = parser.parse_args()


# manually set the random seed for reproducibility
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not arg.cuda:
        print("WARNING: You have a CUDA device, you should probably run with --cuda")

device = torch.device('cuda' if args.cuda else 'cpu')

if arg.temporature < 1e-3:
    parser.error("ERROR: Temporature has to be greater or equal to 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)

model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

