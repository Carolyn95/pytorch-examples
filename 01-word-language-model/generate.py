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

if model.model_type != 'Transformer':
    hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad(): # no tracking history
        for i in range(args.words):
            if model.model_type != 'Transformer':
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temporature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temporature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 ==19 else ' '))

            if i % args.long_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))

