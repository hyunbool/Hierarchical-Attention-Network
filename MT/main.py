from __future__ import unicode_literals, print_function, division
import random
from model import *
from utils import *
import torch




input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

use_cuda = torch.cuda.is_available()

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)


if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder1, attn_decoder1, 75000, input_lang, output_lang, pairs, print_every=5000)