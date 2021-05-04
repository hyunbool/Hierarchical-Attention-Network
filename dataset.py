import torch
from torch import nn
import numpy as np
from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from tqdm import tqdm
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
import json
import gensim
import logging


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

classes = ['positive', 'negative']
label_map = {k: v for v, k in enumerate(classes)}
rev_label_map = {v: k for k, v in label_map.items()}

# Tokenizers
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()

from torch.utils.data import Dataset
import torch
import os


class HANDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        split = split.upper()
        assert split in {'TRAIN', 'VALID', 'TEST'}
        self.split = split

        # Load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

    def __getitem__(self, i):
        return torch.LongTensor(self.data['docs'][i]), \
               torch.LongTensor([self.data['segments_per_document'][i]]), \
               torch.LongTensor([self.data['sentences_per_segment'][i]]), \
               torch.LongTensor(self.data['words_per_sentence'][i]), \
               torch.LongTensor([self.data['labels'][i]])

    def __len__(self):
        return len(self.data['labels'])



def preprocess(text):
    if isinstance(text, float):
        return ''

    return text.lower().replace('<br />', '\n').replace('<br>', '\n').replace('\\n', '\n').replace('&#xd;', '\n')


def read_csv(csv_folder, split, segment_limit, sentence_limit, word_limit):
    """
    Read CSVs containing raw training data, clean documents and labels, and do a word-count.

    :param csv_folder: folder containing the CSV
    :param split: train or test CSV?
    :param sentence_limit: truncate long documents to these many sentences
    :param word_limit: truncate long sentences to these many words
    :return: documents, labels, a word-count
    """
    assert split in {'train', 'test'}

    docs = []
    labels = []
    word_counter = Counter()
    data = pd.read_csv(os.path.join(csv_folder, "short_concat_" + split + '.csv'), header=None)
    for i in tqdm(range(data.shape[0])):
        # 전체 문서
        row = list(data.loc[i, :])
        segments = list()
        text = row[0]

        # 각 문단을 문장 단위로 잘라 저장
        for paragraph in preprocess(text).splitlines():
            segments.append([s for s in sent_tokenizer.tokenize(paragraph)])

        # 단어 단위로 토크나이징
        sentences = list()

        for paragraph in segments[:segment_limit]:
            words = list()
            for s in paragraph[:sentence_limit]:
                w = word_tokenizer.tokenize(s)[:word_limit]
                # If sentence is empty (due to removing punctuation, digits, etc.)
                if len(w) == 0:
                    continue
                words.append(w)
                word_counter.update(w)
            sentences.append(words)
        # If all sentences were empty
        if len(words) == 0:
            continue

        labels.append(int(row[1]))  # since labels are 1-indexed in the CSV
        docs.append(sentences)

    return docs, labels, word_counter


def create_input_files(csv_folder, output_folder, segment_limit, sentence_limit, word_limit, min_word_count=5,
                       save_word2vec_data=True):
    """
    training
    """
    # Read training data
    train_docs, train_labels, word_counter = read_csv(csv_folder, 'train', segment_limit, sentence_limit, word_limit)

    # word2vec 위한 데이터 저장
    if save_word2vec_data:
        torch.save(train_docs, os.path.join(output_folder, 'word2vec_data.pth.tar'))


    # Create word map
    word_map = dict()
    word_map['<pad>'] = 0
    for word, count in word_counter.items():
        if count >= min_word_count:
            word_map[word] = len(word_map)

    word_map['<unk>'] = len(word_map) # 가장 마지막 인덱스가 <unk>

    with open(os.path.join(output_folder, 'word_map.json'), 'w') as j:
        json.dump(word_map, j)


    # 검증 데이터 나누기
    train_docs, valid_docs, train_labels, valid_labels = train_test_split(train_docs, train_labels, test_size=0.3)

    segments_per_train_document = list(map(lambda doc: len(doc), train_docs))
    sentences_per_train_segment = list(map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (segment_limit - len(doc)), train_docs))
    words_per_train_sentences = list(map(lambda doc: list(map(lambda seg: list(map(lambda sent: len(sent), seg)) + [0] * (sentence_limit - len(seg)) , doc)), train_docs))


    segments_per_valid_document = list(map(lambda doc: len(doc), valid_docs))
    sentences_per_valid_segment = list(map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (segment_limit - len(doc)), valid_docs))
    words_per_valid_sentences = list(map(lambda doc: list(map(lambda seg: list(map(lambda sent: len(sent), seg)) + [0] * (sentence_limit - len(seg)) , doc)), valid_docs))


    # 단어가 word_map에 있으면 해당 key 리턴하고 없으면 <unk>의 key 리턴
    encoded_train_docs = list(map(lambda doc: list(map(lambda seg: list(map(lambda sent: list(map(lambda word: word_map.get(word, word_map['<unk>']),sent)) + [0] * (word_limit - len(sent)), seg)) + [[0] * word_limit] * (sentence_limit - len(seg)), doc)), train_docs))
    encoded_val_docs = list(map(lambda doc: list(map(lambda seg: list(map(lambda sent: list(map(lambda word: word_map.get(word, word_map['<unk>']), sent)) + [0] * (word_limit - len(sent)), seg)) + [[0] * word_limit] * (sentence_limit - len(seg)), doc)), valid_docs))


    assert len(encoded_train_docs) == len(train_labels) == len(segments_per_train_document) == len(sentences_per_train_segment) == len(words_per_train_sentences)
    assert len(encoded_val_docs) == len(valid_labels) == len(segments_per_valid_document) == len(sentences_per_valid_segment) == len(words_per_valid_sentences)


    # Because of the large data, saving as a JSON can be very slow
    torch.save({'docs': encoded_train_docs,
                'labels': train_labels,
                'segments_per_document': segments_per_train_document,
                'sentences_per_segment': sentences_per_train_segment,
                'words_per_sentence': words_per_train_sentences},
               os.path.join(output_folder, 'TRAIN_data.pth.tar'))
    torch.save({'docs': encoded_val_docs,
                'labels': valid_labels,
                'segments_per_document': segments_per_valid_document,
                'sentences_per_segment': sentences_per_valid_segment,
                'words_per_sentence': words_per_valid_sentences},
               os.path.join(output_folder, 'VALID_data.pth.tar'))

    # Free some memory
    del train_docs, encoded_train_docs, train_labels, segments_per_train_document, sentences_per_train_segment, words_per_train_sentences
    del valid_docs, encoded_val_docs, valid_labels, segments_per_valid_document, sentences_per_valid_segment, words_per_valid_sentences



    """
    testing
    """
    # Read test data
    print('Reading and preprocessing test data...\n')
    test_docs, test_labels, _ = read_csv(csv_folder, 'test', segment_limit, sentence_limit, word_limit)


    segments_per_test_document = list(map(lambda doc: len(doc), test_docs))
    sentences_per_test_segment = list(map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (segment_limit - len(doc)), test_docs))
    words_per_test_sentences = list(map(lambda doc: list(map(lambda seg: list(map(lambda sent: len(sent), seg)) + [0] * (sentence_limit - len(seg)) , doc)), test_docs))
    encoded_test_docs = list(map(lambda doc: list(map(lambda seg: list(map(lambda sent: list(map(lambda word: word_map.get(word, word_map['<unk>']), sent)) + [0] * (word_limit - len(sent)), seg)) + [[0] * word_limit] * (sentence_limit - len(seg)), doc)), test_docs))


    assert len(encoded_test_docs) == len(test_labels) == len(segments_per_test_document) == len(sentences_per_test_segment) == len(words_per_test_sentences)

    torch.save({'docs': encoded_test_docs,
                'labels': test_labels,
                'segments_per_document': segments_per_test_document,
                'sentences_per_segment': sentences_per_test_segment,
                'words_per_sentence': words_per_test_sentences},
               os.path.join(output_folder, 'TEST_data.pth.tar'))

    print('All done!\n')


def train_word2vec_model(data_folder, algorithm='skipgram'):
    """
    Train a word2vec model for word embeddings.

    See the paper by Mikolov et. al. for details - https://arxiv.org/pdf/1310.4546.pdf

    :param data_folder: folder with the word2vec training data
    :param algorithm: use the Skip-gram or Continous Bag Of Words (CBOW) algorithm?
    """
    assert algorithm in ['skipgram', 'cbow']

    sg = 1 if algorithm is 'skipgram' else 0

    # Read data

    segments = torch.load(os.path.join(data_folder, 'word2vec_data.pth.tar'))


    # 모든 문서의 sentence들을 통채로 합해주기
    sentences = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(segments))))
    # Activate logging for verbose training
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Initialize and train the model (this will take some time)
    model = gensim.models.word2vec.Word2Vec(sentences=sentences, size=200, workers=8, window=10, min_count=5,
                                            sg=sg)

    # Normalize vectors and save model
    model.init_sims(True)
    model.wv.save(os.path.join(data_folder, 'word2vec_model'))


def init_embedding(input_embedding):
    """
    Initialize embedding tensor with values from the uniform distribution.

    :param input_embedding: embedding tensor
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def load_word2vec_embeddings(word2vec_file, word_map):
    """
    Load pre-trained embeddings for words in the word map.

    :param word2vec_file: location of the trained word2vec model
    :param word_map: word map
    :return: embeddings for words in the word map, embedding size
    """
    # Load word2vec model into memory
    w2v = gensim.models.KeyedVectors.load(word2vec_file, mmap='r')



    # Create tensor to hold embeddings for words that are in-corpus
    # word_map 내 단어들에 대한 임베딩 벡터 만들기
    embeddings = torch.FloatTensor(len(word_map), w2v.vector_size)
    init_embedding(embeddings)

    # Read embedding file

    for word in word_map:
        if word in w2v.vocab:
            embeddings[word_map[word]] = torch.FloatTensor(w2v[word])


    return embeddings, w2v.vector_size


create_input_files(csv_folder='./data',
                   output_folder='./data',
                   segment_limit=3,
                   sentence_limit=20,
                   word_limit=20,
                   min_word_count=5)