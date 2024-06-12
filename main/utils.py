import os
import sys
from collections import Counter

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

import numpy as np

class Vocabulary(object):
    def __init__(self, pad_word='<pad>', start_word='<sos>', end_word='<eos>', unk_word=None):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_counts = {}

        self.pad_word = pad_word
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        for special_token in [pad_word, start_word, end_word, unk_word]:
            if special_token is not None:
                self.add_word(special_token)

    def __call__(self, word):
        if not word in self.word2idx:
            if self.unk_word is None:
                return None
            else:
                return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word, freq=None):
        if not word in self.word2idx and word is not None:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        if freq is not None:
            self.word_counts[word] = freq
        else:
            self.word_counts[word] = 0

    def get_bias_vector(self):
        words = sorted(self.word2idx.keys())
        bias_vector = np.array([1.0 * self.word_counts[word] for word in words])
        bias_vector /= np.sum(bias_vector)
        bias_vector = np.log(bias_vector)
        bias_vector -= np.max(bias_vector)
        return bias_vector

def build_vocab(texts, frequency=None, filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ', lower=True, split=" ", start_word='<sos>', end_word='<eos>', unk_word=None):
    counter = Counter()
    for i, text in enumerate(texts):
        tokens = word_tokenize(text, filters, lower, split)
        counter.update(tokens)
        if (i + 1) % 5000 == 0:
            print('{} captions tokenized...'.format(i + 1))
    print('Done.')

    if frequency is not None:
        counter = {word: cnt for word, cnt in counter.items() if cnt >= frequency}
    else:
        counter = counter

    vocab = Vocabulary(start_word=start_word, end_word=end_word, unk_word=unk_word)

    words = sorted(counter.keys())
    for word in words:
        vocab.add_word(word, counter[word])
    return vocab

def get_maxlen(texts):
    return max([len(x.split(" ")) for x in texts])

def word_tokenize(text, filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ', lower=True, split=" "):
    if lower:
        text = text.lower()
    text = text.translate(maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [i for i in seq if i]

def text_to_sequence(text, vocab, filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ', lower=True, split=" "):
    tokens = word_tokenize(text, filters, lower, split)
    seq = []
    for token in tokens:
        word_index = vocab(token)
        if word_index is not None:
            seq.extend([word_index])
    return seq

def sequence_to_text(seq, vocab, filter_specials=True, specials=['<pad>', '<sos>', '<eos>']):
    tokens = []
    for idx in seq:
        tokens.append(vocab.idx2word.get(idx))
    if filter_specials:
        tokens = filter_tokens(tokens, specials)
    return ' '.join(tokens)

def texts_to_sequences(texts, vocab, filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ', lower=True, split=" "):
    seqs = []
    for text in texts:
        seqs.append(text_to_sequence(text, vocab, filters, lower, split))
    return seqs

def filter_tokens(tokens, specials=['<pad>', '<sos>', '<eos>']):
    filtered = []
    for token in tokens:
        if token not in specials:
            filtered.append(token)
    return filtered

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
