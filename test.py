#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import os
import logging
import time
import numpy as np
from utils.data_preprocessor import data_preprocessor
from utils.minibatch_loader import minibatch_loader
from utils.misc import check_dir, load_word2vec_embeddings
from model import GAReader


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser(
        description='Gated Attention Reader for \
        Text Comprehension Using TensorFlow')
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--ckpt_epoch', type=int, default=10,
                        help='which epoch checkpoint to restore the model from.')
    parser.add_argument('--use_feat', type='bool', default=False,
                        help='whether to use extra features')
    parser.add_argument('--train_emb', type='bool', default=True,
                        help='whether to train embedding')
    parser.add_argument('--init_test', type='bool', default=True,
                        help='whether to perform initial test')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='data directory containing input')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='directory to store checkpointed models')
    parser.add_argument('--embed_file', type=str,
                        default='data/word2vec_glove.txt',
                        help='word embedding initialization file')
    parser.add_argument('--gru_size', type=int, default=256,
                        help='size of word GRU hidden state')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of layers of the model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='mini-batch size')
    parser.add_argument('--n_epoch', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--eval_every', type=int, default=10000,
                        help='evaluation frequency')
    parser.add_argument('--print_every', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--grad_clip', type=float, default=10,
                        help='clip gradients at this value')
    parser.add_argument('--init_learning_rate', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for tensorflow')
    parser.add_argument('--max_example', type=int, default=None,
                        help='maximum number of training examples')
    parser.add_argument('--char_dim', type=int, default=0,
                        help='size of character GRU hidden state')
    parser.add_argument('--gating_fn', type=str, default='tf.multiply',
                        help='gating function')
    parser.add_argument('--drop_out', type=float, default=0.1,
                        help='dropout rate')
    args = parser.parse_args()
    return args

def get_text(idx_to_word, idx, mask):
    idx = idx[:np.sum(mask)]
    words = [idx_to_word[i] for i in idx] 
    return ' '.join(words)


def test(args):
    use_chars = args.char_dim > 0
    # load data
    dp = data_preprocessor()
    data = dp.preprocess(
        question_dir=args.data_dir,
        no_training_set=False,
        max_example=args.max_example,
        use_chars=use_chars)
    #import ipdb; ipdb.set_trace()
    idx_to_word = dict([(v, k) for (k, v) in data.dictionary[0].items()])

    # build minibatch loader
    test_batch_loader = minibatch_loader(
        data.test, args.batch_size, shuffle=False)

    model = GAReader(args.n_layers, data.vocab_size, data.n_chars,
                     args.gru_size, 100, args.train_emb,
                     args.char_dim, args.use_feat, args.gating_fn, save_attn=True)
    with tf.Session() as sess:
        model.restore(sess, args.save_dir, args.ckpt_epoch)
        logging.info('-' * 50)
        logging.info("Start testing...")
        test_writer = tf.summary.FileWriter('logs/test',
                                      sess.graph)
        model.validate(sess, test_batch_loader, write_results=True)


if __name__ == "__main__":
    args = get_args()
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    # check the existence of directories
    args.data_dir = os.path.join(os.getcwd(), args.data_dir)
    check_dir(args.data_dir, exit_function=True)
    args.log_dir = os.path.join(os.getcwd(), args.log_dir)
    args.save_dir = os.path.join(os.getcwd(), args.save_dir)
    check_dir(args.log_dir, args.save_dir, exit_function=False)
    # initialize log file
    log_file = os.path.join(args.log_dir, 'log')
    if args.log_dir is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
    logging.info(args)
    test(args)
