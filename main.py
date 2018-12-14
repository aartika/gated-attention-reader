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
import re
import math


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser(
        description='Gated Attention Reader for \
        Text Comprehension Using TensorFlow')
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--resume', type='bool', default=False,
                        help='whether to keep training from previous model')
    parser.add_argument('--ckpt', type=str, default='',
                        help='which checkpoint to resume the training from')
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
    parser.add_argument('--eval_every', type=int, default=1000,
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


def train(args):
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
    train_batch_loader = minibatch_loader(
        data.training, args.batch_size, sample=1.0)
    valid_batch_loader = minibatch_loader(
        data.validation, args.batch_size, shuffle=False)
    test_batch_loader = minibatch_loader(
        data.test, args.batch_size, shuffle=False)
    with tf.device('/device:GPU:0'):
        if not args.resume:
            logging.info("loading word2vec file ...")
            embed_init, embed_dim = \
                load_word2vec_embeddings(data.dictionary[0], args.embed_file)
            logging.info("embedding dim: {}".format(embed_dim))
            logging.info("initialize model ...")
            model = GAReader(args.n_layers, data.vocab_size, data.n_chars,
                             args.gru_size, embed_dim, args.train_emb,
                             args.char_dim, args.use_feat, args.gating_fn, True)
            model.build_graph(args.grad_clip, embed_init)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(tf.global_variables())
        else:
            model = GAReader(args.n_layers, data.vocab_size, data.n_chars,
                             args.gru_size, 100, args.train_emb,
                             args.char_dim, args.use_feat, args.gating_fn, True)

        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
            # training phase
            if not args.resume:
                step = 0
                sess.run(init)
            else:
                step = int(re.search('step_([0-9]+?)-(.*?)', args.ckpt).group(1))
                model.restore(sess, args.save_dir, args.ckpt)
                saver = tf.train.Saver(tf.global_variables())
            if args.init_test:
                logging.info('-' * 50)
                logging.info("Initial test ...")
                best_loss, best_acc = model.validate(sess, valid_batch_loader)
            else:
                best_acc = 0.
 
            logging.info('-' * 50)
            logging.info("Start training ...")
            train_writer = tf.summary.FileWriter('logs/train',
                                          sess.graph)
            while step < args.n_epoch * len(train_batch_loader):
                epoch = int(math.floor(step / len(train_batch_loader)))
                start = time.time()
                it = loss = acc = n_example = 0
                lr = args.init_learning_rate
                if epoch >= 2:
                    lr = args.init_learning_rate / 2**(epoch - 1)
                for dw, dt, qw, qt, a, m_dw, m_qw, tt, \
                        tm, c, m_c, cl, fnames in train_batch_loader:
                    step += 1
                    tf.summary.text('doc', tf.constant(get_text(idx_to_word, dw[0], m_dw[0])))

                    if step % 1000 == 0:
                        logging.info('running train step with summary..')
                        loss_, acc_, summary = model.train(sess, dw, dt, qw, qt, a, m_dw,
                                                  m_qw, tt, tm, c, m_c, cl, fnames,
                                                  args.drop_out, lr, True)
                        train_writer.add_summary(summary, step)
                    else:
                        loss_, acc_ = model.train(sess, dw, dt, qw, qt, a, m_dw,
                                                  m_qw, tt, tm, c, m_c, cl, fnames,
                                                  args.drop_out, lr)
                    loss += loss_
                    acc += acc_
                    it += 1
                    n_example += dw.shape[0]
                    tf.summary.scalar('train_loss', tf.constant(loss_))
                    tf.summary.scalar('train_accuracy', tf.constant(acc_))
                    if step % args.print_every == 0 or \
                            it % len(train_batch_loader) == 0:
                        spend = (time.time() - start) / 60
                        statement = "Epoch: {}, it: {} (max: {}), "\
                            .format(epoch, it, len(train_batch_loader))
                        statement += "loss: {:.3f}, acc: {:.3f}, "\
                            .format(loss / args.print_every,
                                    acc / n_example)
                        statement += "time: {:.1f}(m)"\
                            .format(spend)
                        logging.info(statement)
                        loss = acc = n_example = 0
                        start = time.time()
                    # save model
                    if step % args.eval_every == 0 or \
                            it % len(train_batch_loader) == 0:
                        valid_loss, valid_acc = model.validate(
                            sess, valid_batch_loader)
                        tf.summary.scalar('val_loss', tf.constant(valid_loss))
                        tf.summary.scalar('val_accuracy', tf.constant(valid_acc))
                        if valid_acc >= best_acc:
                            best_loss = valid_loss
                            best_acc = valid_acc
                            logging.info("Best valid acc: {}".format(best_acc))
                            model.save(sess, saver, args.save_dir, step, valid_acc, valid_loss)
                        start = time.time()
                train_writer.close()
            # test model
            logging.info("Final test ...")
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
    train(args)
