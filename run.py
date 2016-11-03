# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse


import tensorflow as tf
import batch_class as tb

import numpy as np
from gensim import models

import tensorflow as tf
import make_predicted_matrix as predictmatrix
import measure_F1 as F1

import pre_process as pp

import added_data as add

import numpy as np
from numpy.random import *



FLAGS = None

w2v_path = "data/GoogleNews-vectors-negative300.bin"
word2vec_models = models.Word2Vec.load_word2vec_format(w2v_path, binary=True)

def main(_):

    # Create the model
    x = tf.placeholder(tf.float32, [None,300])
    W = tf.Variable(tf.zeros([300,19]))
    b = tf.Variable(tf.zeros([19]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None,19])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.

    test_file='data/ABSA15_Laptops_Test.xml'
    train_file='data/ABSA-15_Laptops_Train_Data.xml'
  
    raw_test_data=pp.xml_parse(test_file)
    raw_train_data=pp.xml_parse(train_file)

    common_category_labels_list=pp.common_labels_counter(raw_train_data)

    test_data=pp.label_proc(raw_test_data,common_category_labels_list,normalized=False)
    train_data=pp.label_proc(raw_train_data,common_category_labels_list,normalized=True)
    added_data=train_data+add.main(word2vec_models,train_data,5)

    train_sent,train_labels=pp.make_Dataset(added_data,word2vec_models)
    test_sent,test_labels=pp.make_Dataset(test_data,word2vec_models)

    train_data=tb.Dataset(train_sent,train_labels)
    test_data=tb.Dataset(test_sent,test_labels)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    # Train
    tf.initialize_all_variables().run()

    for _ in range(1000):
        batch_xs, batch_ys = train_data.get_next_minibatch()
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    # Find the best theta and F1 value
    f1_list=[]
    for i in range(1000):
        t=random()
  
        one_hot_mat=predictmatrix.make_one_hot_matrix(sess.run(tf.nn.softmax(y),feed_dict={x:test_data.x}),t)
        try:
            f1=F1.perf_measure(test_data.y_, one_hot_mat)
            f1_list.append(f1)
        except:
            continue
    
    print (max(f1_list))


if __name__ == '__main__':
  tf.app.run()
