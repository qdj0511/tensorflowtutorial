#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 21:30:36 2017

@author: root
"""

from tensorflow.examples.tutorials.mnist import input_data 
mnist  = input_data.read_data_sets("/root/tensorflow/mnistdata" , one_hot=True)

batch_size = 100
xs ,ys = mnist.train.next_batch(batch_size)
print "X shape : " , xs.shape

print "Y shape : " , ys.shape

