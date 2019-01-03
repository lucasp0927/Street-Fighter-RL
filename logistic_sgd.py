import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import h5py
import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    def __init__(self, input1, n_in, n_out, W_values=None, b_values=None):
        self.W = theano.shared(
                value=W_values,
                name='W',
                borrow=True
                )
        self.b = theano.shared(
            value=b_values,
            name='b',
            borrow=True
            )
        self.Q = T.dot(input1, self.W) + self.b
        self.action = T.argmax(self.Q, axis=1)
        self.params = [self.W, self.b]
        self.input1 = input1

def load_data(dataset,n_batch,minibatch,slen,train_interval):
    state_buffer = dataset
    state_num = state_buffer.shape[0]
    train_num = n_batch*minibatch
    order = numpy.random.randint(1,state_num,size = train_num)
    #avoid not continuous transition
    for i in enumerate(order):
        if i[1] % train_interval == 0:
            order[i[0]] += numpy.random.randint(1,train_interval)
            order[i[0]] = min(order[i[0]],train_num-1)
    train_set = (numpy.zeros([train_num,slen]),numpy.zeros([train_num,slen]))

    for i in range(train_num):
        train_set[0][i,:] = state_buffer[order[i],:]
        train_set[1][i,:] = state_buffer[order[i]-1,:]

    # def shared_dataset(data_xy, borrow=True):
    #     data_x, data_y = data_xy
    #     shared_x = theano.shared(numpy.asarray(data_x,
    #                                            dtype=theano.config.floatX),
    #                              borrow=borrow)
    #     shared_y = theano.shared(numpy.asarray(data_y,
    #                                            dtype=theano.config.floatX),
    #                              borrow=borrow)
    #     return shared_x, shared_y
    return train_set
