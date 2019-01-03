import theano
import theano.tensor as T
import numpy
from theano.tensor.nnet.bn import batch_normalization
# start-snippet-1
class HiddenLayer(object):
    def __init__(self, input1, n_in, n_out, W_values=None, b_values=None,
                 activation=T.tanh,batch_norm = True):
        self.input1 = input1
        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        lin_output = T.dot(input1, self.W) + self.b

        # self.gamma = theano.shared(value = numpy.ones((n_out,), dtype=theano.config.floatX), name='gamma')
        # self.beta = theano.shared(value = numpy.zeros((n_out,), dtype=theano.config.floatX), name='beta')
        # bn_output = batch_normalization(inputs = lin_output,
        #                                 gamma = self.gamma, beta = self.beta, mean = lin_output.mean((0,), keepdims=True),
        #                                 std = lin_output.std((0,), keepdims = True),
        #                                 mode='high_mem')
        # self.output1 = (
        #     bn_output if activation is None
        #     else activation(bn_output)
        # )
        if batch_norm:
            self.gamma = theano.shared(value = numpy.ones((n_out,), dtype=theano.config.floatX), name='gamma',borrow=True)
            self.beta = theano.shared(value = numpy.zeros((n_out,), dtype=theano.config.floatX), name='beta',borrow=True)
	    # bn_output = batch_normalization(inputs = self.linear,
	    #     	                    gamma = self.gamma, beta = self.beta, mean = self.linear.mean((0,), keepdims=True),
	    #     	                    std = T.ones_like(self.linear.var((0,), keepdims = True)), mode='high_mem')
#            xmean = lin_output.mean(0, keepdims=True)
#            xstd = T.sqrt(lin_output.std(0, keepdims=True)**2+1e-6)

            bn_output = batch_normalization(inputs = lin_output,
                                            gamma = self.gamma, beta = self.beta, mean = lin_output.mean(0, keepdims=True),
                                            std = T.sqrt(lin_output.std(0, keepdims=True)**2+1e-6),
                                            mode='high_mem')
            self.output1 = T.clip(bn_output,0,40)
            self.params = [self.W, self.b, self.gamma, self.beta]

        else:
            self.output1 = (
                lin_output if activation is None
                else activation(lin_output)
            )
            self.params = [self.W, self.b]
