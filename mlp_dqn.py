import numpy
import theano
import theano.tensor as T
from hiddenlayer import HiddenLayer
from logistic_sgd import LogisticRegression

class MLP_DQN(object):
    def __init__(self, rng, input1,input2, n_in, n_hidden1, n_hidden2,n_out,model=None,gamma=0.99):
        self.rng = rng
        self.n_in = n_in
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_out = n_out
        if model is None:
            model = self.init_model()

        self.hiddenLayer1 = HiddenLayer(
            input1=input1,
            n_in=n_in,
            n_out=n_hidden1,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[0],
            b_values = model[2]
        )

        self.hiddenLayer2 = HiddenLayer(
            input1=self.hiddenLayer1.output1,
            n_in=n_hidden1,
            n_out=n_hidden2,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[1],
            b_values = model[3]
        )
        self.logRegressionLayer = LogisticRegression(
            input1=self.hiddenLayer2.output1,
            n_in=n_hidden2,
            n_out=n_out,
            W_values = model[4],
            b_values = model[5]
        )

        self.hiddenLayer1_t = HiddenLayer(
            input1=input2,
            n_in=n_in,
            n_out=n_hidden1,
            activation=T.nnet.relu,
            W_values = model[0],
            b_values = model[2]
        )

        self.hiddenLayer2_t = HiddenLayer(
            input1=self.hiddenLayer1_t.output1,
            n_in=n_hidden1,
            n_out=n_hidden2,
            activation=T.nnet.relu,
            W_values = model[1],
            b_values = model[3]
        )
        self.logRegressionLayer_t = LogisticRegression(
            input1=self.hiddenLayer2_t.output1,
            n_in=n_hidden2,
            n_out=n_out,
            W_values = model[4],
            b_values = model[5]
        )

        self.L1 = (
            abs(self.hiddenLayer1.W).sum()
            + abs(self.hiddenLayer2.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer1.W ** 2).sum()
            + (self.hiddenLayer2.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.params = self.hiddenLayer1.params + self.hiddenLayer2.params + self.logRegressionLayer.params
#        self.params = self.hiddenLayer1.params+ self.logRegressionLayer.params
        # end-snippet-3
        # keep track of model input
        self.Qs = self.logRegressionLayer.Q
        self.Qsp = self.logRegressionLayer_t.Q
        self.input1 = input1
        self.input2 = input2
        self.aidx = T.cast(input1[:,5]+1, 'int64')
#        self.cost = T.mean(T.max(self.logRegressionLayer.Qsp,axis=1))
#        self.cost = T.mean(T.max(self.logRegressionLayer.Qsp,axis=1)-T.max(self.logRegressionLayer.Qs,axis=1))
#        self.cost = self.Qs[0,0]-self.Qsp[0,0]

        self.target = input1[:,0]+gamma*T.max(self.Qsp,axis=1)
        self.Qcost = T.mean(0.5*(self.target-self.Qs[T.arange(self.aidx.shape[0]),self.aidx])**2)
#        self.Qcost = T.mean(0.5*(self.target-self.Qs[T.arange(self.aidx.shape[0]),self.aidx])**2)
        self.cost = self.Qcost#+0.0001*self.L2_sqr
#        self.errors = T.sqrt(T.mean(((input1[:,0]+0.97*T.max(self.logRegressionLayer.Qsp,axis=1)-T.max(self.logRegressionLayer.Qs,axis=1))/(input1[:,0]+0.95*T.max(self.logRegressionLayer.Qsp,axis=1)))**2))
        #######parameters
        self.Wh1 = self.hiddenLayer1.W
        self.Wh2 = self.hiddenLayer2.W
        self.bh1 = self.hiddenLayer1.b
        self.bh2 = self.hiddenLayer2.b
        self.OW  = self.logRegressionLayer.W
        self.Ob  = self.logRegressionLayer.b
        self.Wh1t = self.hiddenLayer1_t.W
        self.Wh2t = self.hiddenLayer2_t.W
        self.bh1t = self.hiddenLayer1_t.b
        self.bh2t = self.hiddenLayer2_t.b
        self.OWt  = self.logRegressionLayer_t.W
        self.Obt  = self.logRegressionLayer_t.b
        self.Wh1ddqn = self.hiddenLayer1_ddqn.W
        self.Wh2ddqn = self.hiddenLayer2_ddqn.W
        self.bh1ddqn = self.hiddenLayer1_ddqn.b
        self.bh2ddqn = self.hiddenLayer2_ddqn.b
        self.OWddqn  = self.logRegressionLayer_ddqn.W
        self.Obddqn  = self.logRegressionLayer_ddqn.b
    def init_weight(self,shape):
        return self.rng.normal(loc=0.0,scale=0.1,size=shape)
    def init_bias(self,shape):
        return numpy.full(shape,0.1)
    def init_model(self):
        a0 = self.init_weight((self.n_in,self.n_hidden1))
        a1 = self.init_weight((self.n_hidden1,self.n_hidden2))
        a2 = self.init_bias((self.n_hidden1,))
        a3 = self.init_bias((self.n_hidden2,))
        a4 = self.init_weight((self.n_hidden2,self.n_out))
        a5 = self.init_bias((self.n_out,))
        return (a0,a1,a2,a3,a4,a5)
