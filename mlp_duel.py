import numpy
import theano
import theano.tensor as T
from hiddenlayer import HiddenLayer
from logistic_sgd import LogisticRegression

class MLP_DUEL(object):
    def __init__(self, rng, input1,input2, n_in, n_hidden1, n_hidden2,n_out,model=None,gamma=0.99):
        self.rng = rng
        self.n_in = n_in
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_out = n_out
        if model is None:
            model = self.init_model()

        self.VhiddenLayer1 = HiddenLayer(
            input1=input1,
            n_in=n_in,
            n_out=n_hidden1,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[0],
            b_values = model[2]
        )
        self.VhiddenLayer2 = HiddenLayer(
            input1=self.VhiddenLayer1.output1,
            n_in=n_hidden1,
            n_out=n_hidden2,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[1],
            b_values = model[3]
        )
        self.VlogRegressionLayer = LogisticRegression(
            input1=self.VhiddenLayer2.output1,
            n_in=n_hidden2,
            n_out=1,
            W_values = model[4],
            b_values = model[5]
        )
        self.AhiddenLayer1 = HiddenLayer(
            input1=input1,
            n_in=n_in,
            n_out=n_hidden1,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[6],
            b_values = model[8]
        )
        self.AhiddenLayer2 = HiddenLayer(
            input1=self.AhiddenLayer1.output1,
            n_in=n_hidden1,
            n_out=n_hidden2,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[7],
            b_values = model[9]
        )
        self.AlogRegressionLayer = LogisticRegression(
            input1=self.AhiddenLayer2.output1,
            n_in=n_hidden2,
            n_out=n_out,
            W_values = model[10],
            b_values = model[11]
        )
        #######ddqn##########
        self.VhiddenLayer1_ddqn = HiddenLayer(
            input1=input2,
            n_in=n_in,
            n_out=n_hidden1,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[0],
            b_values = model[2]
        )
        self.VhiddenLayer2_ddqn = HiddenLayer(
            input1=self.VhiddenLayer1_ddqn.output1,
            n_in=n_hidden1,
            n_out=n_hidden2,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[1],
            b_values = model[3]
        )
        self.VlogRegressionLayer_ddqn = LogisticRegression(
            input1=self.VhiddenLayer2_ddqn.output1,
            n_in=n_hidden2,
            n_out=1,
            W_values = model[4],
            b_values = model[5]
        )
        self.AhiddenLayer1_ddqn = HiddenLayer(
            input1=input2,
            n_in=n_in,
            n_out=n_hidden1,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[6],
            b_values = model[8]
        )
        self.AhiddenLayer2_ddqn = HiddenLayer(
            input1=self.AhiddenLayer1_ddqn.output1,
            n_in=n_hidden1,
            n_out=n_hidden2,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[7],
            b_values = model[9]
        )
        self.AlogRegressionLayer_ddqn = LogisticRegression(
            input1=self.AhiddenLayer2_ddqn.output1,
            n_in=n_hidden2,
            n_out=n_out,
            W_values = model[10],
            b_values = model[11]
        )

        ######target##########
        self.VhiddenLayer1_t = HiddenLayer(
            input1=input2,
            n_in=n_in,
            n_out=n_hidden1,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[0],
            b_values = model[2]
        )
        self.VhiddenLayer2_t = HiddenLayer(
            input1=self.VhiddenLayer1_t.output1,
            n_in=n_hidden1,
            n_out=n_hidden2,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[1],
            b_values = model[3]
        )
        self.VlogRegressionLayer_t = LogisticRegression(
            input1=self.VhiddenLayer2_t.output1,
            n_in=n_hidden2,
            n_out=1,
            W_values = model[4],
            b_values = model[5]
        )
        self.AhiddenLayer1_t = HiddenLayer(
            input1=input2,
            n_in=n_in,
            n_out=n_hidden1,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[6],
            b_values = model[8]
        )
        self.AhiddenLayer2_t = HiddenLayer(
            input1=self.AhiddenLayer1_t.output1,
            n_in=n_hidden1,
            n_out=n_hidden2,
            activation=T.nnet.relu,
#            activation=T.tanh,
            W_values = model[7],
            b_values = model[9]
        )
        self.AlogRegressionLayer_t = LogisticRegression(
            input1=self.AhiddenLayer2_t.output1,
            n_in=n_hidden2,
            n_out=n_out,
            W_values = model[10],
            b_values = model[11]
        )

        self.params = self.VhiddenLayer1.params + self.VhiddenLayer2.params + self.VlogRegressionLayer.params + self.AhiddenLayer1.params + self.AhiddenLayer2.params + self.AlogRegressionLayer.params
        # keep track of model input
        self.Qs = T.extra_ops.repeat(self.VlogRegressionLayer.Q,n_out,axis=1) + (self.AlogRegressionLayer.Q - T.mean(self.AlogRegressionLayer.Q,axis=1,keepdims=True))
        self.Qddqn = T.extra_ops.repeat(self.VlogRegressionLayer_ddqn.Q,n_out,axis=1) + (self.AlogRegressionLayer_ddqn.Q - T.mean(self.AlogRegressionLayer_ddqn.Q,axis=1,keepdims=True))
        self.Qsp = T.extra_ops.repeat(self.VlogRegressionLayer_t.Q,n_out,axis=1) + (self.AlogRegressionLayer_t.Q - T.mean(self.AlogRegressionLayer_t.Q,axis=1,keepdims=True))

        # self.Qs =  (self.AlogRegressionLayer.Q - T.mean(self.AlogRegressionLayer.Q,axis=1,keepdims=True)) + self.VlogRegressionLayer.Q
        # self.Qddqn =  (self.AlogRegressionLayer_ddqn.Q - T.mean(self.AlogRegressionLayer_ddqn.Q,axis=1,keepdims=True)) + self.VlogRegressionLayer_ddqn.Q
        # self.Qsp = (self.AlogRegressionLayer_t.Q - T.mean(self.AlogRegressionLayer_t.Q,axis=1,keepdims=True)) + self.VlogRegressionLayer_t.Q

        # self.Qs = T.extra_ops.repeat(self.VlogRegressionLayer.Q,n_out,axis=1) + (self.AlogRegressionLayer.Q - T.extra_ops.repeat(T.mean(self.AlogRegressionLayer.Q,axis=1,keepdims=True),n_out,axis=1))
        # self.Qddqn = T.extra_ops.repeat(self.VlogRegressionLayer_ddqn.Q,n_out,axis=1) + (self.AlogRegressionLayer_ddqn.Q - T.extra_ops.repeat(T.mean(self.AlogRegressionLayer_ddqn.Q,axis=1,keepdims=True),n_out,axis=1))
        # self.Qsp = T.extra_ops.repeat(self.VlogRegressionLayer_t.Q,n_out,axis=1) + (self.AlogRegressionLayer_t.Q - T.extra_ops.repeat(T.mean(self.AlogRegressionLayer_t.Q,axis=1,keepdims=True),n_out,axis=1))
        self.input1 = input1
        self.input2 = input2
        self.aidx = T.cast(T.round((input1[:,64]*4.0+4.0)*7.0+(input1[:,65]*3.5+3.5)), 'int32')
#        self.cost = T.mean(T.max(self.logRegressionLayer.Qsp,axis=1))
#        self.cost = T.mean(T.max(self.logRegressionLayer.Qsp,axis=1)-T.max(self.logRegressionLayer.Qs,axis=1))
#        self.cost = self.Qs[0,0]-self.Qsp[0,0]

        self.target = input1[:,0]+gamma*T.max(self.Qsp,axis=1)
        self.action_ddqn = T.argmax(self.Qddqn,axis=1)
        self.target_ddqn = input1[:,0]+gamma*self.Qsp[T.arange(self.action_ddqn.shape[0]),self.action_ddqn]
        self.Qcost = T.mean(0.5*(self.target_ddqn-self.Qs[T.arange(self.aidx.shape[0]),self.aidx])**2)
#        self.Qcost = T.mean(0.5*(self.target-self.Qs[T.arange(self.aidx.shape[0]),self.aidx])**2)
        self.cost = self.Qcost#+0.0001*self.L2_sqr
#        self.errors = T.sqrt(T.mean(((input1[:,0]+0.97*T.max(self.logRegressionLayer.Qsp,axis=1)-T.max(self.logRegressionLayer.Qs,axis=1))/(input1[:,0]+0.95*T.max(self.logRegressionLayer.Qsp,axis=1)))**2))
        #######parameters
        self.VWh1 = self.VhiddenLayer1.W
        self.VWh2 = self.VhiddenLayer2.W
        self.Vbh1 = self.VhiddenLayer1.b
        self.Vbh2 = self.VhiddenLayer2.b
        self.VOW  = self.VlogRegressionLayer.W
        self.VOb  = self.VlogRegressionLayer.b
        self.AWh1 = self.AhiddenLayer1.W
        self.AWh2 = self.AhiddenLayer2.W
        self.Abh1 = self.AhiddenLayer1.b
        self.Abh2 = self.AhiddenLayer2.b
        self.AOW  = self.AlogRegressionLayer.W
        self.AOb  = self.AlogRegressionLayer.b
        self.VWh1t = self.VhiddenLayer1_t.W
        self.VWh2t = self.VhiddenLayer2_t.W
        self.Vbh1t = self.VhiddenLayer1_t.b
        self.Vbh2t = self.VhiddenLayer2_t.b
        self.VOWt  = self.VlogRegressionLayer_t.W
        self.VObt  = self.VlogRegressionLayer_t.b
        self.AWh1t = self.AhiddenLayer1_t.W
        self.AWh2t = self.AhiddenLayer2_t.W
        self.Abh1t = self.AhiddenLayer1_t.b
        self.Abh2t = self.AhiddenLayer2_t.b
        self.AOWt  = self.AlogRegressionLayer_t.W
        self.AObt  = self.AlogRegressionLayer_t.b
        self.VWh1ddqn = self.VhiddenLayer1_ddqn.W
        self.VWh2ddqn = self.VhiddenLayer2_ddqn.W
        self.Vbh1ddqn = self.VhiddenLayer1_ddqn.b
        self.Vbh2ddqn = self.VhiddenLayer2_ddqn.b
        self.VOWddqn  = self.VlogRegressionLayer_ddqn.W
        self.VObddqn  = self.VlogRegressionLayer_ddqn.b
        self.AWh1ddqn = self.AhiddenLayer1_ddqn.W
        self.AWh2ddqn = self.AhiddenLayer2_ddqn.W
        self.Abh1ddqn = self.AhiddenLayer1_ddqn.b
        self.Abh2ddqn = self.AhiddenLayer2_ddqn.b
        self.AOWddqn  = self.AlogRegressionLayer_ddqn.W
        self.AObddqn  = self.AlogRegressionLayer_ddqn.b
    def init_weight(self,shape):
#        return numpy.asarray(self.rng.normal(loc=0.0,scale=0.001,size=shape),dtype=theano.config.floatX)
        return numpy.asarray(self.rng.normal(loc=0.0,scale=0.001,size=shape),dtype=theano.config.floatX)
    def init_bias(self,shape):
        return numpy.asarray(numpy.full(shape,1.0),dtype=theano.config.floatX)
    def init_model(self):
        a0 = self.init_weight((self.n_in,self.n_hidden1))
        a1 = self.init_weight((self.n_hidden1,self.n_hidden2))
        a2 = self.init_bias((self.n_hidden1,))
        a3 = self.init_bias((self.n_hidden2,))
        a4 = self.init_weight((self.n_hidden2,1))
        a5 = self.init_bias((1,))
        a6 = self.init_weight((self.n_in,self.n_hidden1))
        a7 = self.init_weight((self.n_hidden1,self.n_hidden2))
        a8 = self.init_bias((self.n_hidden1,))
        a9 = self.init_bias((self.n_hidden2,))
        a10 = self.init_weight((self.n_hidden2,self.n_out))
        a11 = self.init_bias((self.n_out,))
        return (a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11)
