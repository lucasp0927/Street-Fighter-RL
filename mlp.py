import os
import sys
import time
import pickle
import numpy
#import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from mlp_dqn import MLP_DQN
from mlp_ddqn import MLP_DDQN, MLP_DDQN_3
from mlp_duel import MLP_DUEL
from logistic_sgd import load_data
from updates import deepmind_rmsprop
from rank_based import Experience
class RMSProp(object):
    """
    RMSProp with nesterov momentum and gradient rescaling
    """
    def __init__(self, params):
        self.running_square_ = [theano.shared(numpy.zeros_like(p.get_value()))
                                for p in params]
        self.running_avg_ = [theano.shared(numpy.zeros_like(p.get_value()))
                             for p in params]
        self.memory_ = [theano.shared(numpy.zeros_like(p.get_value()))
                        for p in params]
    def updates(self, params, grads, learning_rate, momentum, rescale=5.):
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = T.maximum(rescale, grad_norm)
        # Magic constants
        combination_coeff = 0.9
        minimum_grad = 1E-4
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (scaling_num / scaling_den))
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * T.sqr(grad)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * grad
            rms_grad = T.sqrt(new_square - new_avg ** 2)
            rms_grad = T.maximum(rms_grad, minimum_grad)
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad / rms_grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad / rms_grad
            updates.append((old_square, new_square))
            updates.append((old_avg, new_avg))
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates

class Model(object):
    ### 2 layer hidden network, DQN or DDQN
    def __init__(self,model,slen,gamma=0.995,n_hidden1=25,n_hidden2=25,learning_rate = 0.0002,freeze_interval=1000,momentum=0.0,learner_type = "DDQN",minibatch_size = 20,train_interval = 100):
        self.freeze_interval = freeze_interval
        self.freeze_counter = 0
        self.slen = slen
        #train data
        self.minibatch_size = minibatch_size
        self.train_interval = train_interval
        self.train_set_x = theano.shared(numpy.zeros([minibatch_size*train_interval,slen],dtype=theano.config.floatX),borrow=True)
        self.train_set_y = theano.shared(numpy.zeros([minibatch_size*train_interval,slen],dtype=theano.config.floatX),borrow=True)
        #variables
        self.index = T.lscalar()
        self.s = T.matrix('s')  # the data is presented as rasterized images
        self.sp = T.matrix('sp') #s prime
        self.rng = numpy.random.RandomState(None)
        if learner_type == "DDQN":
            self.classifier = MLP_DDQN(
                rng=self.rng,
                input1=self.s,
                input2=self.sp,
                n_in=slen,
                n_hidden1=n_hidden1,
                n_hidden2=n_hidden2,
                n_out=3,
                model = model,
                gamma = gamma
            )
        elif learner_type == "DQN":
            self.classifier = MLP_DQN(
                rng=self.rng,
                input1=self.s,
                input2=self.sp,
                n_in=slen,
                n_hidden1=n_hidden1,
                n_hidden2=n_hidden2,
                n_out=3,
                model = model,
                gamma = gamma
            )
        self.cost_v = self.classifier.cost_v
        self.cost = self.classifier.cost
        self.rmsprop = RMSProp(self.classifier.params)
        self.gparams = [T.grad(self.cost, param) for param in self.classifier.params]
#        self.updates_no_m = self.rmsprop.updates(self.classifier.params,self.gparams,learning_rate,0.0)
#        self.updates = self.rmsprop.updates(self.classifier.params,self.gparams,learning_rate,momentum)
        self.updates = deepmind_rmsprop(self.gparams,self.classifier.params,learning_rate,momentum,1e-4)
        self.model = (self.classifier.Wh1.get_value(borrow=True),
                      self.classifier.Wh2.get_value(borrow=True),
                      self.classifier.bh1.get_value(borrow=True),
                      self.classifier.bh2.get_value(borrow=True),
                      self.classifier.OW.get_value(borrow=True),
                      self.classifier.Ob.get_value(borrow=True))
        self.model_to_save = (self.classifier.Wh1.get_value(borrow=True),
                              self.classifier.Wh2.get_value(borrow=True),
                              self.classifier.bh1.get_value(borrow=True),
                              self.classifier.bh2.get_value(borrow=True),
                              self.classifier.OW.get_value(borrow=True),
                              self.classifier.Ob.get_value(borrow=True))
        self.to_save_id = 0
        self.saved = True
        self.train_model_prioritize = theano.function(
            inputs=[self.index],
            outputs=self.cost_v,
            updates=self.updates,
            givens={
                    self.s: self.train_set_x[self.index * minibatch_size:(self.index + 1) * minibatch_size],
                    self.sp: self.train_set_y[self.index * minibatch_size:(self.index + 1) * minibatch_size]
            }
        )

        self.train_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                    self.s: self.train_set_x[self.index * minibatch_size:(self.index + 1) * minibatch_size],
                    self.sp: self.train_set_y[self.index * minibatch_size:(self.index + 1) * minibatch_size]
            }
        )
        self.report_action = theano.function(inputs = [self.s],outputs = self.classifier.aidx,allow_input_downcast=True)
        self.action = theano.function(
            inputs = [self.s],
            outputs = T.argmax(self.classifier.Qs),
            allow_input_downcast=True
            )

    def load_model(self,filename):
        self.model = pickle.load(open(filename,'rb'))
        self.classifier.Wh1.set_value(self.model[0])
        self.classifier.Wh2.set_value(self.model[1])
        self.classifier.bh1.set_value(self.model[2])
        self.classifier.bh2.set_value(self.model[3])
        self.classifier.OW.set_value(self.model[4])
        self.classifier.Ob.set_value(self.model[5])
        self.classifier.Wh1t.set_value(self.model[0])
        self.classifier.Wh2t.set_value(self.model[1])
        self.classifier.bh1t.set_value(self.model[2])
        self.classifier.bh2t.set_value(self.model[3])
        self.classifier.OWt.set_value(self.model[4])
        self.classifier.Obt.set_value(self.model[5])
        self.classifier.Wh1ddqn.set_value(self.model[0])
        self.classifier.Wh2ddqn.set_value(self.model[1])
        self.classifier.bh1ddqn.set_value(self.model[2])
        self.classifier.bh2ddqn.set_value(self.model[3])
        self.classifier.OWddqn.set_value(self.model[4])
        self.classifier.Obddqn.set_value(self.model[5])

    def update_target_value(self):
        self.model = (self.classifier.Wh1.get_value(borrow=True),
                      self.classifier.Wh2.get_value(borrow=True),
                      self.classifier.bh1.get_value(borrow=True),
                      self.classifier.bh2.get_value(borrow=True),
                      self.classifier.OW.get_value(borrow=True),
                      self.classifier.Ob.get_value(borrow=True))
        self.classifier.Wh1t.set_value(self.model[0])
        self.classifier.Wh2t.set_value(self.model[1])
        self.classifier.bh1t.set_value(self.model[2])
        self.classifier.bh2t.set_value(self.model[3])
        self.classifier.OWt.set_value(self.model[4])
        self.classifier.Obt.set_value(self.model[5])

    def update_ddqn_value(self):
        self.model = (self.classifier.Wh1.get_value(borrow=True),
                      self.classifier.Wh2.get_value(borrow=True),
                      self.classifier.bh1.get_value(borrow=True),
                      self.classifier.bh2.get_value(borrow=True),
                      self.classifier.OW.get_value(borrow=True),
                      self.classifier.Ob.get_value(borrow=True))
        self.classifier.Wh1ddqn.set_value(self.model[0])
        self.classifier.Wh2ddqn.set_value(self.model[1])
        self.classifier.bh1ddqn.set_value(self.model[2])
        self.classifier.bh2ddqn.set_value(self.model[3])
        self.classifier.OWddqn.set_value(self.model[4])
        self.classifier.Obddqn.set_value(self.model[5])

    def update_model(self):
        self.model = (self.classifier.Wh1.get_value(borrow=True),
                      self.classifier.Wh2.get_value(borrow=True),
                      self.classifier.bh1.get_value(borrow=True),
                      self.classifier.bh2.get_value(borrow=True),
                      self.classifier.OW.get_value(borrow=True),
                      self.classifier.Ob.get_value(borrow=True))

    def set_to_save_id(self,x):
        self.to_save_id = x

    def update_to_save_model(self,x):
        self.saved = False
        self.set_to_save_id(x)
        self.model_to_save = (self.classifier.Wh1.get_value(borrow=True),
                              self.classifier.Wh2.get_value(borrow=True),
                              self.classifier.bh1.get_value(borrow=True),
                              self.classifier.bh2.get_value(borrow=True),
                              self.classifier.OW.get_value(borrow=True),
                              self.classifier.Ob.get_value(borrow=True))

    def train_prioritize(self,experience,train_step,batch_size,n_batch=10):
        train_num = n_batch*batch_size
        train_set_x = numpy.zeros([train_num,self.slen])
        train_set_y = numpy.zeros([train_num,self.slen])

        e_id_list = numpy.zeros(train_num)
        delta_list = numpy.zeros(train_num)
        #prepare data
        for i in range(n_batch):
            sample, w, e_id = experience.sample(train_step)
            e_id_list[i*batch_size:(i+1)*batch_size] = e_id[:]
            for j in range(batch_size):
                train_set_x[i*batch_size+j,:] = sample[j][0][:]
                train_set_y[i*batch_size+j,:] = sample[j][3][:]
        self.train_set_x.set_value(numpy.asarray(train_set_x,dtype=theano.config.floatX))
        self.train_set_y.set_value(numpy.asarray(train_set_y,dtype=theano.config.floatX))
        #train
        minibatch_cost = 0
        for minibatch_index in range(n_batch):
            minibatch_cost_v = self.train_model_prioritize(minibatch_index)
#            print(minibatch_cost_v)
            delta_list[i*batch_size:(i+1)*batch_size] = minibatch_cost_v[:]
            minibatch_cost += numpy.mean(minibatch_cost_v)
            self.update_ddqn_value()
        experience.update_priority(e_id_list, delta_list)
        #print(time.time()-t)
        minibatch_avg_cost = minibatch_cost/n_batch
        if numpy.isnan(minibatch_avg_cost):
            print("minibatch cost is NAN!")
        print(
            'minibatch error %f' %
            (
                minibatch_avg_cost
            )
        )
        #update target
        self.freeze_counter += 1
        if self.freeze_counter == self.freeze_interval:
            self.freeze_counter = 0
            self.update_target_value()

    def train(self,dataset,n_epochs=1,batch_size=20,n_batch=10):
        datasets = load_data(dataset,n_batch,batch_size,self.slen,self.train_interval)
        train_set_x, train_set_y = datasets
        self.train_set_x.set_value(numpy.asarray(train_set_x,dtype=theano.config.floatX))
        self.train_set_y.set_value(numpy.asarray(train_set_y,dtype=theano.config.floatX))
        #n_train_batches = train_set_x.shape[0] // batch_size
        for i in range(n_epochs):
            self.freeze_counter += 1
            minibatch_cost = 0
            #t=time.time()
            for minibatch_index in range(self.train_interval):
                minibatch_cost += self.train_model(minibatch_index)
                # minibatch_cost += self.train_model(
                #     train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],
                #     train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size])
                self.update_ddqn_value()
            #print(time.time()-t)
            minibatch_avg_cost = minibatch_cost/self.train_interval
            if numpy.isnan(minibatch_avg_cost):
                print("minibatch cost is NAN!")
            print(
                'minibatch error %f' %
                (
                    minibatch_avg_cost
                )
            )
        if self.freeze_counter == self.freeze_interval:
            self.freeze_counter = 0
            self.update_target_value()

class Model_duel(object):
    #Dueling network, 2layers hidden network
    def __init__(self,model,slen,gamma=0.98,n_hidden1=25,n_hidden2=25,learning_rate = 0.0002,freeze_interval=1000,momentum=0.0,minibatch_size=20,train_interval = 100):
        self.freeze_interval = freeze_interval
        self.freeze_counter = 0
        self.slen = slen
        self.minibatch_size = minibatch_size
        self.train_interval = train_interval
        self.train_set_x = theano.shared(numpy.zeros([minibatch_size*train_interval,slen],dtype=theano.config.floatX),borrow=True)
        self.train_set_y = theano.shared(numpy.zeros([minibatch_size*train_interval,slen],dtype=theano.config.floatX),borrow=True)

        self.s = T.matrix('s')  # the data is presented as rasterized images
        self.sp = T.matrix('sp') #s prime
        self.rng = numpy.random.RandomState(None)
        self.classifier = MLP_DUEL(
            rng=self.rng,
            input1=self.s,
            input2=self.sp,
            n_in=slen,
            n_hidden1=n_hidden1,
            n_hidden2=n_hidden2,
            n_out=63,
            model = model,
            gamma = gamma
        )
        self.cost = self.classifier.cost
        self.rmsprop = RMSProp(self.classifier.params)
        self.gparams = [T.grad(self.cost, param) for param in self.classifier.params]
#        self.updates_no_m = self.rmsprop.updates(self.classifier.params,self.gparams,learning_rate,0.0)
#        self.updates = deepmind_rmsprop(self.gparams,self.classifier.params,learning_rate,momentum,1e-4)
        self.updates = self.rmsprop.updates(self.classifier.params,self.gparams,learning_rate,momentum)
        self.model = (self.classifier.VWh1.get_value(borrow=True),
                      self.classifier.VWh2.get_value(borrow=True),
                      self.classifier.Vbh1.get_value(borrow=True),
                      self.classifier.Vbh2.get_value(borrow=True),
                      self.classifier.VOW.get_value(borrow=True),
                      self.classifier.VOb.get_value(borrow=True),
                      self.classifier.AWh1.get_value(borrow=True),
                      self.classifier.AWh2.get_value(borrow=True),
                      self.classifier.Abh1.get_value(borrow=True),
                      self.classifier.Abh2.get_value(borrow=True),
                      self.classifier.AOW.get_value(borrow=True),
                      self.classifier.AOb.get_value(borrow=True))
        self.model_to_save = (self.classifier.VWh1.get_value(borrow=True),
                      self.classifier.VWh2.get_value(borrow=True),
                      self.classifier.Vbh1.get_value(borrow=True),
                      self.classifier.Vbh2.get_value(borrow=True),
                      self.classifier.VOW.get_value(borrow=True),
                      self.classifier.VOb.get_value(borrow=True),
                      self.classifier.AWh1.get_value(borrow=True),
                      self.classifier.AWh2.get_value(borrow=True),
                      self.classifier.Abh1.get_value(borrow=True),
                      self.classifier.Abh2.get_value(borrow=True),
                      self.classifier.AOW.get_value(borrow=True),
                      self.classifier.AOb.get_value(borrow=True))
        self.to_save_id = 0
        self.saved = True
        self.index = T.lscalar()  # index to a [mini]batch
        # self.validate_model = theano.function(
        #     inputs=[self.s,self.sp],
        #     outputs=self.cost,
        #     # givens={
        #     #     self.s: valid_set_x[index * batch_size:(index + 1) * batch_size],
        #     #     self.sp: valid_set_y[index * batch_size:(index + 1) * batch_size]
        #     # },
        #     on_unused_input='warn'
        # )
        # self.train_model = theano.function(
        #     inputs=[self.s,self.sp],
        #     outputs=self.cost,
        #     updates=self.updates,
        #     on_unused_input='warn'
        # )
        self.train_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                    self.s: self.train_set_x[self.index * minibatch_size:(self.index + 1) * minibatch_size],
                    self.sp: self.train_set_y[self.index * minibatch_size:(self.index + 1) * minibatch_size]
            }
        )
        # self.train_model_no_m = theano.function(
        #     inputs=[self.s,self.sp],
        #     outputs=self.cost,
        #     updates=self.updates_no_m,
        #     on_unused_input='warn'
        # )
        self.report_action = theano.function(inputs = [self.s],outputs = self.classifier.aidx)
        self.action = theano.function(
            inputs = [self.s],
            outputs = T.argmax(self.classifier.Qs),
            allow_input_downcast=True
            )

    def load_model(self,filename):
        self.model = pickle.load(open(filename,'rb'))
        self.classifier.VWh1.set_value(self.model[0])
        self.classifier.VWh2.set_value(self.model[1])
        self.classifier.Vbh1.set_value(self.model[2])
        self.classifier.Vbh2.set_value(self.model[3])
        self.classifier.VOW.set_value(self.model[4])
        self.classifier.VOb.set_value(self.model[5])
        self.classifier.AWh1.set_value(self.model[6])
        self.classifier.AWh2.set_value(self.model[7])
        self.classifier.Abh1.set_value(self.model[8])
        self.classifier.Abh2.set_value(self.model[9])
        self.classifier.AOW.set_value(self.model[10])
        self.classifier.AOb.set_value(self.model[11])
        self.classifier.VWh1t.set_value(self.model[0])
        self.classifier.VWh2t.set_value(self.model[1])
        self.classifier.Vbh1t.set_value(self.model[2])
        self.classifier.Vbh2t.set_value(self.model[3])
        self.classifier.VOWt.set_value(self.model[4])
        self.classifier.VObt.set_value(self.model[5])
        self.classifier.AWh1t.set_value(self.model[6])
        self.classifier.AWh2t.set_value(self.model[7])
        self.classifier.Abh1t.set_value(self.model[8])
        self.classifier.Abh2t.set_value(self.model[9])
        self.classifier.AOWt.set_value(self.model[10])
        self.classifier.AObt.set_value(self.model[11])
        self.classifier.VWh1ddqn.set_value(self.model[0])
        self.classifier.VWh2ddqn.set_value(self.model[1])
        self.classifier.Vbh1ddqn.set_value(self.model[2])
        self.classifier.Vbh2ddqn.set_value(self.model[3])
        self.classifier.VOWddqn.set_value(self.model[4])
        self.classifier.VObddqn.set_value(self.model[5])
        self.classifier.AWh1ddqn.set_value(self.model[6])
        self.classifier.AWh2ddqn.set_value(self.model[7])
        self.classifier.Abh1ddqn.set_value(self.model[8])
        self.classifier.Abh2ddqn.set_value(self.model[9])
        self.classifier.AOWddqn.set_value(self.model[10])
        self.classifier.AObddqn.set_value(self.model[11])

        self.classifier.AObt.set_value(self.model[11])

    def update_target_value(self):
        self.model = (self.classifier.VWh1.get_value(borrow=True),
                      self.classifier.VWh2.get_value(borrow=True),
                      self.classifier.Vbh1.get_value(borrow=True),
                      self.classifier.Vbh2.get_value(borrow=True),
                      self.classifier.VOW.get_value(borrow=True),
                      self.classifier.VOb.get_value(borrow=True),
                      self.classifier.AWh1.get_value(borrow=True),
                      self.classifier.AWh2.get_value(borrow=True),
                      self.classifier.Abh1.get_value(borrow=True),
                      self.classifier.Abh2.get_value(borrow=True),
                      self.classifier.AOW.get_value(borrow=True),
                      self.classifier.AOb.get_value(borrow=True))
        self.classifier.VWh1t.set_value(self.model[0])
        self.classifier.VWh2t.set_value(self.model[1])
        self.classifier.Vbh1t.set_value(self.model[2])
        self.classifier.Vbh2t.set_value(self.model[3])
        self.classifier.VOWt.set_value(self.model[4])
        self.classifier.VObt.set_value(self.model[5])
        self.classifier.AWh1t.set_value(self.model[6])
        self.classifier.AWh2t.set_value(self.model[7])
        self.classifier.Abh1t.set_value(self.model[8])
        self.classifier.Abh2t.set_value(self.model[9])
        self.classifier.AOWt.set_value(self.model[10])
        self.classifier.AObt.set_value(self.model[11])


    def update_ddqn_value(self):
        self.model = (self.classifier.VWh1.get_value(borrow=True),
                      self.classifier.VWh2.get_value(borrow=True),
                      self.classifier.Vbh1.get_value(borrow=True),
                      self.classifier.Vbh2.get_value(borrow=True),
                      self.classifier.VOW.get_value(borrow=True),
                      self.classifier.VOb.get_value(borrow=True),
                      self.classifier.AWh1.get_value(borrow=True),
                      self.classifier.AWh2.get_value(borrow=True),
                      self.classifier.Abh1.get_value(borrow=True),
                      self.classifier.Abh2.get_value(borrow=True),
                      self.classifier.AOW.get_value(borrow=True),
                      self.classifier.AOb.get_value(borrow=True))
        self.classifier.VWh1ddqn.set_value(self.model[0])
        self.classifier.VWh2ddqn.set_value(self.model[1])
        self.classifier.Vbh1ddqn.set_value(self.model[2])
        self.classifier.Vbh2ddqn.set_value(self.model[3])
        self.classifier.VOWddqn.set_value(self.model[4])
        self.classifier.VObddqn.set_value(self.model[5])
        self.classifier.AWh1ddqn.set_value(self.model[6])
        self.classifier.AWh2ddqn.set_value(self.model[7])
        self.classifier.Abh1ddqn.set_value(self.model[8])
        self.classifier.Abh2ddqn.set_value(self.model[9])
        self.classifier.AOWddqn.set_value(self.model[10])
        self.classifier.AObddqn.set_value(self.model[11])

    def update_model(self):
        self.model = (self.classifier.VWh1.get_value(borrow=True),
                      self.classifier.VWh2.get_value(borrow=True),
                      self.classifier.Vbh1.get_value(borrow=True),
                      self.classifier.Vbh2.get_value(borrow=True),
                      self.classifier.VOW.get_value(borrow=True),
                      self.classifier.VOb.get_value(borrow=True),
                      self.classifier.AWh1.get_value(borrow=True),
                      self.classifier.AWh2.get_value(borrow=True),
                      self.classifier.Abh1.get_value(borrow=True),
                      self.classifier.Abh2.get_value(borrow=True),
                      self.classifier.AOW.get_value(borrow=True),
                      self.classifier.AOb.get_value(borrow=True))

    def set_to_save_id(self,x):
        self.to_save_id = x

    def update_to_save_model(self,x):
        self.saved = False
        self.set_to_save_id(x)
        self.model_to_save = (self.classifier.VWh1.get_value(borrow=True),
                              self.classifier.VWh2.get_value(borrow=True),
                              self.classifier.Vbh1.get_value(borrow=True),
                              self.classifier.Vbh2.get_value(borrow=True),
                              self.classifier.VOW.get_value(borrow=True),
                              self.classifier.VOb.get_value(borrow=True),
                              self.classifier.AWh1.get_value(borrow=True),
                              self.classifier.AWh2.get_value(borrow=True),
                              self.classifier.Abh1.get_value(borrow=True),
                              self.classifier.Abh2.get_value(borrow=True),
                              self.classifier.AOW.get_value(borrow=True),
                              self.classifier.AOb.get_value(borrow=True))


    def train(self,dataset,n_epochs=1,batch_size=20,n_batch=10):
        datasets = load_data(dataset,n_batch,batch_size,self.slen,self.train_interval)
        train_set_x, train_set_y = datasets
        self.train_set_x.set_value(numpy.asarray(train_set_x,dtype=theano.config.floatX))
        self.train_set_y.set_value(numpy.asarray(train_set_y,dtype=theano.config.floatX))
        for i in range(n_epochs):
            self.freeze_counter += 1
            minibatch_cost = 0
            for minibatch_index in range(self.train_interval):
                minibatch_cost += self.train_model(minibatch_index)
                self.update_ddqn_value()
            minibatch_avg_cost = minibatch_cost/self.train_interval
            if numpy.isnan(minibatch_avg_cost):
                print("minibatch cost is NAN!")
            print(
                'minibatch error %f' %
                (
                    minibatch_avg_cost
                )
            )
            # print(
            #     'minibatch error %f, validation error %f' %
            #     (
            #         minibatch_avg_cost,this_validation_loss
            #     )
            # )
        if self.freeze_counter == self.freeze_interval:
            self.freeze_counter = 0
            self.update_target_value()

class Model_3(object):
    #3 layers hidden network.
    def __init__(self,model,slen,gamma=0.99,n_hidden1=25,n_hidden2=25,n_hidden3 = 25,learning_rate = 0.0002,freeze_interval=1000,momentum=0.0,learner_type = "DDQN",minibatch_size = 20,train_interval = 100):
        print("initialize model_3")
        self.freeze_interval = freeze_interval
        self.freeze_counter = 0
        self.slen = slen
        #train data
        self.minibatch_size = minibatch_size
        self.train_interval = train_interval
        self.train_set_x = theano.shared(numpy.zeros([minibatch_size*train_interval,slen],dtype=theano.config.floatX),borrow=True)
        self.train_set_y = theano.shared(numpy.zeros([minibatch_size*train_interval,slen],dtype=theano.config.floatX),borrow=True)
        #variables
        self.index = T.lscalar()
        self.s = T.matrix('s')  # the data is presented as rasterized images
        self.sp = T.matrix('sp') #s prime
        self.rng = numpy.random.RandomState(None)
        if learner_type == "DDQN":
            self.classifier = MLP_DDQN_3(
                rng=self.rng,
                input1=self.s,
                input2=self.sp,
                n_in=slen,
                n_hidden1=n_hidden1,
                n_hidden2=n_hidden2,
                n_hidden3=n_hidden3,
                n_out=63,
                model = model,
                gamma = gamma
            )
        self.cost = self.classifier.cost
        self.cost_v = self.classifier.cost_v
        self.rmsprop = RMSProp(self.classifier.params)
        self.gparams = [T.grad(self.cost, param) for param in self.classifier.params]
#        self.updates_no_m = self.rmsprop.updates(self.classifier.params,self.gparams,learning_rate,0.0)
#        self.updates = self.rmsprop.updates(self.classifier.params,self.gparams,learning_rate,momentum)
        self.updates = deepmind_rmsprop(self.gparams,self.classifier.params,learning_rate,momentum,1e-4)
        self.model = (self.classifier.Wh1.get_value(borrow=True),
                      self.classifier.bh1.get_value(borrow=True),
                      self.classifier.Wh2.get_value(borrow=True),
                      self.classifier.bh2.get_value(borrow=True),
                      self.classifier.Wh3.get_value(borrow=True),
                      self.classifier.bh3.get_value(borrow=True),
                      self.classifier.OW.get_value(borrow=True),
                      self.classifier.Ob.get_value(borrow=True))
        self.model_to_save = (self.classifier.Wh1.get_value(borrow=True),
                              self.classifier.bh1.get_value(borrow=True),
                              self.classifier.Wh2.get_value(borrow=True),
                              self.classifier.bh2.get_value(borrow=True),
                              self.classifier.Wh3.get_value(borrow=True),
                              self.classifier.bh3.get_value(borrow=True),
                              self.classifier.OW.get_value(borrow=True),
                              self.classifier.Ob.get_value(borrow=True))
        self.to_save_id = 0
        self.saved = True

        self.train_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                    self.s: self.train_set_x[self.index * minibatch_size:(self.index + 1) * minibatch_size],
                    self.sp: self.train_set_y[self.index * minibatch_size:(self.index + 1) * minibatch_size]
            }
        )
        self.train_model_prioritize = theano.function(
            inputs=[self.index],
            outputs=self.cost_v,
            updates=self.updates,
            givens={
                    self.s: self.train_set_x[self.index * minibatch_size:(self.index + 1) * minibatch_size],
                    self.sp: self.train_set_y[self.index * minibatch_size:(self.index + 1) * minibatch_size]
            }
        )
        self.report_action = theano.function(inputs = [self.s],outputs = self.classifier.aidx,allow_input_downcast=True)
        self.action = theano.function(
            inputs = [self.s],
            outputs = T.argmax(self.classifier.Qs),
            allow_input_downcast=True
            )

    def load_model(self,filename):
        self.model = pickle.load(open(filename,'rb'))
        self.classifier.Wh1.set_value(self.model[0])
        self.classifier.bh1.set_value(self.model[1])
        self.classifier.Wh2.set_value(self.model[2])
        self.classifier.bh2.set_value(self.model[3])
        self.classifier.Wh3.set_value(self.model[4])
        self.classifier.bh3.set_value(self.model[5])
        self.classifier.OW.set_value(self.model[6])
        self.classifier.Ob.set_value(self.model[7])
        self.classifier.Wh1t.set_value(self.model[0])
        self.classifier.bh1t.set_value(self.model[1])
        self.classifier.Wh2t.set_value(self.model[2])
        self.classifier.bh2t.set_value(self.model[3])
        self.classifier.Wh3t.set_value(self.model[4])
        self.classifier.bh3t.set_value(self.model[5])
        self.classifier.OWt.set_value(self.model[6])
        self.classifier.Obt.set_value(self.model[7])
        self.classifier.Wh1ddqn.set_value(self.model[0])
        self.classifier.bh1ddqn.set_value(self.model[1])
        self.classifier.Wh2ddqn.set_value(self.model[2])
        self.classifier.bh2ddqn.set_value(self.model[3])
        self.classifier.Wh3ddqn.set_value(self.model[4])
        self.classifier.bh3ddqn.set_value(self.model[5])
        self.classifier.OWddqn.set_value(self.model[4])
        self.classifier.Obddqn.set_value(self.model[5])

    def update_target_value(self):
        self.model = (self.classifier.Wh1.get_value(borrow=True),
                      self.classifier.bh1.get_value(borrow=True),
                      self.classifier.Wh2.get_value(borrow=True),
                      self.classifier.bh2.get_value(borrow=True),
                      self.classifier.Wh3.get_value(borrow=True),
                      self.classifier.bh3.get_value(borrow=True),
                      self.classifier.OW.get_value(borrow=True),
                      self.classifier.Ob.get_value(borrow=True))
        self.classifier.Wh1t.set_value(self.model[0])
        self.classifier.bh1t.set_value(self.model[1])
        self.classifier.Wh2t.set_value(self.model[2])
        self.classifier.bh2t.set_value(self.model[3])
        self.classifier.Wh3t.set_value(self.model[4])
        self.classifier.bh3t.set_value(self.model[5])
        self.classifier.OWt.set_value(self.model[6])
        self.classifier.Obt.set_value(self.model[7])

    def update_ddqn_value(self):
        self.model = (self.classifier.Wh1.get_value(borrow=True),
                      self.classifier.bh1.get_value(borrow=True),
                      self.classifier.Wh2.get_value(borrow=True),
                      self.classifier.bh2.get_value(borrow=True),
                      self.classifier.Wh3.get_value(borrow=True),
                      self.classifier.bh3.get_value(borrow=True),
                      self.classifier.OW.get_value(borrow=True),
                      self.classifier.Ob.get_value(borrow=True))
        self.classifier.Wh1ddqn.set_value(self.model[0])
        self.classifier.bh1ddqn.set_value(self.model[1])
        self.classifier.Wh2ddqn.set_value(self.model[2])
        self.classifier.bh2ddqn.set_value(self.model[3])
        self.classifier.Wh3ddqn.set_value(self.model[4])
        self.classifier.bh3ddqn.set_value(self.model[5])
        self.classifier.OWddqn.set_value(self.model[6])
        self.classifier.Obddqn.set_value(self.model[7])

    def update_model(self):
        self.model = (self.classifier.Wh1.get_value(borrow=True),
                      self.classifier.bh1.get_value(borrow=True),
                      self.classifier.Wh2.get_value(borrow=True),
                      self.classifier.bh2.get_value(borrow=True),
                      self.classifier.Wh3.get_value(borrow=True),
                      self.classifier.bh3.get_value(borrow=True),
                      self.classifier.OW.get_value(borrow=True),
                      self.classifier.Ob.get_value(borrow=True))

    def set_to_save_id(self,x):
        self.to_save_id = x

    def update_to_save_model(self,x):
        self.saved = False
        self.set_to_save_id(x)
        self.model_to_save = (self.classifier.Wh1.get_value(borrow=True),
                              self.classifier.bh1.get_value(borrow=True),
                              self.classifier.Wh2.get_value(borrow=True),
                              self.classifier.bh2.get_value(borrow=True),
                              self.classifier.Wh3.get_value(borrow=True),
                              self.classifier.bh3.get_value(borrow=True),
                              self.classifier.OW.get_value(borrow=True),
                              self.classifier.Ob.get_value(borrow=True))

    def train_prioritize(self,experience,train_step,batch_size,n_batch=10):
        train_num = n_batch*batch_size
        train_set_x = numpy.zeros([train_num,self.slen])
        train_set_y = numpy.zeros([train_num,self.slen])

        e_id_list = numpy.zeros(train_num)
        delta_list = numpy.zeros(train_num)
        #prepare data
        for i in range(n_batch):
            sample, w, e_id = experience.sample(train_step)
            e_id_list[i*batch_size:(i+1)*batch_size] = e_id[:]
            for j in range(batch_size):
                train_set_x[i*batch_size+j,:] = sample[j][0][:]
                train_set_y[i*batch_size+j,:] = sample[j][3][:]
        self.train_set_x.set_value(numpy.asarray(train_set_x,dtype=theano.config.floatX))
        self.train_set_y.set_value(numpy.asarray(train_set_y,dtype=theano.config.floatX))
        #train
        minibatch_cost = 0
        for minibatch_index in range(n_batch):
            minibatch_cost_v = self.train_model_prioritize(minibatch_index)
#            print(minibatch_cost_v)
            delta_list[i*batch_size:(i+1)*batch_size] = minibatch_cost_v[:]
            minibatch_cost += numpy.mean(minibatch_cost_v)
            self.update_ddqn_value()
        experience.update_priority(e_id_list, delta_list)
        #print(time.time()-t)
        minibatch_avg_cost = minibatch_cost/n_batch
        if numpy.isnan(minibatch_avg_cost):
            print("minibatch cost is NAN!")
        print(
            'minibatch error %f' %
            (
                minibatch_avg_cost
            )
        )
        #update target
        self.freeze_counter += 1
        if self.freeze_counter == self.freeze_interval:
            self.freeze_counter = 0
            self.update_target_value()

    def train(self,dataset,n_epochs=1,batch_size=20,n_batch=10):
        datasets = load_data(dataset,n_batch,batch_size,self.slen,self.train_interval)
        train_set_x, train_set_y = datasets
        self.train_set_x.set_value(numpy.asarray(train_set_x,dtype=theano.config.floatX))
        self.train_set_y.set_value(numpy.asarray(train_set_y,dtype=theano.config.floatX))
        #n_train_batches = train_set_x.shape[0] // batch_size
        for i in range(n_epochs):
            self.freeze_counter += 1
            minibatch_cost = 0
            #t=time.time()
            for minibatch_index in range(self.train_interval):
                minibatch_cost += self.train_model(minibatch_index)
                self.update_ddqn_value()
            #print(time.time()-t)
            minibatch_avg_cost = minibatch_cost/self.train_interval
            if numpy.isnan(minibatch_avg_cost):
                print("minibatch cost is NAN!")
            print(
                'minibatch error %f' %
                (
                    minibatch_avg_cost
                )
            )
        if self.freeze_counter == self.freeze_interval:
            self.freeze_counter = 0
            self.update_target_value()
