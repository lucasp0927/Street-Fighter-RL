import numpy as np
import pickle
import random
from mlp import Model
from enum import Enum
import pylab

class lstate(Enum):
    fill_current_state = 1
    fill_memory_pool = 2
    train = 3
    evaluate = 4

class Player(object):
    def __init__(self,dlen,state_frame,learning_rate,memory_pool_size = 10000,train_interval = 100,freeze_interval=20,exp_epoch = 2000,epsilon_rate = 0.0001,evaluate_interval = 100,save_threshold = 50,plot_fig=True,filename="best_model.pkl"):
        #hyper-parameters
        self.train_interval = train_interval
        self.dlen = dlen
        self.state_frame = state_frame
        self.state_length = dlen*state_frame
        self.memory_pool_size = memory_pool_size
        self.freeze_interval = 20
        self.learning_rate = learning_rate
        #variables
        self.tot_frame_counter = -1*state_frame
        self.train_counter = 0
        self.epoch = 0
        self.current_state = np.zeros([1,self.state_length])
        self.memory_pool = np.zeros([memory_pool_size,self.state_length])
        self.model = Model(self.load_model(filename),self.state_length,gamma=0.99,learning_rate = learning_rate,freeze_interval=freeze_interval)
        self.ls = lstate.fill_current_state
        #### greedy exploration
        self.epsilon = 1.0
        self.final_epsilon = 0.1
        self.exploration_epoch = exp_epoch
        self.epsilon_rate = epsilon_rate
        #### evaluation
        self.evaluate_interval = evaluate_interval
        self.evaluate_reward = 0.0
        self.evaluate_counter = 0
        self.evaluate_frame = 10000
        self.save_threshold = save_threshold
        ###plt
        self.plot_fig = plot_fig
        self.plot_epoch = np.array([])
        self.plot_reward = np.array([])
        # if self.plot_fig:
        #     pylab.ion()
        #     pylab.plot(self.plot_epoch,self.plot_reward)[0]
        #     pylab.draw()
        #     pylab.show()
        ##### print hyper parameters
        print("learning rate: " + str(self.learning_rate))
        print("memory pool size: " + str(self.memory_pool_size))
        print("train interval: " + str(self.train_interval)+" frame")
        print("freeze interval: " + str(self.freeze_interval)+ " epoch")
        print("exp epoch: " + str(self.exploration_epoch)+ " epoch")
        print("epsilon rate: " + str(self.epsilon_rate)+" /epoch")
    def model_action(self,s,epsilon):
        if random.random() < epsilon:
            action_idx = random.randint(0,2)
        else:
            assert s.shape[0] == 1
            action_idx = self.model.action(s)
        assert action_idx >=0 and action_idx <=2
        return action_idx

    def load_model(self,filename):
        return pickle.load(open(filename,'rb'))

    def push_to_current_state(self,s):
        assert s.size == self.dlen
        self.current_state = np.roll(self.current_state,self.dlen,axis=1)
        self.current_state[0,0:self.dlen] = s

    def learner_state(self):
        self.tot_frame_counter += 1
        if self.tot_frame_counter <= 0:
            self.ls = lstate.fill_current_state
        elif self.tot_frame_counter <= self.memory_pool_size:
            self.ls = lstate.fill_memory_pool
        elif self.tot_frame_counter > self.memory_pool_size:
            if self.epoch % self.evaluate_interval == self.evaluate_interval -1:
                self.evaluate_counter = 0
            if self.epoch % self.evaluate_interval == 0 and self.evaluate_counter <= self.evaluate_frame:
                self.ls = lstate.evaluate
            else:
                self.ls = lstate.train

    def draw(self):
        return False
#        return self.ls == lstate.evaluate

    def update_epsilon(self):
        if self.epoch >= self.exploration_epoch and self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_rate

    def train(self):
        if self.train_counter == 0:
            self.update_epsilon()
            self.epoch += 1
            #print("epoch: "+ str(self.epoch))
            self.model.train(self.memory_pool,n_epochs=1 ,batch_size = 32,n_batch=self.train_interval)
            self.memory_pool = np.roll(self.memory_pool,self.train_interval,axis = 0)
        self.train_counter += 1
        self.memory_pool[self.train_interval - self.train_counter,:] = self.current_state
        if self.train_counter == self.train_interval:
            self.train_counter = 0

    def get_reward(self):
        return self.current_state[0,0]

    def evaluate(self):
        if self.evaluate_counter == 0:
            self.evaluate_reward = 0.0
        elif self.evaluate_counter == self.evaluate_frame:
            print("epoch:" + str(self.epoch))
            print("epsilon:" + str(self.epsilon))
            print("reward:" + str(self.evaluate_reward))
            self.plot_epoch = np.append(self.plot_epoch,self.epoch)
            self.plot_reward = np.append(self.plot_reward,self.evaluate_reward)
            with open('log.pkl', 'wb') as f:
                pickle.dump((self.plot_epoch,self.plot_reward), f)

            if self.evaluate_reward >= self.save_threshold:
                filename = "model_"+str(self.epoch)+".pkl"
                with open(filename, 'wb') as f:
                    self.model.update_model()
                    pickle.dump(self.model.model, f)
            # if self.plot_fig:

            #     self.plot_reward = np.append(self.plot_reward,self.evaluate_reward)
            #     self.graph.set_xdata(self.plot_epoch)
            #     self.graph.set_xdata(self.plot_reward)
            #     plt.xlim(0,self.plot_epoch[-1])
        self.evaluate_reward += self.get_reward()
        self.evaluate_counter += 1

    def learn(self,s):
        self.learner_state()
        self.push_to_current_state(s)
        # if self.ls == lstate.fill_current_state:
        #     return self.model_action(self.current_state,self.epsilon)
        # elif self.ls == lstate.fill_memory_pool:
        #     self.memory_pool[self.memory_pool_size-self.tot_frame_counter,:] = self.current_state
        #     return self.model_action(self.current_state,self.epsilon)
        # elif self.ls == lstate.train:
        #     self.train()
        #     return self.model_action(self.current_state,self.epsilon)
        # elif self.ls == lstate.evaluate:
        #     self.evaluate()
        return self.model_action(self.current_state,0.0)
#        print(self.tot_frame_counter)
#        print(self.memory_pool.reshape([self.memory_pool_size,self.state_length]))
#        print(self.current_state.reshape([self.state_frame,self.dlen]))
