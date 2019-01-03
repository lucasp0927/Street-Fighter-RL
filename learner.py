import numpy as np
import pickle
import random
from mlp import Model, Model_duel, Model_3
from enum import Enum
import pylab
from rank_based import Experience

class lstate(Enum):
    fill_current_state = 1
    fill_memory_pool = 2
    train = 3
    evaluate = 4
    play = 5

class Learner(object):
    def __init__(self, dlen, state_frame = 20, learning_rate = 0.0005, memory_pool_size = 10000,
                minibatch_size = 32, train_interval = 100, freeze_interval = 20, exp_epoch = 2000,
                epsilon_rate = 0.0001, evaluate_interval = 66, save_threshold = 100,
                momentum = 0.0, learner_type = "DDQN", plot_fig = True,n_hidden1 = 50,
                n_hidden2 = 50, n_hidden3 = 50, prioritize = False):
        np.random.seed()
        # hyper-parameters
        self.train_interval = train_interval
        self.dlen = dlen
        self.state_frame = state_frame
        self.state_length = dlen*state_frame
        self.memory_pool_size = memory_pool_size
        self.freeze_interval = freeze_interval
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.momentum = momentum
        self.learner_type = learner_type
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        # variables
        self.tot_frame_counter = -1*state_frame
        self.train_counter = 0
        self.global_train_counter = 0
        self.epoch = 0
        self.current_state = np.zeros([1,self.state_length])
        self.last_state = np.copy(self.current_state)
        self.memory_pool = np.zeros([memory_pool_size,self.state_length])
    ### prioritize
        self.learn_start = 0
        self.prioritize = prioritize
        conf = {'size': memory_pool_size,
                'learn_start': self.learn_start,
                'steps':45000,
                'batch_size': minibatch_size}
        self.experience = Experience(conf)
    ### 3-layered
#        self.model = Model_3(None,self.state_length,gamma = 0.995,n_hidden1 = n_hidden1,
#               n_hidden2 = n_hidden2,n_hidden3 = n_hidden3,learning_rate = learning_rate,
#               freeze_interval = freeze_interval,momentum = momentum,learner_type = learner_type,
#               minibatch_size = minibatch_size,train_interval = train_interval)
        self.model = Model(None,self.state_length, gamma = 0.995, n_hidden1 = n_hidden1, n_hidden2 = n_hidden2,
                  learning_rate = learning_rate, freeze_interval = freeze_interval, momentum = momentum,
                  learner_type = learner_type, minibatch_size = minibatch_size, train_interval = train_interval)
    ### dueling network
#        self.model = Model_duel(None,self.state_length,gamma=0.99,n_hidden1=n_hidden1,n_hidden2=n_hidden2,
#               learning_rate = learning_rate,freeze_interval=freeze_interval,momentum = momentum,
#               minibatch_size = minibatch_size,train_interval=train_interval)
        self.ls = lstate.fill_current_state
        #### greedy exploration
        self.epsilon = 1.0
        self.final_epsilon = 0.05
        self.exploration_epoch = exp_epoch
        self.epsilon_rate = epsilon_rate
        #### evaluation
        self.evaluate_interval = evaluate_interval
        self.evaluate_reward = 0.0
        self.evaluate_counter = 0
        self.evaluate_frame = 1200
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
        print("-------------Hyper parameters-------------")
        print("frame per state: " + str(self.state_frame))
        print("learning rate: " + str(self.learning_rate))
        print("memory pool size: " + str(self.memory_pool_size))
        print("minibatch size: "+str(self.minibatch_size))
        print("train interval: " + str(self.train_interval)+" frame")
        print("freeze interval: " + str(self.freeze_interval)+ " epoch")
        print("exp epoch: " + str(self.exploration_epoch)+ " epoch")
        print("epsilon rate: " + str(self.epsilon_rate)+" /epoch")
        print("momentum: " + str(self.momentum))
        print("n_hidden1: " + str(self.n_hidden1))
        print("n_hidden2: " + str(self.n_hidden2))
        print("------------------------------------------")
    def model_action(self,s,epsilon):
        if random.random() < epsilon:
            action_idx = random.randint(0,62)
        else:
            assert s.shape[0] == 1
            action_idx = self.model.action(s)
        assert action_idx >=0 and action_idx <=62
        return action_idx

    def last_action(self,s):
        return self.model.report_action(s)

    def load_model(self):
        return pickle.load(open('best_model.pkl','rb'))

    def push_to_current_state(self,s):
        assert s.size == self.dlen
        self.current_state = np.roll(self.current_state,self.dlen,axis=1)
        self.current_state[0,0:self.dlen] = s

    def learner_state(self):
        if self.ls != lstate.play:
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

    def set_to_play_mode(self):
        self.ls = lstate.play

    def update_epsilon(self):
        if self.epoch >= self.exploration_epoch and self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_rate

    def get_train_counter(self):
        return self.train_counter

    def is_training(self):
        return self.ls == lstate.train

    def train(self):
        if self.prioritize:
            if self.train_counter == 0:
                self.global_train_counter += 1
                self.update_epsilon()
                self.epoch += 1
                #print("epoch: "+ str(self.epoch))
                self.model.train_prioritize(self.experience,self.global_train_counter,self.minibatch_size,n_batch=self.train_interval)
            self.train_counter += 1
            if self.train_counter >= 0:
                to_insert = (self.last_state,1,1,self.current_state,1)
                self.experience.store(to_insert)
            if self.train_counter == self.train_interval:
                self.train_counter = -1*self.state_frame
        else:
            if self.train_counter == -1*self.state_frame:
                self.update_epsilon()
                self.epoch += 1
                #print("epoch: "+ str(self.epoch))
                self.model.train(self.memory_pool,n_epochs=1 ,batch_size = self.minibatch_size,n_batch=self.train_interval)
                self.memory_pool = np.roll(self.memory_pool,self.train_interval,axis = 0)
            self.train_counter += 1
            if self.train_counter >= 0:
                self.memory_pool[self.train_interval - self.train_counter,:] = self.current_state
            if self.train_counter == self.train_interval:
                self.train_counter = -1*self.state_frame

    def get_reward(self):
        return self.current_state[0,0]

    def get_epoch(self):
        return self.epoch

    def evaluate(self):
        if self.evaluate_counter == 0:
            self.evaluate_reward = 0.0
            print("\a")
            print("evaluating...")
        elif self.evaluate_counter == self.evaluate_frame:
            self.plot_epoch = np.append(self.plot_epoch,self.epoch)
            self.plot_reward = np.append(self.plot_reward,self.evaluate_reward)
            print("\a")
            print("epoch:" + str(self.epoch))
            print("epsilon:" + str(self.epsilon))
            print("reward:" + str(self.evaluate_reward))
            print("max reward:" + str(np.max(self.plot_reward)))
            with open('log2.pkl', 'wb') as f:
                pickle.dump((self.plot_epoch,self.plot_reward), f)
            if self.evaluate_reward >= self.save_threshold:
                self.model.update_to_save_model(self.epoch)
                # filename = "model_"+str(self.epoch)+".pkl"
                # with open(filename, 'wb') as f:
                #     self.model.update_model()
                #     pickle.dump(self.model.model, f)
            # if self.plot_fig:

            #     self.plot_reward = np.append(self.plot_reward,self.evaluate_reward)
            #     self.graph.set_xdata(self.plot_epoch)
            #     self.graph.set_xdata(self.plot_reward)
            #     plt.xlim(0,self.plot_epoch[-1])
        self.evaluate_reward += self.get_reward()
        self.evaluate_counter += 1

    def learn(self,s):
        self.last_state = np.copy(self.current_state)
        self.push_to_current_state(s)
#        print("last action "+str(self.last_action(self.current_state)))
        if self.ls == lstate.fill_current_state:
            self.learner_state()
            return self.model_action(self.current_state,self.epsilon)
        elif self.ls == lstate.fill_memory_pool:
            if self.prioritize:
                print(self.memory_pool_size-self.tot_frame_counter)
                to_insert = (self.last_state,1,1,self.current_state,1)
                self.experience.store(to_insert)
                self.learner_state()
                return self.model_action(self.current_state,self.epsilon)
            else:
                self.memory_pool[self.memory_pool_size-self.tot_frame_counter,:] = self.current_state
                print(self.memory_pool_size-self.tot_frame_counter)
                self.learner_state()
                return self.model_action(self.current_state,self.epsilon)
        elif self.ls == lstate.train:
            self.train()
            self.learner_state()
            return self.model_action(self.current_state,self.epsilon)
        elif self.ls == lstate.evaluate:
            self.evaluate()
            self.learner_state()
            return self.model_action(self.current_state,0.01)
        elif self.ls == lstate.play:
            self.evaluate()
            self.learner_state()
            return self.model_action(self.current_state,0.01)
