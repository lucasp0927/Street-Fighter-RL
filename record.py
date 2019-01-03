import os
import sys
import asyncore
import socket
import time
import random
import numpy
import h5py
import pickle
import timeit
from learner import Learner

class MsgHandler(asyncore.dispatcher_with_send):
    def __init__(self,sock):
        asyncore.dispatcher_with_send.__init__(self,sock)
        self.record_time = 1 #in second
        self.frame_per_second = 15
        self.buffer_size = self.record_time*self.frame_per_second
        self.episode_size = 55
        self.state_buffer_size = 10000
        self.state_buffer = numpy.zeros((self.state_buffer_size,self.episode_size*self.frame_per_second))
        self.state_buffer_full = False
        self.buffer = numpy.zeros((self.buffer_size,self.episode_size))
        self.current_frame = 0
        self.total_frame_count = 0
        self.batch_num = 0
        self.epsilon = 1.0
        self.model = None
        self.accu_q = 0.0
        # self.ll = Learner(66,10,0.0002,memory_pool_size = 100000,exp_epoch=10000,epsilon_rate = 0.0001,momentum=0.9,learner_type="DDQN",n_hidden1 = 750, n_hidden2 = 750,freeze_interval=50,train_interval=100)
#        self.ll = Learner(66,10,0.0002,memory_pool_size = 100000,exp_epoch=10000,epsilon_rate = 0.0001,momentum=0.80,learner_type="DDQN",n_hidden1 = 800, n_hidden2 = 700,freeze_interval=70,train_interval=100)
#        self.ll.model.load_model('model_198.pkl')
        self.record = []
        self.record_count = 0
        # print("load memory pool")
        # memory_pool = pickle.load(open('memory_pool2.pkl','rb'))
        # memory_pool = memory_pool[0:100000]
        # for episode in memory_pool:
        #     self.ll.learn(episode)
        # print("done.")

    def action_idx_to_pair(self,idx):
        #return [dir, attack]
        idir = idx//7
        iatk = idx%7
        assert idir*7+iatk == idx
        return [idir,iatk]

    # def model_action(self,s):
    #     assert s.shape[0] == 1
    #     hl1 = numpy.maximum(0,numpy.dot(s,self.model[0])+self.model[2])
    #     hl2 = numpy.maximum(0,numpy.dot(hl1,self.model[1])+self.model[3])
    #     q = numpy.dot(hl2,self.model[4]) + self.model[5]
    #     self.accu_q += numpy.max(q)
    #     action_idx = numpy.argmax(q)
    #     return self.action_idx_to_pair(action_idx)

    def IOMsgGenerator(self,s):
 #       print("send action " + str(s))
        IOlist = self.action_idx_to_pair(s)
        IOMsg = ''.join(str(x) for x in IOlist)+"\n"
        return IOMsg
    def norm_x(self,x):
#        return x
        return (x-235.0)/175.0

    def norm_y(self,y):
#        return y
        return (y-130.0)/86.0

    def norm_dx(self,x):
        return x/(175.0*2.0)

    def norm_dy(self,y):
        return y/(86.0*2.0)

    def calc_episode(self,rev_episode):
        episode = numpy.zeros(self.episode_size)
        #reward
        episode[0] = rev_episode[0]/5.0
        #p1 x y
        p1x = numpy.trim_zeros(rev_episode[1:23:3])
        p1y = numpy.trim_zeros(rev_episode[2:24:3])
        p1x_avg = numpy.mean(p1x)
        p1y_avg = numpy.mean(p1y)
        #p2 x y
        p2x = numpy.trim_zeros(rev_episode[31:53:3])
        p2y = numpy.trim_zeros(rev_episode[32:54:3])
        p2x_avg = numpy.mean(p2x)
        p2y_avg = numpy.mean(p2y)
        #p1 delta x,y
        episode[1:1+len(p1x)] = self.norm_dx(p1x-p1x_avg)
        episode[9:9+len(p1y)] = self.norm_dy(p1y-p1y_avg)
        episode[17:17+len(p2x)] = self.norm_dx(p2x-p2x_avg)
        episode[25:25+len(p2y)] = self.norm_dy(p2y-p2y_avg)
        #delta xy_avg
        episode[35] = self.norm_dx(p1x_avg-p2x_avg)
        episode[36] = self.norm_dy(p1y_avg-p2y_avg)
        #p1 xavg yavg
        episode[37] = self.norm_x(p1x_avg)
        episode[38] = self.norm_y(p1y_avg)
        #projectile
        if rev_episode[61] == 0.0:
            episode[33] = 0.0
        else:
            episode[33] = self.norm_dx(rev_episode[61]-p1x_avg)
        if rev_episode[62] == 0.0:
            episode[34] = 0.0
        else:
            episode[34] = self.norm_dx(rev_episode[62]-p1y_avg)
        #action
        episode[39:55] = -1.0
        episode[int(rev_episode[64])+39] = 1.0
        episode[int(rev_episode[65])+48] = 1.0
        return episode

    def handle_read(self):
        data = self.recv(8192)
        if data:
            rev_episode = numpy.array(list(map(float,str.split(data.decode()))))
            assert rev_episode.size == 66
            episode = self.calc_episode(rev_episode)
            self.record.append(episode)
            self.record_count += 1
            if self.record_count % 1000 == 0:
                print(self.record_count)
                with open('human_play.pkl', 'wb') as f:
                    pickle.dump(self.record, f)
            # print(self.ll.get_train_counter())
            # print(self.ll.ls)
            # if self.ll.get_train_counter() == 0 and self.ll.is_training() == True:
            #     self.send(bytes("pause\n","ascii"))
            #     self.ll.learn(episode)
            #     self.record_count += 1
            #     if self.record_count >= 60:
            #         self.record_count = 0
            #         with open('memory_pool.pkl', 'wb') as f:
            #             pickle.dump(self.record, f)
            #     ####save model
            #     if self.ll.model.saved == False:
            #         filename = "model_"+str(self.ll.model.to_save_id)+".pkl"
            #         with open(filename, 'wb') as f:
            #             pickle.dump(self.ll.model.model_to_save, f)
            #             self.ll.model.saved = True
            #     self.send(bytes("unpause\n","ascii"))
            # else:
            #     action = self.ll.learn(episode)
            #     outputmsg = self.IOMsgGenerator(action)
            #     self.send(bytes(outputmsg,"ascii"))
            sys.stdout.flush()
class MsgServer(asyncore.dispatcher):
    def __init__(self, host, port):
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind((host, port))
        self.listen(5)

    def handle_accept(self):
        pair = self.accept()
        if pair is not None:
            sock, addr = pair
            print('Incoming connection from %s' % repr(addr))
            handler = MsgHandler(sock)

if __name__ == "__main__":
    server = MsgServer('0.0.0.0', 54321)
    asyncore.loop()
