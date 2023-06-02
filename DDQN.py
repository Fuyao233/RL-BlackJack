import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

class DDQN(nn.Module):
    def __init__(self, input_size, action_num):
        # input is a binary group (s_0, a_0) size: 4 
        super().__init__()
        self.input_size = input_size
        self.action_num = action_num
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Linear(32,32),
            nn.Linear(32, action_num)
        )
        
        self.target_model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Linear(32,32),
            nn.Linear(32, action_num)
        )
        
        print(2)
        # synchronize the two networks
        # for para, target_para in zip(self.model.parameters(), self.target_model.parameters()):
        #     target_para.data = para.data.clone()
            # print(type(para))
        
        
    def forward(self, state, target_flag=False):
        x = torch.Tensor(state)
        # x.cuda()
        if target_flag:
            # target model
            return self.target_model(x)
        else:
            # DQN
            return self.model(x)
    
    def epsilon_policy(self, state, epsilon):
        
        # epsilon = epsilon + 0.88 ** 
        
        if random.uniform(0,1) < epsilon:
            return random.randint(0,self.action_num - 1)
        else:
            x = torch.Tensor(state)
            # x.cuda()
            action_value = self.model(x)
            return action_value.argmax().item()
    
    def update_target(self, tau):
        for para, target_para in zip(self.model.parameters(), self.target_model.parameters()):
            target_para.data = tau * para.data.clone() + (1-tau) * target_para.data.clone() 
    
    # def predicate(self, state):
    #     output = self.model(state)
    #     return output.argmax().item(), output    

class Trainer:
    def __init__(self, env, model, 
                 step_per_epoch = 750, 
                 updata_num_per_step = 750,
                 epoch=20, 
                 batch_size=64, 
                 lr=1e-3, 
                 initialization_num=100000, 
                 epsilon = 0.1, 
                 buffer_capacity=100000,
                 update_target=0.1,
                 ) -> None:
        self.buffer = [] # store training samples
        self.env = env # 训练时玩家采用带DQN的ε-greedy策略，庄家采用简单的17策略 TODO:比较庄家和玩家策略相同时的结果
        # 思考：其实这里庄家的策略是我们对对手策略的估计
        # self.buffer_capacity = buffer_capacity if buffer_capacity > 100000 else 100000
        self.buffer_capacity = buffer_capacity
        self.new_sample_idx = -1 # record the place to add the new sample
        self.initialization_num = initialization_num if initialization_num < buffer_capacity else buffer_capacity
        self.model = model
        self.epsilon = epsilon
        self.step_per_epoch = step_per_epoch
        self.epoch = epoch
        self.batch_size = batch_size
        self.update_num_per_step = updata_num_per_step
        self.optimizor = torch.optim.Adam(self.model.model.parameters(), lr=lr)   
        self.update_target = update_target
            
    def test(self):
        pass
    
    
    def train(self):
        # initialize the buffer
        
        # TODO:采取PER策略
        while len(self.buffer) <= self.initialization_num:
            print("\r", f'content:[{len(self.buffer)}/{self.initialization_num}]',end="",flush=True)
            s_0 = self.env.reset()
            a_0 = self.model.epsilon_policy(s_0, self.epsilon)
            s_1, r_0, is_done, _ = self.env.step(a_0)
            if is_done == False:
                r_0 = 0.5 # 鼓励多拿，TODO: 消融实验
                
            self.buffer.append([s_0, a_0, r_0, s_1])
            self.new_sample_idx = self.new_sample_idx + 1
            
            while is_done == False:
                s_0 = s_1
                a_0 = self.model.epsilon_policy(s_0, self.epsilon)
                s_1, r_0, is_done, _ = self.env.step(a_0)
                
                if is_done == False:
                    r_0 = 0.5 # 鼓励多拿，TODO: 消融实验
                    
                self.buffer.append([s_0, a_0, r_0, s_1])
                self.new_sample_idx = self.new_sample_idx + 1
        
        
        for epo_idx in range(self.epoch):
            print(f'epoch: [{epo_idx}/{self.epoch}]')
            loss_list = []
            for step in range(self.step_per_epoch):
                
                idx_list = [random.randint(0, len(self.buffer)-1) for _ in range(self.batch_size)]
                train_batch = [self.buffer[idx] for idx in idx_list]
                
                # calculate loss and update parameters
                s0_batch = [sample[0] for sample in train_batch]
                a0_batch = np.array([sample[1] for sample in train_batch])
                r0_batch = [sample[2] for sample in train_batch]
                s1_batch = [sample[3] for sample in train_batch]
                
                output = self.model(s0_batch, False)
                a0_batch = torch.from_numpy(a0_batch).long()
                q0_batch = output[torch.arange(output.size(0)), a0_batch]
                
                # with torch.no_grad():
                #     # do not update the target model here
                output_target = self.model(s1_batch, True)
                q1_batch = output_target.max(dim=1)[0]
            
                TD_error = q0_batch - (q1_batch + torch.Tensor(r0_batch))
                
                loss = 0.5 * (TD_error * TD_error).mean()
                
                if step % 100 == 0:
                    loss_list.append(loss)
                
                self.optimizor.zero_grad()
                loss.backward()
                
                self.optimizor.step()
                
                self.model.update_target(self.update_target)
                print("\r", f'step:[{step}/{self.step_per_epoch}], loss:{loss}',end="",flush=True)
                
                # update samples in the buffer
                for _ in range(self.update_num_per_step):
                    s_0 = self.env.reset()
                    a_0 = self.model.epsilon_policy(s_0, self.epsilon)
                    s_1, r_0, is_done, _ = self.env.step(a_0)
                    if is_done == False:
                        r_0 = 0.5 # 鼓励多拿，TODO: 消融实验
                    
                    if len(self.buffer) < self.buffer_capacity:
                        self.buffer.append([s_0, a_0, r_0, s_1])
                        self.new_sample_idx = self.new_sample_idx + 1
                    else:
                        self.buffer[self.new_sample_idx] = [s_0, a_0, r_0, s_1]
                        self.new_sample_idx = (self.new_sample_idx + 1) % self.buffer_capacity
                    
                    while is_done == False:
                        s_0 = s_1
                        a_0 = self.model.epsilon_policy(s_0, self.epsilon)
                        s_1, r_0, is_done, _ = self.env.step(a_0)
                        
                        if is_done == False:
                            r_0 = 0.5 # 鼓励多拿，TODO: 消融实验
                            
                        if len(self.buffer) <= self.buffer_capacity:
                            self.buffer.append([s_0, a_0, r_0, s_1])
                            self.new_sample_idx = self.new_sample_idx + 1
                        else:
                            self.buffer[self.new_sample_idx] = [s_0, a_0, r_0, s_1]
                            self.new_sample_idx = (self.new_sample_idx + 1) % self.buffer_capacity
                
                
                # break
            
            plt.figure()
            plt.plot(np.arange(len(loss_list)), loss_list)
            plt.savefig('./loss.jpg')

            
            
            if epo_idx % 10 == 0:
                torch.save(self.model.state_dict(), f'./checkpoint/{epo_idx}.pth')
            
        
                 
class DDQN_Policy:
    def __init__(self, model) -> None:
        self.model = model
        
    def act(self, state):
        predication = self.model(state)
        return predication.argmax().item()