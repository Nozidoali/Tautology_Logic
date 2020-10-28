import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import re
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
from subprocess import check_output
import numpy as np
import random

class env():
    def __init__(self):
        self.state = 0

    def run(self, action):
        reward = 0
        done=False
        if action== 0:
            self.state += 7
        if action== 1:
            self.state += 11
        if self.state in [95,97,99]:
            reward=0.5
        if self.state == 100:
            reward = 1
        if self.state>=100:
            done=True
        return self.state, reward, done

    def reset(self):
        self.state=0
        return 0

class abc_env:
    def __init__(self, filename, abc_path):
        self.filename = filename
        self.abc_path = abc_path
        self.state = self.delay = self.area = 0
        self.step = 0
        self.max_step = 100
    
    def run(self, action):
        action_command_list = [
            'balance',
            'rewrite',
            'refactor',
            'resub'
        ]
        abc_commands = [
            'read_blif step{}.blif'.format(self.step),
            'strash',
            action_command_list[action],
            'print_stats',
            'write_blif step{}.blif'.format(self.step+1)
        ]
        proc = check_output([
            self.abc_path,
            '-c',
            ';'.join(abc_commands)
        ])
        self.step += 1
        delay, area = self.parse_result(proc)
        
        reward = self.area - area if self.delay >= delay else -100
        
        self.delay, self.area = delay, area
        self.state = self.step, self.delay, self.area

        return self.state, reward, (self.step>self.max_step and reward==0)
    
    def reset(self):
        self.step = 0

        abc_commands = [
            'read_blif {}'.format(self.filename),
            'strash',
            'print_stats',
            'write_blif step{}.blif'.format(self.step)
        ]
        proc = check_output([
            self.abc_path,
            '-c',
            ';'.join(abc_commands)
        ])
        self.delay, self.area = self.parse_result(proc)
        self.state = self.step, self.delay, self.area
        return self.state

    def parse_result(self, proc):
        line = proc.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
        ob = re.search(r'lev *= *[0-9]+.?[0-9]*', line)
        delay = int(ob.group().split('=')[1].strip())
        
        ob = re.search(r'and *= *[0-9]+.?[0-9]*', line)
        area = int(ob.group().split('=')[1].strip())

        return delay, area

class DQNagent:
    def __init__(self,env,state_size,action_size):
        self.env=env
        self.state_size=state_size
        self.action_size=action_size
        self.memory=deque(maxlen=100)
        self.DQNmodel=self.build_model(self.state_size,self.action_size)

        self.gamma = 0.95    # discount rate
        self.epsilon_max = 1.0  # exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.001
        self.epsilon=1
        self.n_update=4
        self.batch_size = 16
        self.train_start = 128
        self.train_episode=2000
        self.test_epsiode=20
        
    def build_model(self,state_size, action_size):
        X_input = Input(state_size) 

        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(1) and Hidden Layer with 64 nodes
        X = Dense(64, input_shape=(state_size,), activation="relu", kernel_initializer='he_uniform')(X_input)

        # Hidden layer with 64 nodes
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(action_size, activation="linear", kernel_initializer='he_uniform')(X)

        model = Model(inputs = X_input, outputs = X)
        model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.DQNmodel.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.DQNmodel.predict(state)
        target_next = self.DQNmodel.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.DQNmodel.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.DQNmodel = load_model(name)

    def save(self, name):
        self.DQNmodel.save(name)

    def train(self):
        total_timestep=0
        score_history=[]
        for e in range(self.train_episode):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            reward_this_episode=0
            while not done:
                total_timestep+=1
                i += 1
                self.epsilon=self.epsilon_min+(self.epsilon_max-self.epsilon_min)*np.exp(-self.epsilon_decay*total_timestep)

                action = self.act(state)
                next_state,reward,done=self.env.run(action)
                print("episode_index {} timestep {} agent_state {}  action {} reward {} done {} next_agent_state {} episilon {}".format(e+1,i,state,action,reward,done,next_state,self.epsilon))
                reward_this_episode+=reward
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if i%self.n_update==0:
                    self.replay()
                if done:                   
                    score_history.append(reward_this_episode)
                    print("episode: {}/{}, score: {:.2f}, avg: {:.2f} e: {:.2f}".format(e, self.train_episode, reward_this_episode,np.mean(score_history[-30:]), self.epsilon))
                    
        print("Saving trained model as staircase-dqn.h5")
        self.save("staircase-dqn.h5")
    
    def test(self):
        self.load("staircase-dqn.h5")
        for e in range(self.test_epsiode):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                action = np.argmax(self.DQNmodel.predict(state))
                next_state,reward,done=self.env.run(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.test_epsiode, i))

if __name__=="__main__":
    env = abc_env(filename='step0.blif', abc_path='./abc')
    agent=DQNagent(env=env,state_size=3,action_size=4)
    agent.train()
    agent.test()

"""
q_table=np.zeros((11,2))

replay_buffer = []

num_episode=2000
epsilon_max=1
epsilon_min=0.001
episode_decay=0.001
total_timestep=0
total_reward_history=[]
env = env()
discount=0.95
for i in range(num_episode):
    agent_state=env.reset()
    done=False
    timestep=0
    total_reward=0
    while not done:
        timestep+=1
        total_timestep+=1
        epsilon=epsilon_min+(epsilon_max-epsilon_min)*np.exp(-total_timestep*episode_decay)
        random_number=random.random()
        if random_number<epsilon:
            action=random.randint(0,1)
        else:
            action=np.argmax(q_table[agent_state])
        next_agent_state,reward,done=env.run(action)
        total_reward+=reward
        print("episode_index {} timestep {} agent_state {}  action {} reward {} done {} next_agent_state {} episilon {}".format(i+1,timestep,agent_state,action,reward,done,next_agent_state,epsilon))
        q_table[agent_state][action]=reward+discount*(np.max(q_table[next_agent_state])) if not done else reward
        agent_state=next_agent_state
        if done:
            total_reward_history.append(total_reward)
            print("episode_index {} total reward {} recent reward {:.2f}".format(i+1,total_reward,np.mean(total_reward_history[-100:])))    
"""
            