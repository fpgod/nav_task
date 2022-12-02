
import math

import torch

from utilities import normalizeToRange
from controller import Supervisor
import numpy as np
import os
from models.networks import DDPG
from sac_rl import *
from gym import spaces

import wandb

#定义robot：self.supervisor.getFrameDef('$robot_name')
#定义robot的emitter和receiver：self.supervisor.getDevice('$emitter_name')
# self.robot[i].getPosition() :获取第i个robot的x，y，z坐标
# self.robot[i].getVelocity():获取第i个robot的x，y，y方向的速度


class EpuckSupervisor:
    def __init__(self, num_robots=5):
        super().__init__()
        self.num_robots = num_robots
        self.num_walls = 7
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.steps = 0
        self.steps_limit = 1000
        self.communication = self.initialize_comms()
        self.action_space = spaces.Box(low = np.array([-1,-1]),high=np.array([1,1]),dtype=np.float32)
        self.observation_space = spaces.Box(low = -np.ones(11),high = np.ones(11),dtype=np.float64)
        self.obs_history = [ ]
        self.robot = [self.supervisor.getFromDef("e-puck" + str(i)) for i in range(self.num_robots)]
        self.wall = [self.supervisor.getFromDef("wall" + str(i)) for i in range(self.num_walls)]
        self.messageReceived = None
        self.episode_score = 0
        self.episode_score_list = []
        self.is_solved = False
        self.targetx = 0
        self.targety = 0.9
        self.evaluate_reward_history = []


    def is_done(self):
        if self.steps >= self.steps_limit:
            return True
        else:
            return False

    def initialize_comms(self):
        communication = []
        for i in range(self.num_robots):
            emitter = self.supervisor.getDevice(f'emitter{i}')
            receiver = self.supervisor.getDevice(f'receiver{i}')

            emitter.setChannel(i)
            receiver.setChannel(i)

            receiver.enable(self.timestep)

            communication.append({
                'emitter': emitter,
                'receiver': receiver,
            })

        return communication


    def step(self, action):
        if self.supervisor.step(self.timestep) == -1:
            exit()
        self.steps +=1
        self.handle_emitter(action)

        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info()
        )

    def handle_emitter(self, actions):
        for i, action in enumerate(actions):
            message = (",".join(map(str, action))).encode("utf-8")
            self.communication[i]['emitter'].send(message)

    def handle_receiver(self):# 获取某个通道上发布的信息，存入messages中。
        messages = []
        for com in self.communication:
            receiver = com['receiver']
            if receiver.getQueueLength() > 0:
                messages.append(receiver.getData().decode("utf8"))
                receiver.nextPacket()
            else:
                messages.append(None)
        return messages

    def get_observations(self):
        positions_x = np.array([normalizeToRange(self.robot[i].getPosition()[0], -0.97, 0.97, -1.0, 1.0)
                                for i in range(self.num_robots)])
        # 限制位置到-0.97
        positions_y = np.array([normalizeToRange(self.robot[i].getPosition()[1], -0.97, 0.97, -1.0, 1.0)
                                for i in range(self.num_robots)])


        self.messageReceived = self.handle_receiver()

        ds_value = np.empty([self.num_robots, 8], float)


        for i, message in enumerate(self.messageReceived):
            if message is not None:
                message = message.split(',')
                ds_value[i] = [normalizeToRange(message[j], 0, 1023, -1.0, 1.0, clip=True) for j in range(8)]
                self.ds = ds_value
            else:
                ds_value = np.zeros((self.num_robots, 8), float)

        self.observations = np.empty((self.num_robots, self.observation_space.shape[0]), float)

        for i in range(self.num_robots):
            for k in range(8):
                if ds_value[i][k] >= 0.6:
                    ds_value[i][k] = 1
                else:
                    ds_value[i][k] = -1
            self.observations[i] = np.append(ds_value[i],[self.targetx-positions_x[i],
                                             self.targety- positions_y[i], self.robot[i].getField("rotation").getSFRotation()[3]%(2*np.pi)
                                             if self.robot[i].getField("rotation").getSFRotation()[2] >0 else (-self.robot[i].getField("rotation").getSFRotation()[3])%(2*np.pi),
                                              ])
        self.obs_history.append(self.observations)
        return self.observations




    def get_reward(self, action=None):

        rewards = np.empty((self.num_robots,1),float)
        for i in range(self.num_robots):
            ds_current = np.sqrt(self.observations[i][8]*self.observations[i][8]+self.observations[i][9]*self.observations[i][9])
            ds_pre = np.sqrt(self.obs_history[self.steps-2][i][8]*self.obs_history[self.steps-2][i][8]+self.obs_history[self.steps-2][i][9]*self.obs_history[self.steps-2][i][9])
            rewards[i] = ds_pre - ds_current
            for k in range(8):
                rewards[i] = rewards[i]-0.01*self.observations[i][k]
            if(ds_current<0.08):
                rewards[i] += 1
        return rewards



    def get_default_observation(self):
        observation = []
        for _ in range(self.num_robots):
            robot_obs = [0.0 for _ in range(self.observation_space.shape[0])]
            observation.append(robot_obs)
        return observation

    def get_info(self):
        pass



    def reset(self):
        self.steps = 0
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()
        self.supervisor.step(int(self.supervisor.getBasicTimeStep()))
        px = 0.0016 * (np.random.choice(500, 5, replace=False) - 500)
        py = 0.0016 * (np.random.choice(500, 5, replace=False) - 500)
        for i in range(self.num_robots):
            self.robot[i].getField('translation').setSFVec3f([px[i],py[i],0])
            self.robot[i].getField('rotation').setSFRotation([0.0,0.0,1.0,0.0])
        self.wall[0].getField('translation').setSFVec3f([np.random.choice([0.58,0.68,0.48]),-0.2,0])
        self.wall[1].getField('translation').setSFVec3f([np.random.choice([0.57,0.67,0.48]),0.49,0])
        self.wall[2].getField('translation').setSFVec3f([np.random.choice([0.58,0.68,0.48]),0.157,0])
        self.wall[3].getField('translation').setSFVec3f([np.random.choice([-0.58,-0.68,-0.48]),0.05,0])
        self.wall[4].getField('translation').setSFVec3f([np.random.choice([0.255,0.355,0.455]),-0.69,0])
        self.wall[5].getField('translation').setSFVec3f([np.random.choice([-0.59,-0.69,-0.48]),0.507,0])
        self.wall[6].getField('translation').setSFVec3f([np.random.choice([0.02,0.12,0.22]),0.25,0])

        self.supervisor.step(int(self.supervisor.getBasicTimeStep()))
        for i in range(self.num_robots):
            self.communication[i]['receiver'].disable()
            self.communication[i]['receiver'].enable(self.timestep)
            receiver = self.communication[i]['receiver']
            while receiver.getQueueLength() > 0:
                receiver.nextPacket()

        return self.get_default_observation()

def create_path(path):
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)



if __name__ =='__main__':
    create_path("./models/saved/sac_all_map/")
    create_path("./exports_sac_all_map/")
    env_name= 'juji'
    env=EpuckSupervisor()
    env_evaluate = env
    number = 1
    seed = 0
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_episode_steps = env.steps_limit  # Maximum number of steps per episode
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))

    agent = SAC(state_dim, action_dim, max_action)

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    max_train_steps = 3e6  # Maximum number of training steps  3e6
    random_steps = 0#25e3  # Take the random actions in the beginning for the better exploration 25e3
    evaluate_freq = 1#5000  # Evaluate the policy every 'evaluate_freq' steps 5000
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training


    while total_steps < max_train_steps:
        s = env.reset()
        episode_steps = 0
        done = False
        a = np.empty((env.num_robots, 2), float)
        while not done:
            episode_steps += 1
            for n in range(env.num_robots):
                if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                    a[n] = env.action_space.sample()
                else:
                    a[n]= agent.choose_action(s[n])
            s_, r, done, _ = env.step(a)
            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False
            for j in range(env.num_robots):
                replay_buffer.store(s[j], a[j], r[j], s_[j], dw)  # Store the transition
            s = s_

            if total_steps >= random_steps:
                agent.learn(replay_buffer)

            if (total_steps + 1) % evaluate_freq == 0:
                print(total_steps)
                print('eva')
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
                if evaluate_num % 10 == 0:
                    np.save('./data_train/SAC_env_{}_number_{}.npy'.format(env_name, number), np.array(evaluate_rewards))
            total_steps += 1




