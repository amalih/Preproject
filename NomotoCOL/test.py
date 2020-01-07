import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
import container_env
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from keras.layers import Dense, Dropout, MaxPooling1D, Activation, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import math

STEPS = 1000
OBSERVATION_SPACE_VALUES = 10

deg2rad = math.pi/180


env = gym.make('ContainerEnv-v0')
model = load_model('models/COLAV200_Nomoto-Eps03_01-Epochs2-Eps1000_3000-Steps1000-linear-CNN_300_200-YEMAX1000-LR0001-Inputs10-Outputs21-UpdateTarget1-Triangle10_Pi4-MB64___113.27max___25.18avg__-60.44min__1578389122.model')

def get_qs(state):
    return model.predict(np.array(state).reshape(1,1,OBSERVATION_SPACE_VALUES))[0]

def evaluate():


    enemy_array = [1000,2000,3000]

    for i in range(len(enemy_array)):

        total_reward = 0
        step = 1
        done = False
        #env.reset()
        curr_state = env.init_eval(enemy_array[i])

        while not done and step < STEPS:
            action = np.argmax(get_qs(curr_state))
            new_state, reward, done, _ = env.step(action)
            curr_state = new_state

            total_reward += reward
            step += 1
        print(total_reward)
        env.render()


evaluate()
