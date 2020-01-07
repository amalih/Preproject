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

STEPS = 500
OBSERVATION_SPACE_VALUES = 6

deg2rad = math.pi/180


env = gym.make('ContainerEnv-v0')
model = load_model('models/1O_Nomoto-Eps01-Epochs1-Eps4000-Steps1000-linear-CNN_300_200-YEMAX1000-LR0001-Inputs6-Outputs21-UpdateTarget1-Triangle10_Pi4-MB64___965.74max__456.27avg___-1.00min__1578265051.model')

def get_qs(state):
    return model.predict(np.array(state).reshape(1,1,OBSERVATION_SPACE_VALUES))[0]

def evaluate():

    total_reward = 0
    y_init = 0
    psi_init = 0
    #psi_array = [0, -math.pi/5, 0, math.pi/6, 0]
    psi_c_array = [0, 90, -45,45]
    psi_y_array = [0, 1000, 0,1000]
    psi_x_array = [0,1000,2000,3000]
    #psi_c_array = [0,0,0,0,0]
    #dist_array = [-50, 75, -100, 150, 0]
    curr_state = env.init_eval(y_init,psi_init,psi_c_array[0]*deg2rad)


    for i in range(len(psi_c_array)):
        psi_c = psi_c_array[i]*deg2rad
        psi_x = psi_x_array[i]
        psi_y = psi_y_array[i]

        step = 1
        done = False



        while not done and step < STEPS:
            action = np.argmax(get_qs(curr_state))
            new_state, reward, done, _ = env.step(action, psi_c)
            curr_state = new_state

            total_reward += reward
            step += 1
        print(total_reward)
    env.render()


evaluate()
