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

STEPS = 1500
OBSERVATION_SPACE_VALUES = 6

deg2rad = math.pi/180


env = gym.make('ContainerEnv-v0')
model = load_model('models/FixedYeD-ContainerPF-Eps02_015_01-Epochs3-Eps2000_2000_2000-Steps1500-linear-CNN_400_300-YEMAX500-LR0001-Outputs23-UpdateTarget1-Triangle10-MB256_____8.33max____0.20avg___-1.00min__1576269992.model')

def get_qs(state):
    return model.predict(np.array(state).reshape(1,1,OBSERVATION_SPACE_VALUES))[0]

def evaluate():

    psi_array = [math.pi/2, -math.pi/5, 0, math.pi/6, 0]
    dist_array = [-50, 75, -100, 200, 0]
    psi_c_array = [-10, 20, -40, 90, 0]
    for i in range(len(psi_array)):
        psi_init = psi_array[i]
        y_init = dist_array[i]
        psi_c_init = psi_c_array[i]*deg2rad

        total_reward = 0
        step = 1
        done = False

        curr_state = env.init_eval(y_init,psi_init,psi_c_init)

        while not done and step < STEPS:
            action = np.argmax(get_qs(curr_state))
            new_state, reward, done, _ = env.step(action)
            curr_state = new_state

            total_reward += reward
            step += 1
        print(total_reward)
        env.render()


evaluate()
