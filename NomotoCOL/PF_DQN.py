
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import numpy as np
import gym
import container_env
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, MaxPooling1D, Activation, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.initializers import glorot_normal
import tensorflow as tf
from collections import deque
import time
import random
import os
#from PIL import Image


# Constants
deg2rad = math.pi/180
rad2deg = 180/math.pi
m2km = 1/1000

DISCOUNT = 0.9
REPLAY_MEMORY_SIZE = 1_000_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 256 # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64 # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 1  # Terminal states (end of episodes)
MODEL_NAME = 'COLAV_Nomoto-Eps03_02_01-Epochs1-Eps500_500_4000-Steps1500-linear-CNN_300_200-YEMAX2000-LR0001-Inputs10-Outputs21-UpdateTarget1-Triangle10_Pi4-MB64'
MIN_REWARD = 0  # For model save
OBSERVATION_SPACE_VALUES = 10
ACTION_SPACE_VALUES = 21
#MODEL_FILE = 'models/COLAV200_Nomoto-Eps08dec-Epochs2-Eps5000-Steps2000-linear-CNN_300_200-YEMAX1000-LR0001-Inputs10-Outputs21-UpdateTarget1-Triangle10_Pi4-MB64___237.68max__117.22avg__-33.97min__1578431292.model'

# Environment settings
EPISODE_START =0# 4050
#EPISODES = [2000, 2000, 2000]
#EPISODES = [3000,1000]
EPISODES = [500,500,4000]
EPOCHS = 3

MAX_CTE = 2000
# Exploration settings
EPSILON = [0.3,0.2,0.1]
#EPSILON = [0.25,0.10,0.05]  # not a constant, going to be decayed
EPSILON_DECAY = 1
MIN_EPSILON = 0.1
LEARNING_RATE = 0.001


#  Stats settings
AGGREGATE_STATS_EVERY = 25  # episodes
SHOW_PREVIEW = False
SHOW_EVERY = 1
SAVE_MODEL = True
STEPS = 1500


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQN_Agent:
    def __init__(self):

        # Main model, used for training
        self.model = self.create_model()
        #self.old_model = load_model(MODEL_FILE)
        #self.model = load_model(MODEL_FILE)

        #self.model.set_weights(self.old_model.get_weights())

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0


    def create_model(self):

        model = Sequential()

        model.add(Dense(300, input_shape=(1,OBSERVATION_SPACE_VALUES), activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dense(200, activation='relu'))
        #model.add(Dense(100, activation='relu'))
        model.add(Flatten())
        model.add(Dense(ACTION_SPACE_VALUES, activation='linear'))

        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        model.summary()

        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, done):

        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        samples = random.sample(self.replay_memory,MINIBATCH_SIZE)
        samples[-1] = self.replay_memory[-1]

        states = []
        new_states = []
        for state, action, reward, new_state, done in samples:
            #print('--------------------------------------------------')
            #print(f'Prev state: {state}')
            #print(f'Curr state: {new_state}')
            #print(f'Action: {action}')
            #print(f'Reward: {reward}')
            states.append(state)
            new_states.append(new_state)

        pred_qs = self.model.predict(np.array(states).reshape(MINIBATCH_SIZE, 1, OBSERVATION_SPACE_VALUES))
        pred_future_qs = self.target_model.predict(np.array(new_states).reshape(MINIBATCH_SIZE, 1, OBSERVATION_SPACE_VALUES))

        for index, sample in enumerate(samples):
            state, action, reward, new_state, done = sample
            pred_q = pred_qs[index]
            if done:
                pred_q[action] = reward
            else:
                future_q = max(pred_future_qs[index])
                pred_q[action] = reward + future_q * DISCOUNT
            pred_qs[index] = pred_q

        self.model.fit(np.array(states).reshape(MINIBATCH_SIZE, 1,OBSERVATION_SPACE_VALUES), pred_qs, batch_size=MINIBATCH_SIZE, epochs=1, verbose=0)


    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(1,1,OBSERVATION_SPACE_VALUES))[0]

    def get_qs_target(self, state):
        return self.target_model.predict(np.array(state).reshape(1,1,OBSERVATION_SPACE_VALUES))[0]

    def evaluate(self, ep):

        average_reward = 0
        enemy_array = [2000,3000,4000]

        for i in range(len(enemy_array)):
            ep_reward = 0
            step = 1
            done = False
            curr_state = env.init_eval(enemy_array[i])

            while not done and step < STEPS:
                action = np.argmax(agent.get_qs_target(curr_state))
                new_state, reward, done, _ = env.step(action)
                curr_state = new_state

                average_reward += reward
                ep_reward += reward
                step += 1

            print(f'Evaluative reward for enemy_init = {enemy_array[i]} was: {ep_reward}')

        return average_reward/len(enemy_array)


def run_experiment(agent):

    # For training
    ep_rewards = []
    # For validation
    val_rewards = []

    for epoch in range(EPOCHS):

        epsilon = EPSILON[epoch]

        for episode in range(1, EPISODES[epoch]+1):
            print('----------------')
            prev_eps = 0

            if epoch > 0:
                for i in range(epoch):
                    prev_eps += EPISODES[i]
            curr_ep = EPISODE_START + prev_eps + episode

            # Update tensorboard step every episode
            agent.tensorboard.step = curr_ep

            # Restarting episode - reset episode reward, step number, and flag
            episode_reward = 0
            step = 1
            done =  False

            # Reset environment and get initial state, might have to reshape
            curr_state = env.reset()
            y_init = curr_state[0]*MAX_CTE
            psi_init = curr_state[2]*180

            while not done and step < STEPS:

        

                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs_target(curr_state))
                else:
                    # Get random action
                    action = np.random.randint(0,ACTION_SPACE_VALUES)

                new_state, reward, done, _ = env.step(action)

                # Every step we update replay memory and train main network
                agent.update_replay_memory([curr_state, action, reward, new_state, done])
                agent.train(done)

                episode_reward += reward
                curr_state = new_state
                step += 1

            print(f'Episode: {curr_ep} -- (psi: {psi_init}, y: {y_init}) ---- Episode reward: {episode_reward} -- Epsilon: {epsilon}')

            if not episode%SHOW_EVERY and SHOW_PREVIEW:
                env.render()

            if not episode%UPDATE_TARGET_EVERY:
                agent.target_model.set_weights(agent.model.get_weights())

            ep_rewards.append(episode_reward)




            if (not episode%AGGREGATE_STATS_EVERY):
                val_reward = agent.evaluate(episode)
                val_rewards.append(val_reward)

                if SAVE_MODEL:
                    average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    print(f'Average reward last batch: {average_reward} -- Evaluative reward after last batch: {val_reward}')
                    #dist_to_path = sum(dist_to_path[-AGGREGATE_STATS_EVERY:])/(STEPS*len(dist_to_path[-AGGREGATE_STATS_EVERY:]))
                    agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon, evaluative_reward=val_reward)

                    # Save model, but only when min reward is greater or equal a set value
                    if average_reward > MIN_REWARD:
                        agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon
            #epsilon *= EPSILON_DECAY
            #epsilon = max(MIN_EPSILON, epsilon)




env = gym.make('ContainerEnv-v0')
agent = DQN_Agent()
run_experiment(agent)
