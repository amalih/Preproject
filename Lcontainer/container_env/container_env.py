
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import gym
from container_model import Lcontainer

# Constants
deg2rad = math.pi/180
rad2deg = 180/math.pi
m2km = 1/1000

# Simulation
h = 0.1

actions = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,1,2,3,4,5,6,5,8,9,10]

# Limiting constants
MAX_DELTA = 10*deg2rad
MAX_DELTA_D = 5*deg2rad
MAX_CTE = 500
MAX_INIT_CTE = 200
MAX_INIT_PSI = math.pi/2

# speed
SPEED = 7
OFFSET = 0

def rot_matrix(alpha):
    return [[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]]

def transpose2D(matrix):
    temp = matrix[0][1]
    matrix[0][1] = matrix[1][0]
    matrix[1][0] = temp

    return matrix

def map_to_negpi_pi(angle):
    angle = angle%(2*math.pi)
    if angle > math.pi:
        angle = angle - 2*math.pi
    if angle < -math.pi:
        angle = angle + 2*math.pi

    return angle


def plot(psi_array, ct_error_array, ship, controller, psi_c):

    plt.figure(1)
    plt.subplot(311)

    #params = {'backend': 'ps',
    #  'axes.labelsize': 12,
    #  'font.size': 12,
    #  'legend.fontsize': 12,
    #  'xtick.labelsize': 10,
    #  'ytick.labelsize': 10,
    #  'text.usetex': True}
    #plt.rcParams.update(params)

    #plt.plot([0,10_000], [0, 0])

    plt.plot(ship.xpos_array, ship.ypos_array)
    #plt.plot([x*m2km for x in self.xpos_array], [y*m2km for y in self.ypos_array])
    #plt.plot([0, 7_000], [0, 0], linestyle='--')

    plt.title("Position of ship",
              fontsize=12,fontweight="bold")
    plt.xlabel("X position [km]",fontsize=10,fontweight="bold")
    plt.ylabel("Y position [km]",fontsize=10,fontweight="bold")

    #plt.plot([0, math.cos(psi_c)*0.5], [0, math.sin(psi_c)*0.5])

    #

    #plt.xlim(-1,3)
    #plt.ylim(-1,3)


#    plt.subplot(312)
#    params = {'backend': 'ps',
#      'axes.labelsize': 12,
    #  'font.size': 12,
    #  'legend.fontsize': 12,
    #  'xtick.labelsize': 10,
    #  'ytick.labelsize': 10,
    #  'text.usetex': True}
    #plt.rcParams.update(params)
    #plt.title("Control input",
    #          fontsize=12,fontweight="bold")
    #plt.xlabel("Episode",fontsize=10,fontweight="bold")
    #plt.ylabel("Rudder angle [deg]",fontsize=10,fontweight="bold")
    #plt.plot(controller.time_array, controller.action_array)

    plt.subplot(312)
    plt.title("Cross-track error",
              fontsize=12,fontweight="bold")
    plt.plot(ct_error_array)

    plt.subplot(313)
    plt.title("Psi",
              fontsize=12,fontweight="bold")
    plt.plot(psi_array)

    plt.show()

    # Plot waypoints as regions of acceptance
    #for wp in controller.waypoints:
    #    circle = plt.Circle((wp[0]*m2km, wp[1]*m2km), controller.R*m2km, color='r', fill=False)
    #    plt.gcf().gca().add_artist(circle)



    # Plot desired path as line segments
    #for i in range(len(controller.waypoints)-1):
    #    plt.plot([m2km*controller.waypoints[i][0], m2km*controller.waypoints[i+1][0]], [m2km*controller.waypoints[i][1], m2km*controller.waypoints[i+1][1]], linestyle = '--', color='g')


class ContainerEnv(gym.Env):
    metadata = {'render.modes': ['container_vessel']}

    def __init__(self):

        self._seed = 99

        self.ct_error_d = 0
        self.r = 0
        self.u = 0
        self.v = 0

        self.pf_psi_array = []
        self.ct_error_array =[]


    def step(self, action):

        done = self._take_action(action)
        obs = self._get_obs()
        reward = self._get_reward(done)

        return obs, reward, done, {}

    def reset(self):
        self.pf_psi_array = []
        self.ct_error_array =[]
        x_init = 0
        y_init = random.randint(-MAX_INIT_CTE, MAX_INIT_CTE)
        psi_init = random.uniform(-MAX_INIT_PSI, MAX_INIT_PSI)
        psi_c_init = 0#random.uniform(-MAX_INIT_PSI, MAX_INIT_PSI)

        #rot = rot_matrix(psi_c_init)
        #rot_T = transpose2D(rot)

        # Translation to path fixed frame
        #pf_origin = [0,0]
        #pf_pos = [x_init - pf_origin[0], y_init - pf_origin[1]]

        # Position in path fixed frame (rotation)
        self.ct_error = y_init
        #self.ct_error  = rot_T[1][0]*pf_pos[0] + rot_T[1][1]*pf_pos[1]

        self.psi_c = psi_c_init

        self.ship = Ship(x_init, y_init, psi_init)
        self.controller = Controller(self.ship)

        self.ct_error_d = 0
        self.pf_psi = psi_init#-psi_c_init
        self.pf_psi_array.append(self.pf_psi)
        self.r = 0
        self.u = SPEED
        self.v = 0

        return self._get_obs()

    def init_eval(self, y_init, psi_init, psi_c):
        self.pf_psi_array = []
        self.ct_error_array =[]

        #rot = rot_matrix(psi_c)
        #rot_T = transpose2D(rot)
        x_init = 0

        # Translation to path fixed frame
        #pf_origin = [0,0]
        #pf_pos = [x_init - pf_origin[0], y_init - pf_origin[1]]

        # Position in path fixed frame (rotation)
        #self.ct_error  = rot_T[1][0]*pf_pos[0] + rot_T[1][1]*pf_pos[1]
        self.ct_error = y_init
        self.psi_c = psi_c

        self.ship = Ship(x_init, y_init, psi_init)
        self.controller = Controller(self.ship)

        self.ct_error_d = 0
        self.pf_psi = psi_init#-psi_c
        self.pf_psi_array.append(self.pf_psi)
        self.r = 0
        self.u = SPEED
        self.v =0

        return self._get_obs()

    def _get_obs(self):
        # 6 elements
        ct_error_norm = self.ct_error/MAX_CTE
        ct_error_d_norm = self.ct_error_d/SPEED
        pf_psi_norm = self.pf_psi/math.pi
        pf_r_std = self.r*100
        u_norm = self.u/SPEED
        v_norm = self.v/SPEED

        obs = [ct_error_norm, ct_error_d_norm, pf_psi_norm, pf_r_std, u_norm, v_norm]

        return obs

    def _take_action(self, action_idx):
        delta_c = actions[action_idx]*deg2rad
        #delta_c = -10*deg2rad
        #delta_c = 0
        ct_error_prev = self.ct_error

        self.ct_error,self.pf_psi, self.r, self.u, self.v = self.controller.autopilot(self.ship, self.psi_c, delta_c)
        self.ct_error_array.append(self.ct_error*m2km)
        self.pf_psi_array.append(self.pf_psi*rad2deg)

        self.ct_error_d = (abs(self.ct_error) - abs(ct_error_prev))/h

        if abs(self.ct_error) > MAX_CTE:
            return True

        else:
            return False

    def _get_reward(self, done):

        if done:
            return -1

        #if abs(self.pf_psi) < math.pi/2:
        if abs(self.pf_psi) < math.pi/2 and abs(self.ct_error) < 10:

            #std = 20
            #amp = 1
            #reward = amp * math.e**(-(self.ct_error**2)/(2*std**2))

            reward = 1-(1/10)*abs(self.ct_error)
        #    if self.ct_error_d >= 0:
        #        reward = reward/10
            if self.ct_error_d > 0:
                reward/2

            return reward

        else:
            return 0


    def render(self, mode='container_vessel', close=False):

        plot(self.pf_psi_array, self.ct_error_array, self.ship, self.controller, self.psi_c)

        return

class Ship(object):
    def __init__(self, x_init, y_init, psi_init):

        # Position and heading in NED frame
        self.xpos = x_init
        self.ypos = y_init
        self.psi = psi_init
        self.r = 0

        self.psi_array = []
        self.ypos_array = []
        self.xpos_array = []
        self.time  = 0


        self.container_state = [SPEED, 0, 0, 0, y_init, psi_init, 0, 0,0]

class Controller(object):
    def __init__(self, ship):
        # Radius of acceptance (nLpp)
        self.R = 2*175

        self.action_array = [0]
        self.time_array = [0]


    def autopilot(self, ship, psi_c, delta_c):

        self.action_array.append(delta_c)
        time = self.time_array[-1]

        #rot = rot_matrix(psi_c)
        #rot_T = transpose2D(rot)

        #pf_origin = [0,0]
        #pf_pos = [ship.xpos - pf_origin[0], ship.ypos - pf_origin[1]]

        #ypos_p_prev = rot_T[1][0]*pf_pos[0] + rot_T[1][1]*pf_pos[1]

        xdot, _ = Lcontainer(ship.container_state, [delta_c,SPEED], SPEED)

        for i in range(len(ship.container_state)):
            ship.container_state[i] = ship.container_state[i] + h*xdot[i]
        #ship.container_state[5] = map_to_negpi_pi(ship.container_state[5])
        #print(f'Container state: {ship.container_state}')

        x_dot = ship.container_state[0]
        y_dot = ship.container_state[1]
        ship.r = ship.container_state[2]
        ship.xpos = ship.container_state[3]
        ship.ypos = ship.container_state[4]
        ship.psi = ship.container_state[5]
        ship.psi_array.append(ship.psi)
        #print('----------------')
        #print(f'Xpos: {ship.container_state[3]} -- Ypos: {ship.container_state[4]} -- Xdot: {ship.container_state[0]} -- Ydot: {ship.container_state[1]}')


        # Translation to path fixed frame
        #pf_pos = [ship.xpos - pf_origin[0], ship.ypos - pf_origin[1]]

        # Position in path fixed frame (rotation)
        #xpos_p = rot_T[0][0]*pf_pos[0] + rot_T[0][1]*pf_pos[1]
        #ypos_p = rot_T[1][0]*pf_pos[0] + rot_T[1][1]*pf_pos[1]
        #y_dot_p = rot_T[1][0]*x_dot + rot_T[1][1]*y_dot
        #x_dot_p = rot_T[0][0]*x_dot + rot_T[0][1]*y_dot
        ypos_p = ship.ypos


        #print(f'ypos_p: {ypos_p} -- psi: {ship.psi} -- psi_c: {psi_c}')
        #print('---------------')

        # Save time and position (in km)
        time += h
        self.time_array.append(time)
        ship.xpos_array.extend([ship.xpos*m2km])
        ship.ypos_array.extend([ship.ypos*m2km])

        return ypos_p, ship.psi-psi_c, ship.r, x_dot, y_dot
