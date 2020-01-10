

import matplotlib.pyplot as plt
import math
import numpy as np
import random
import gym


# Constants
deg2rad = math.pi/180
rad2deg = 180/math.pi
m2km = 1/1000

# Simulation
h = 1

actions = [-10,-8,-6,-5,-4,-3,-2,-1,-0.5,-0.1,0,0.1,0.5,1,2,3,4,5,6,8,10]


# Limiting constants
MAX_DELTA = 10*deg2rad
MAX_DELTA_D = 5*deg2rad
MAX_CTE = 2000
MAX_INIT_CTE = 500
MAX_INIT_PSI = math.pi/4

# speed
SPEED = 4
ENEMY_SPEED = 2
SAFE_DIST = 200
MAX_ENEMY = 3000

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


def plot(ship, controller, psi_c, ct_error_array, rewards, enemy_x, enemy_y):

    plt.figure(1)
    plt.subplot(221)

    #params = {'backend': 'ps',
    #  'axes.labelsize': 12,
    #  'font.size': 12,
    #  'legend.fontsize': 12,
    #  'xtick.labelsize': 10,
    #  'ytick.labelsize': 10,
    #  'text.usetex': True}
    #plt.rcParams.update(params)

    #plt.plot([0,10_000], [0, 0])

    plt.title("Position of ship",
              fontsize=12,fontweight="bold")
    plt.xlabel("X position [km]",fontsize=10,fontweight="bold")
    plt.ylabel("Y position [km]",fontsize=10,fontweight="bold")

    plt.plot(ship.xpos_array, ship.ypos_array)
    plt.plot([i*m2km for i in enemy_x], [i*m2km for i in enemy_y])
    #plt.plot([x*m2km for x in self.xpos_array], [y*m2km for y in self.ypos_array])
    #plt.plot([0, 7_000], [0, 0], linestyle='--')

    plt.plot([0, 6*math.cos(psi_c)], [0, 6*math.sin(psi_c)], '--')


    plt.xlim(0,4)
    plt.ylim(-2,2)


    plt.subplot(222)
    #params = {'backend': 'ps',
    #  'axes.labelsize': 12,
    #  'font.size': 12,
    #  'legend.fontsize': 12,
    #  'xtick.labelsize': 10,
    #  'ytick.labelsize': 10,
    #  'text.usetex': True}
    #plt.rcParams.update(params)
    plt.title("Control input",
              fontsize=12,fontweight="bold")
    plt.xlabel("Step",fontsize=10,fontweight="bold")
    plt.ylabel("Rudder angle [deg]",fontsize=10,fontweight="bold")
    plt.plot(controller.time_array, controller.action_array)

    plt.subplot(223)

    plt.plot([i*m2km for i in ct_error_array])
    plt.title("Cross-track error",
              fontsize=12,fontweight="bold")
    plt.xlabel("Step",fontsize=10,fontweight="bold")
    plt.ylabel("Y position [km]",fontsize=10,fontweight="bold")

    plt.subplot(224)

    plt.plot(rewards)
    plt.title("Reward",
              fontsize=12,fontweight="bold")
    print(len(rewards))

    plt.xlabel("Step",fontsize=10,fontweight="bold")



    #plt.subplot(414)

    #plt.plot(ship.psi_array)
    #plt.plot(pf_psi_array)

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

        self.ct_error = 0
        self.ct_error_d = 0
        self.pf_psi = 0
        self.r = 0
        self.u = 0
        self.v = 0
        self.ct_error_array = []
        self.pf_psi_array = []
        self.rewards = []


    def step(self, action):

        done = self._take_action(action)
        obs = self._get_obs()
        reward = self._get_reward(done)
        self.rewards.append(reward)
        return obs, reward, done, {}

    def reset(self):

        x_init = 0
        y_init = 0#random.randint(-MAX_INIT_CTE, MAX_INIT_CTE)
        psi_c_init = 0#random.uniform(-90*deg2rad, 90*deg2rad)

        psi_init = 0#random.uniform(psi_c_init-MAX_INIT_PSI, psi_c_init+MAX_INIT_PSI)
        delta_init = 0#random.uniform(-MAX_DELTA/5, MAX_DELTA/5)


        rot = rot_matrix(psi_c_init)
        rot_T = transpose2D(rot)

        # Translation to path fixed frame
        pf_origin = [0,0]
        pf_pos = [x_init - pf_origin[0], y_init - pf_origin[1]]

        # Position in path fixed frame (rotation)
        self.ct_error = rot_T[1][0]*pf_pos[0] + rot_T[1][1]*pf_pos[1]

        self.psi_c = psi_c_init
        self.ship = Ship(x_init, y_init, psi_init,delta_init)
        self.controller = Controller(self.ship)


        self.ct_error_d = 0
        self.pf_psi = psi_init-psi_c_init

        self.ct_error_array = [self.ct_error]
        self.pf_psi_array = [self.pf_psi]
        self.rewards = []

        self.r = 0
        self.u = SPEED*math.cos(psi_init)
        self.v = SPEED*math.sin(psi_init)

        self.enemy_y = 0
        self.enemy_x = 3000#random.randint(2000,3000)
        self.enemy_xe = self.enemy_x-x_init
        self.enemy_ye = self.enemy_y-y_init
        self.enemy_ue = -ENEMY_SPEED-self.u
        self.enemy_ve = 0-self.v



        self.enemy_x_array = [self.enemy_x]
        self.enemy_y_array = [self.enemy_y]


        return self._get_obs()

    def init_eval(self, enemy_init):

        x_init = 0
        y_init = 0
        psi_init = 0
        delta_init = 0

        # Position in path fixed frame (rotation)
        self.ct_error  = 0

        self.psi_c = 0
        delta_init = 0

        self.ship = Ship(x_init, y_init, psi_init,delta_init)
        self.controller = Controller(self.ship)

        self.ct_error_d = 0
        self.pf_psi = psi_init-self.psi_c
        self.ct_error_array = [self.ct_error]
        self.pf_psi_array = [self.pf_psi]
        self.rewards = []

        self.r = 0
        self.u = SPEED*math.cos(psi_init)
        self.v = SPEED*math.sin(psi_init)

        self.enemy_y = 0
        self.enemy_x = enemy_init
        self.enemy_xe = self.enemy_x-x_init
        self.enemy_ye = self.enemy_y-y_init
        self.enemy_ue = -ENEMY_SPEED-self.u
        self.enemy_ve = 0-self.v

        self.enemy_x_array = [self.enemy_x]
        self.enemy_y_array = [self.enemy_y]

        #self.ct_error_array.append(self.ct_error)
        #self.pf_psi_array.append(psi_init)

        self.ct_error_array = [self.ct_error]
        self.pf_psi_array = [self.pf_psi]
        self.rewards = []


        return self._get_obs()

    def _get_obs(self):
        # 6 elements
        ct_error_norm = self.ct_error/MAX_CTE
        ct_error_d_norm = self.ct_error_d/SPEED
        pf_psi_norm = self.pf_psi/math.pi
        pf_r_std = self.r*100
        u_norm = self.u/SPEED
        v_norm = self.v/SPEED
        enemy_xe_norm = self.enemy_xe/MAX_ENEMY
        if abs(self.enemy_xe) > MAX_ENEMY:
            enemy_xe_norm = np.sign(self.enemy_xe)*1
        enemy_ye_norm = self.enemy_ye/MAX_ENEMY
        if abs(self.enemy_ye) > MAX_ENEMY:
            enemy_ye_norm = np.sign(self.enemy_ye)*1
        enemy_ue_norm = self.enemy_ue/(ENEMY_SPEED+SPEED)
        enemy_ve_norm = self.enemy_ve/(ENEMY_SPEED+SPEED)


        obs = [ct_error_norm, ct_error_d_norm, pf_psi_norm, pf_r_std, u_norm, v_norm, enemy_xe_norm, enemy_ye_norm, enemy_ue_norm, enemy_ve_norm]

        return obs

    def _take_action(self, action_idx):
        delta_c = actions[action_idx]*deg2rad
        #delta_c = 10*deg2rad
        ct_error_prev = self.ct_error

        self.ct_error, self.pf_psi, self.r, self.u, self.v = self.controller.autopilot(self.ship, self.psi_c, delta_c, self.u, self.v)

        self.enemy_x = self.enemy_x -ENEMY_SPEED*h
        self.enemy_y = 0

        self.enemy_ye = 0 - self.ct_error
        self.enemy_xe = self.enemy_x - self.ship.xpos

        self.enemy_ue = -ENEMY_SPEED - self.u
        self.enemy_ve = 0 - self.v

        self.enemy_x_array.append(self.enemy_x)
        self.enemy_y_array.append(self.enemy_y)

        self.pf_psi_array.append(self.pf_psi)
        self.ct_error_array.append(self.ct_error)

        self.ct_error_d = (abs(self.ct_error) - abs(ct_error_prev))/h

        if abs(self.ct_error) > MAX_CTE:
            return True
        if math.sqrt(self.enemy_xe**2 + self.enemy_ye**2) < SAFE_DIST:
            print('Hit the enemy!')
            return True

        else:
            return False

    def _get_reward(self, done):


        if math.sqrt(self.enemy_xe**2 + self.enemy_ye**2) < SAFE_DIST:
            return -1000
        #if abs(self.pf_psi) < math.pi/2:
        elif abs(self.pf_psi) < math.pi/4 and abs(self.ct_error) < 10:
            reward = 1-(1/10)*abs(self.ct_error)
            
        elif abs(self.ct_error) > MAX_CTE:
            return -1


            #std = 20
           # amp = 1
            #reward = amp * math.e**(-(self.ct_error**2)/(2*std**2))


        #    if self.ct_error_d >= 0:
        #        reward = reward/10

            return reward

        else:
            return 0


    def render(self, mode='container_vessel', close=False):

        plot(self.ship, self.controller, self.psi_c, self.ct_error_array, self.rewards, self.enemy_x_array, self.enemy_y_array)

        return

class Ship(object):
    def __init__(self, xpos_init, ypos_init, psi_init,delta_init):

        # Position and heading in NED frame
        self.xpos = xpos_init
        self.ypos = ypos_init
        self.psi = psi_init
        self.r = 0

        self.psi_array = [psi_init]
        self.ypos_array = [ypos_init*m2km]
        self.xpos_array = [xpos_init*m2km]
        self.time  = 0

        self.delta = delta_init


class Controller(object):
    def __init__(self, ship):

        # Gains, yaw dynamics
        self.T = 20
        self.K = 0.1
        self.b = 0.001

        # Controller gains
        self.k_d = 0.04
        self.k_p = 0.0005
        self.k_i = 0.000001

        # Angular acceleration "in yaw"
        self.r_dot = 0

        # Radius of acceptance (nLpp)
        self.R = 2*175

        self.action_array = [0]
        self.time_array = [0]

        #self.pf_pos_array =[]


    def autopilot(self, ship, psi_c, delta_c, u, v):

        self.action_array.append(delta_c*rad2deg)

        rot = rot_matrix(psi_c)
        rot_T = transpose2D(rot)

        time = self.time_array[-1]



        # Translation to path fixed frame
        pf_origin = [0,0]




        #x_dot_p = rot_T[0][0]*x_dot + rot_T[0][1]*y_dot

        #self.pf_pos_array.append(ypos_p*m2km)


        # Integral update, y-direction in path fixed frame
        #y_i_p = y_i_p + h*ypos_p

        # Control law
        #delta = -self.k_p*ypos_p - self.k_d*y_dot_p - self.k_i*y_i_p

        # Yaw dynamics in 3-DOF (Nomoto)
        if abs(delta_c) >= MAX_DELTA:
           delta_c = np.sign(delta_c)*MAX_DELTA

        self.r_dot = (self.K*delta_c + self.b - ship.r)/self.T
        ship.r = ship.r + h*self.r_dot

        ship.psi = ship.psi + h*ship.r
        ship.psi = map_to_negpi_pi(ship.psi)

        ship.psi_array.append(ship.psi)
        #print(f'r: {ship.r}, Psi: {ship.psi}')

        # Velocities in x and y direction
        #x_dot = u*math.cos(ship.psi) + v*math.sin(ship.psi)
        #y_dot = v*math.cos(ship.psi) + u*math.sin(ship.psi)

        x_dot = SPEED*math.cos(ship.psi)
        y_dot = SPEED*math.sin(ship.psi)


        #print(f'xdot: {x_dot}, ydot: {y_dot}')

        # Euler integration, position
        ship.ypos = ship.ypos + h*y_dot
        ship.xpos = ship.xpos + h*x_dot

        pf_pos = [ship.xpos - pf_origin[0], ship.ypos - pf_origin[1]]


        # Position in path fixed frame (rotation)
        #xpos_p = pf_pos[0]*math.cos(psi_c) + pf_pos[1]*math.sin(psi_c)
        #ypos_p = -pf_pos[0]*math.sin(psi_c) + pf_pos[1]*math.cos(psi_c)
        xpos_p = rot_T[0][0]*pf_pos[0] + rot_T[0][1]*pf_pos[1]
        ypos_p = rot_T[1][0]*pf_pos[0] + rot_T[1][1]*pf_pos[1]

        #x_dot_p = x_dot*math.cos(-psi_c) - y_dot*math.sin(-psi_c)
        #y_dot_p = x_dot*math.sin(-psi_c) + y_dot*math.cos(-psi_c)

        #y_dot_p = rot_T[1][0]*x_dot + rot_T[1][1]*y_dot


        # Save time and position (in km)
        time += h
        self.time_array.append(time)
        ship.xpos_array.extend([ship.xpos*m2km])
        ship.ypos_array.extend([ship.ypos*m2km])

        #return ypos_p, ship.psi-psi_c, ship.r, x_dot_p, y_dot_p
        return ship.ypos, ship.psi-psi_c, ship.r, x_dot, y_dot
