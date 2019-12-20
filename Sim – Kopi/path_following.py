#import simpy
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import math
import numpy as np

# Constants
deg2rad = math.pi/180
rad2deg = 180/math.pi
m2km = 1/1000

# Simulation
h = 1
simtime = 2000

# Initial conditions
x_init = 0
y_init = 4000



# speed
speed = 7


def rot_matrix(alpha):
    return [[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]]

def transpose2D(matrix):
    temp = matrix[0][1]
    matrix[0][1] = matrix[1][0]
    matrix[1][0] = temp

    return matrix



def plot_position(ship, controller):


    xmin, xmax, ymin, ymax = (0, 8.750, 0, 5.000)

    # Plot waypoints as regions of acceptance
    for wp in controller.waypoints:
        circle = plt.Circle((wp[0]*m2km, wp[1]*m2km), controller.R*m2km, color='r', fill=False)
        plt.gcf().gca().add_artist(circle)

    plt.plot(ship.xpos_array, ship.ypos_array)


    # Plot desired path as line segments
    for i in range(len(controller.waypoints)-1):
        plt.plot([m2km*controller.waypoints[i][0], m2km*controller.waypoints[i+1][0]], [m2km*controller.waypoints[i][1], m2km*controller.waypoints[i+1][1]], linestyle = '--', color='g')

    plt.title("Position of ship",
              fontsize=12,fontweight="bold")
    plt.xlabel("X position [km]",fontsize=9,fontweight="bold")
    plt.ylabel("Y position [km]",fontsize=9,fontweight="bold")


    plt.show()


class Ship(object):
    def __init__(self, xpos_init, ypos_init):

        # Position and heading in NED frame
        self.xpos = xpos_init
        self.ypos = ypos_init
        self.psi = 0
        self.r = 0

        self.ypos_array = []
        self.xpos_array = []
        self.time = [0]


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

        self.waypoints = []

        self.path_planner()
        self.path_following(ship)
        #self.path_following = env.process(self.path_following(env, ship))


    # Very simple "planner", to be substituted for DRL planner
    def path_planner(self):

        self.waypoints.extend([[x_init,y_init]])
        self.waypoints.extend([[10_000,0]])
        #self.waypoints.extend([[3500,2000]])
        #self.waypoints.extend([[7500,0000]])
        #self.waypoints.extend([[8000,-3000]])


    def path_following(self, ship):

        for i in range(len(self.waypoints)-1):

            # atan2(y[k+1] - y[k], x[k+1] - x[k])
            alpha = math.atan2(self.waypoints[i+1][1] - self.waypoints[i][1], self.waypoints[i+1][0] - self.waypoints[i][0])
            prev_goal = self.waypoints[i]
            goal = self.waypoints[i+1]

            self.autopilot(ship, alpha, prev_goal, goal)


    def autopilot(self, ship, psi_c, prev_goal, goal):

        rot = rot_matrix(psi_c)
        rot_T = transpose2D(rot)

        time = ship.time[-1]
        y_i_p = 0
        pf_origin = [prev_goal[0], prev_goal[1]]

        print('At time:', time)
        print('---> Angle: %s, Origin of path fixed frame: %s' % (psi_c*rad2deg, pf_origin))

        while ((goal[0] - ship.xpos)**2 + (goal[1] - ship.ypos)**2) > self.R**2:
            # Velocities in x and y direction
            x_dot = speed*math.cos(ship.psi)
            y_dot = speed*math.sin(ship.psi)

            # Euler integration, position
            ship.ypos = ship.ypos + h*y_dot
            ship.xpos = ship.xpos + h*x_dot

            # Translation to path fixed frame
            pf_pos = [ship.xpos - pf_origin[0], ship.ypos - pf_origin[1]]

            # Position in path fixed frame (rotation)
            xpos_p = rot_T[0][0]*pf_pos[0] + rot_T[0][1]*pf_pos[1]
            ypos_p = rot_T[1][0]*pf_pos[0] + rot_T[1][1]*pf_pos[1]
            y_dot_p = rot_T[1][0]*x_dot + rot_T[1][1]*y_dot

            # Integral update, y-direction in path fixed frame
            y_i_p = y_i_p + h*ypos_p

            # Control law
            delta = -self.k_p*ypos_p - self.k_d*y_dot_p - self.k_i*y_i_p

            # Yaw dynamics in 3-DOF (Nomoto)
            self.r_dot = (self.K*delta + self.b - ship.r)/self.T
            ship.r = ship.r + h*self.r_dot
            ship.psi = ship.psi + h*ship.r

            # Save time and position (in km)
            time += h
            ship.time.extend([time])
            ship.xpos_array.extend([ship.xpos*m2km])
            ship.ypos_array.extend([ship.ypos*m2km])



ship = Ship(x_init, y_init)
controller = Controller(ship)

plot_position(ship,controller)
