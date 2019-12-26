import math
import numpy as np

def Lcontainer(x,ui,U0):
    # [xdot,U] = Lcontainer(x,ui,U0) returns the speed U in m/s (optionally) and the
    # time derivative of the state vector: x = [ u v r x y psi p phi delta ]'  using the
    # the LINEARIZED model corresponding to the nonlinear model container.m.
    #
    # u     = surge velocity          (m/s)
    # v     = sway velocity           (m/s)
    # r     = yaw velocity            (rad/s)
    # x     = position in x-direction (m)
    # y     = position in y-direction (m)
    # psi   = yaw angle               (rad)
    # p     = roll velocity           (rad/s)
    # phi   = roll angle              (rad)
    # delta = actual rudder angle     (rad)
    #
    # The inputs are :
    #
    # Uo     = service speed (optinally. Default speed U0 = 7 m/s
    # ui     = commanded rudder angle   (rad)
    #
    # Reference:  Son og Nomoto (1982). On the Coupled Motion of Steering and
    #             Rolling of a High Speed Container Ship, Naval Architect of Ocean Engineering,
    #             20: 73-83. From J.S.N.A. , Japan, Vol. 150, 1981.
    #
    # Author:    Thor I. Fossen
    # Date:      23rd July 2001
    # Revisions:

    # Check of input and state dimensions
    if (len(x) != 9):
        print('Error: x-vector must have dimension 9!')

    # Check of service speed
    if U0==0:
        U0=7.0
    if U0 <=0:
        print('Error: The ship must have speed greater than zero')

    # Normalization variables
    rho = 1025                 # water density (kg/m^3)
    L = 175                    # length of ship (m)
    U = math.sqrt(U0**2 + (x[1])**2)    # ship speed (m/s)

    # rudder limitations
    delta_max  = 10             # max rudder angle (deg)
    Ddelta_max = 5              # max rudder rate (deg/s)

    # States and inputs
    delta_c = ui[0]

    v       = x[1]
    y       = x[4]
    r       = x[2]
    psi     = x[5]
    p       = x[6]
    phi     = x[7]
    nu      = np.array([v,r,p]).reshape(3,1)
    eta     = np.array([y,psi,phi]).reshape(3,1)
    delta   = x[8]

    # Linear model using nondimensional matrices and states with dimension (see Fossen 2002):
    # TM'inv(T) dv/dt + (U/L) TN'inv(T) v + (U/L)^2 TG'inv(T) eta = (U^2/L) T b' delta

    T    = np.array([[1,0,0],[0,1/L,0],[0,0,1/L]])
    Tinv = np.array([[1,0,0],[0,L,0],[0,0,L]])

    M = np.array([[0.01497,0.0003525,-0.0002205], [0.0003525,0.000875,0],[-0.0002205,0,0.0000210]])

    N = np.array([[0.012035,0.00522,0],[0.0038436,0.00243,-0.000213],[-0.000314,0.0000692,0.0000075]])

    G = np.array([[0,0,0.0000704],[0,0,0.0001468],[0,0,0.0004966]])

    b = np.array([-0.002578,0.00126,0.0000855]).reshape(3,1)

    # Rudder saturation and dynamics
    if abs(delta_c) >= delta_max* math.pi/180:
       delta_c = np.sign(delta_c)*delta_max* math.pi/180

    delta_dot = delta_c - delta

    if abs(delta_dot) >= Ddelta_max* math.pi/180:
       delta_dot = np.sign(delta_dot)*Ddelta_max* math.pi/180

    # TM'inv(T) dv/dt + (U/L) TN'inv(T) v + (U/L)^2 TG'inv(T) eta = (U^2/L) T b' delta
    nudot = np.dot(np.linalg.inv(np.dot(np.dot(T,M),Tinv)),(((U**2)/L)*np.dot(T,b)*delta-(U/L)*np.dot(np.dot(T,N),np.dot(Tinv,nu))-((U/L)**2)*np.dot(np.dot(T,G),np.dot(Tinv,eta))))
    #print(f'First part:{np.linalg.inv(np.dot(np.dot(T,M),Tinv))}')
    #print(f'Last part: {(((U**2)/L)*np.dot(T,b)*delta-(U/L)*np.dot(np.dot(T,N),np.dot(Tinv,nu))-((U/L)**2)*np.dot(np.dot(T,G),np.dot(Tinv,eta)))}')
    #print(f'Nudot: {nudot}')

    # Dimensional state derivatives xdot = [ u v r x y psi p phi delta ]'
    xdot =[0,nudot[0],nudot[1],math.cos(psi)*U0-math.sin(psi)*math.cos(phi)*v,math.sin(psi)*U0+math.cos(psi)*math.cos(phi)*v,math.cos(phi)*r,nudot[2],p,delta_dot]
    xdot = [float(i) for i in xdot]

    return [xdot, U]
