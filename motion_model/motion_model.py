from filterpy.kalman import KalmanFilter, IMMEstimator
from filterpy.common import Q_discrete_white_noise
from generate_model import CellWalker
from random import uniform as randuniform
import random
from math import pi as PI
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# Make runs deterministic
random.seed(0)

# Each cell has a 2-D position and velocity. All-together, (x, y, xdot, ydot)
BOUNDS = ((0,0), (100,100))
x = randuniform(BOUNDS[0][0], BOUNDS[1][0])
y = randuniform(BOUNDS[0][1], BOUNDS[1][1])

# Random starting velocity
speed = randuniform(0, 1)
bearing = randuniform(0, 2 * PI)
xdot = speed * math.cos(bearing)
ydot = speed * math.sin(bearing)

cell = CellWalker((x,xdot,y,ydot))

print('Cell with starting position ({x:.3f},{y:.3f}) and velocity ({xdot:.3f},{ydot:.3f}) created'.format(
    x=x, y=y, xdot=xdot, ydot=ydot))

times = np.linspace(0, 50, 50)
positions = []
for t in times:
    positions.append(cell.get_measurement(t))

xs, ys = zip(*positions)
f2 = plt.figure()
plt.scatter(xs, ys)
plt.plot(xs, ys)
plt.title('Measurements')
plt.xlabel('x')
plt.ylabel('y')
f2.show()

# Basic Kalman filter
# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
# See Chapter 08
tracker = KalmanFilter(dim_x=4, dim_z=2)
dt = times[1] - times[0]
tracker.F = np.array([[1, dt, 0, 0],
                      [0,  1, 0, 0],
                      [0,  0, 1, dt],
                      [0,  0, 0, 1]])
q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
tracker.Q = block_diag(q, q)
tracker.B = 0
tracker.H = np.array([[1.0, 0, 0, 0],
                      [0  , 0, 1, 0]])
tracker.R = np.array([[0.1, 0],
                      [0,   0.1]])
tracker.x = np.array([[x, y, 0, 0]]).T
tracker.P = np.eye(4)*10.

# Convert measurements to expected form
# Note: n x 4 x 1 np array, measurements must be a col vector
measurements = np.array([np.array([pos]).T for pos in positions])

mu, cov, _, _ = tracker.batch_filter(measurements)

f3 = plt.figure()
plt.scatter(xs, ys)
plt.plot(mu[:,0], mu[:,2])
plt.title('Comparison of Measurements and Kalman Filter')
plt.xlabel('x')
plt.ylabel('y')
f3.show()

# IMM

1