# Generate models and test them
#   Start with the model, generate data and
#   TODO: Generate images and process them to validate approach

from random import uniform as randuniform
from numpy.random import normal as randnorm
from math import pi as PI
import math
import numpy as np
import matplotlib.pyplot as plt


class CellWalker:
    """Lazy evaluation random walker cell"""

    def __init__(self, x):
        """
        Cell starts off stationary

        Args:
            x (4-tuple of doubles): Starting state (x, xdot, y, ydot)
        """

        # Current time, mode, and state
        self.t = 0  # initial time
        self.mode = 0
        self.x = x

        # Cumulative time, mode, and state
        self.ts = [self.t]
        self.modes = [self.mode]
        self.xs = [self.x]

        # Internal model components
        self.noise = 0.1  # std dev
        # The cell does 1 of 2 things: move forward with some bearing and velocity, or is stopped, and turning (changing bearing)
        # self.Tij = np.array([[0.2, 0.4, 0.4],
        #                      [0.1, 0.8, 0.1],
        #                      [0.4, 0.4, 0.2]])  # Transition probability i(row)->j(col)
        self.Tij = np.array([[0.0, 1.0, 0.0],
                             [0.0, 0.9, 0.1],
                             [0.0, 0.8, 0.2]])  # Transition probability i(row)->j(col)
        self.dt = 1  # time interval for evaluating transition

    def get_state(self, t):
        """Return the state (x,y,xdot,ydot) at user-specifed time t. Controls lazy evaluation of Cell's state by calling
        advance if the desired time hasn't been simulated yet or interpolating the state if it's already been simulated.
        """
        # Advance to or past t if necessary
        while self.t < t:
            self.advance()

        # Exact equality with a time
        if t in self.ts:
            return self.xs[self.ts.index(t)]

        # Interpolate state
        #   Within any timestep, the state can be calculated by interpolating the position since there's no velocity
        #   change within a timestep
        ind = np.argmax(t < self.ts)  # get the index of times just past the user-specified time
        state_left = self.xs[ind-1]
        state_right = self.xs[ind]
        state_out = [0]*4
        for i in range(4):
            state_out[i] = (state_left[i] + state_right[i])/2

        return tuple(state_out)

    def get_measurement(self, t):
        """Get noisy position measurements from state"""
        x = self.get_state(t)
        return (x[0] + randnorm(0, self.noise),
                x[2] + randnorm(0, self.noise))

    def advance(self):
        """Advance the cell's position"""
        self.t += self.dt
        self.mode = self.next_mode()
        self.x = self.next_state()

        self.ts.append(self.t)
        self.modes.append(self.mode)
        self.xs.append(self.x)

    def next_mode(self):
        """Pick next mode randomly according to current mode and transition matrix T_{i->j}"""
        Ti = self.Tij[self.mode,:]
        Ti_cum = np.cumsum(Ti)
        x = randuniform(0, 1)
        return np.argmax(x<Ti_cum)

    def next_state(self):
        """Get next state according to current mode and state"""
        if self.mode == 0:  # Const pos
            return self.x[0], 0, self.x[2], 0
        elif self.mode == 1:  # Const vel
            return self.x[0] + self.x[1] * self.dt, self.x[1], self.x[2] + self.x[3] * self.dt, self.x[3]
        elif self.mode == 2:  # Const pos + rotation
            speed = randuniform(0, 1)
            bearing = randuniform(0, 2 * PI)
            return self.x[0], speed * math.cos(bearing), self.x[2], speed * math.sin(bearing)
        else:
            raise ValueError('State must be an integer 0, 1, or 2')


def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


if __name__ == "__main__":
    NCELLS = 1
    BOUNDS = ((0,0), (100,100))

    # Initialize cells
    cells = []
    for i in range(NCELLS):
        # Random starting position
        x = randuniform(BOUNDS[0][0], BOUNDS[1][0])
        y = randuniform(BOUNDS[0][1], BOUNDS[1][1])

        # Random starting velocity
        speed = randuniform(0,1)
        bearing = randuniform(0, 2*PI)
        xdot = speed*math.cos(bearing)
        ydot = speed*math.sin(bearing)

        cells.append(CellWalker((x,xdot,y,ydot)))
        print('Cell with starting position ({x:.3f},{y:.3f}) and velocity ({xdot:.3f},{ydot:.3f}) created'.format(
            x=x, y=y, xdot=xdot, ydot=ydot))

    # Debug: advance cell a bunch of times
    #   This isn't necessary, but it's just here to show what's being calculated inside the cell
    nt = 150
    for cell in cells:
        for t in range(nt):
            cell.advance()
        maxt = cell.t  # makes sure the later commands end in exactly the same place

        # Plot exact states as advanced
        xs, xdots, ys, ydots = zip(*cell.xs)  # Unzips list of pairs into 2 lists
        f1 = plt.figure()
        plt.scatter(xs, ys)
        plt.plot(xs, ys)  # connect points with lines
        xmin_, xmax_ = plt.xlim()
        ymin_, ymax_ = plt.ylim()
        plt.title('Exact States as Advanced')
        plt.xlabel('x')
        plt.ylabel('y')
        f1.show()

    # Report cell state at user-specified times
    times = np.linspace(0, maxt, 20)
    for cell in cells:
        # Plot exact states
        positions = []
        for t in times:
            positions.append(cell.get_state(t))

        xs, xdots, ys, ydots = zip(*positions)
        f2 = plt.figure()
        plt.scatter(xs, ys)
        plt.plot(xs, ys)
        plt.xlim(xmin_, xmax_)
        plt.ylim(ymin_, ymax_)
        plt.title('Exact States')
        plt.xlabel('x')
        plt.ylabel('y')
        f2.show()

        # Plot noisy measurements
        positions = []
        for t in times:
            positions.append(cell.get_measurement(t))

        xs, ys = zip(*positions)
        f3 = plt.figure()
        plt.scatter(xs, ys)
        plt.plot(xs, ys)
        plt.xlim(xmin_, xmax_)
        plt.ylim(ymin_, ymax_)
        plt.title('Noisy Measurements')
        plt.xlabel('x')
        plt.ylabel('y')
        f3.show()

        # Plot distance of exact state from starting point
        f4 = plt.figure()
        start = positions[0]
        dists = []
        for pos in positions:
            dists.append(distance(start, pos))
        plt.plot(times, dists)
        plt.title('Distance From Start')
        plt.xlabel('t')
        plt.ylabel('distance')
        f4.show()

1