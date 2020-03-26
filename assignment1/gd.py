"""
Generic implementation of gradient descent.
"""

from numpy import *
import util
import matplotlib.pyplot as plt

def gd(func, grad, x0, numIter, stepSize):
    """
    Perform gradient descent on some function func, where grad(x)
    computes its gradient at position x.  Begin at position x0 and run
    for exactly numIter iterations.  Use stepSize/sqrt(t+1) as a
    step-size, where t is the iteration number.

    We return the final solution as well as the trajectory of function
    values.
    """
    
    # initialize current location
    x = x0

    # set up storage for trajectory of function values
    trajectory = zeros(numIter + 1)
    trajectory[0] = func(x)

    # begin iterations
    for iter in range(numIter):
        # compute the gradient at the current location
        g = grad(x)


        # compute the step size
        eta = stepSize/(sqrt(iter+1))


        # step in the direction of the gradient
        x = x - eta * g


        # record the trajectory
        trajectory[iter+1] = func(x)

    # return the solution
    return (x, trajectory)


# func = lambda x: 1/x * sin(x)
# grad = lambda x: cos(x) - 1/(x**2)
#
# # xs = arange(-20,20,0.1)
# # ys = func(xs)
# # plt.plot(xs, ys)
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # # plt.show()
# # plt.savefig('./plot.png')
#
# x, trajectory = gd(func, grad, 15, 1000, 0.2)
# print(x)
# plt.plot(trajectory)
# plt.show()

# x, trajectory = gd(lambda x: linalg.norm(x)**2, lambda x: 2*x, array([10,5]), 100, 0.2)