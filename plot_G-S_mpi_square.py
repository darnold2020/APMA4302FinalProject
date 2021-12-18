import os
import sys
import glob

import numpy
import matplotlib.pyplot as plt
import math

#This script plots the output from simulations with a grid-based partition scheme
#Its figures are not used in the report, but could be helpful in future investigations as discussed in the Appendix

def load_data(path):

    # Estimate number of processors
    num_procs = len(glob.glob(os.path.join(path, "G-S_square_*_local.txt")))

    # Load all data
    data = []
    rank_N = numpy.empty(num_procs, dtype=int)
    N = numpy.empty(num_procs, dtype=int)
    for i in range(num_procs):
        data.append(numpy.loadtxt(os.path.join(path, "G-S_square_%s_local.txt" % i)))
        N[i] = data[-1].shape[1]
        rank_N[i] = data[-1].shape[0]
    
    print("Grids: N = %s, rank_N = %s" % (N, rank_N))
    data = numpy.array(data)
    print(data.shape)
    print(data[0].shape[0])
    print(data[0].shape[1])
    print(data[0])
    # assert(N == rank_N.sum())
    
    # Create data arrays
    x = numpy.linspace(0, 1.0, int(math.sqrt(num_procs)) * int(rank_N[0]))
    y = numpy.linspace(-0.5, 0.5, int(math.sqrt(num_procs)) * int(rank_N[0]))
    X, Y = numpy.meshgrid(x,y)
    
    U = numpy.empty((int(math.sqrt(num_procs)) * int(rank_N[0]), int(math.sqrt(num_procs)) * int(rank_N[0])))
    index_h = 0
    index_v = 0
    for i in range(num_procs):
        print(index_h)
        print(index_v)
        U[index_h*data[i].shape[0]:index_h*data[i].shape[0] + data[i].shape[0], index_v*data[i].shape[1]:index_v*data[i].shape[1] + data[i].shape[1]] = data[i]
        if i % math.sqrt(num_procs) == 0:
            index_h = index_h + 1
        else:
            index_v = index_v + 1
            index_h = 0
        # U[:, index:index + data[i].shape[0]] = data[i].transpose()
        # index += data[i].shape[0]

    return X, Y, U.transpose()

def plot_solution(x, y, u):
    fig = plt.figure()
    fig.set_figwidth(fig.get_figwidth())
    
    axes = fig.add_subplot(1, 1, 1)
    plot = axes.pcolor(X, Y, U)
    fig.colorbar(plot)
    axes.set_title("Computed Solution")
    axes.set_xlabel("x")
    axes.set_ylabel("y")

    # axes = fig.add_subplot(1, 3, 2)
    # plot = axes.pcolor(X, Y, true_solution(U.shape[0] - 1))
    # print(U.shape[0])
    # fig.colorbar(plot)
    # axes.set_title("True Solution")
    # axes.set_xlabel("x")
    # axes.set_ylabel("y")

    # axes = fig.add_subplot(1, 3, 3)
    # plot = axes.pcolor(X, Y, numpy.abs(U - true_solution(U.shape[0] - 1)))
    # fig.colorbar(plot)
    # axes.set_title("Error")
    # axes.set_xlabel("x")
    # axes.set_ylabel("y")

    return None

def true_solution(N):

    x = numpy.linspace(0, numpy.pi, N + 1)
    y = numpy.linspace(0, numpy.pi, N + 1)
    X, Y = numpy.meshgrid(x,y)

    U_true = 2 * numpy.sin(X) * numpy.cos(3 * Y)

    return U_true

if __name__ == '__main__':
    path = os.getcwd()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    X, Y, U = load_data(path)

    fig = plot_solution(X, Y, U)
    plt.show()