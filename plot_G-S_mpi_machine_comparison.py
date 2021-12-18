import os
import sys
import glob

import numpy
import matplotlib.pyplot as plt
import math

#this script generates the plot used in the Appendix to show the two hardware setups output the same
#numerical result to within machine epsilon
#requires the relevant txt files from alfven as well as the local computer

def load_data(path):

    # Estimate number of processors
    num_procs = len(glob.glob(os.path.join(path, "G-S_row_nonlinear_*_local.txt")))

    # Load all data
    data = []
    rank_N = numpy.empty(num_procs, dtype=int)
    # N = numpy.empty(num_procs, dtype=int)
    for i in range(num_procs):
        data.append(numpy.loadtxt(os.path.join(path, "G-S_row_nonlinear_%s_local.txt" % i)))
        N = data[-1].shape[1]
        rank_N[i] = data[-1].shape[0]
    
    print("Grids: N = %s, rank_N = %s" % (N, rank_N))
    # data = numpy.array(data)
    # print(data.shape)
    # print(data[0].shape[0])
    # print(data[0].shape[1])
    # print(data[0])
    # assert(N == rank_N.sum())
    
    # Create data arrays
    x = numpy.linspace(0, 1, N)
    y = numpy.linspace(-0.5, 0.5, rank_N.sum())
    X, Y = numpy.meshgrid(x,y)
    
    # U = numpy.empty((int(math.sqrt(num_procs)) * int(rank_N[0]), int(math.sqrt(num_procs)) * int(rank_N[0])))
    U = numpy.empty((N,rank_N.sum()))
    index = 0
    for i in range(num_procs):
        U[:, index:index + data[i].shape[0]] = data[i].transpose()
        index += data[i].shape[0]

    return X, Y, U.transpose()

def load_data_alfven(path):

    # Estimate number of processors
    num_procs = len(glob.glob(os.path.join(path, "G-S_row_nonlinear_*_alfven.txt")))

    # Load all data
    data = []
    rank_N = numpy.empty(num_procs, dtype=int)
    # N = numpy.empty(num_procs, dtype=int)
    for i in range(num_procs):
        data.append(numpy.loadtxt(os.path.join(path, "G-S_row_nonlinear_%s_alfven.txt" % i)))
        N = data[-1].shape[1]
        rank_N[i] = data[-1].shape[0]
    
    print("Grids: N = %s, rank_N = %s" % (N, rank_N))
    # data = numpy.array(data)
    # print(data.shape)
    print(data[0].shape[0])
    print(data[0].shape[1])
    print(data[0])
    # assert(N == rank_N.sum())
    
    # Create data arrays
    x = numpy.linspace(0, 1, N)
    y = numpy.linspace(-0.5, 0.5, rank_N.sum())
    X, Y = numpy.meshgrid(x,y)
    
    # U = numpy.empty((int(math.sqrt(num_procs)) * int(rank_N[0]), int(math.sqrt(num_procs)) * int(rank_N[0])))
    U = numpy.empty((N,rank_N.sum()))
    index = 0
    for i in range(num_procs):
        U[:, index:index + data[i].shape[0]] = data[i].transpose()
        index += data[i].shape[0]

    return X, Y, U.transpose()

def plot_solution(x, y, u, u_alfven):
    fig = plt.figure()
    fig.set_figwidth(fig.get_figwidth() * 1)
    
    axes = fig.add_subplot(1, 1, 1)
    plot = axes.contourf(x, y, u, 20, cmap = 'RdBu_r')
    fig.colorbar(plot)
    axes.set_title("Picard Iteration Results for $\psi$ With High B")
    axes.set_xlabel("R Coordinate")
    axes.set_ylabel("Z Coordinate")

    fig3 = plt.figure()
    fig3.set_figwidth(fig.get_figwidth() * 1)
    axes = fig3.add_subplot(1,1,1)
    plot3 = axes.contourf(x, y, (u - u_alfven), 20, cmap = 'RdBu_r')
    # fig.colorbar(plot3)
    axes.set_title('Difference Between Solutions on Different Machines (Zero Everywhere)')
    axes.set_xlabel("R Coordinate")
    axes.set_ylabel("Z Coordinate")

    return None

if __name__ == '__main__':
    path = os.getcwd()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    X, Y, U = load_data(path)
    X, Y, U_alfven = load_data_alfven(path)

    fig = plot_solution(X, Y, U, U_alfven)
    plt.show()