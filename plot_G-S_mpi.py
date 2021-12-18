import os
import sys
import glob

import numpy
import matplotlib.pyplot as plt
import math

#This script generates the plots used in sections V. A and C displaying the behavior of the step size
#in the solution Psi over each Picard iteration as well as the contour plots for the nonlinear solutions
#with a given constant B, the value of which is set in the G-S_Solve_mpi.cpp file

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

def plot_solution(x, y, u):
    fig = plt.figure()
    fig.set_figwidth(fig.get_figwidth() * 1)
    
    axes = fig.add_subplot(1, 1, 1)
    plot = axes.contourf(X, Y, u, 20, cmap = 'RdBu_r')
    fig.colorbar(plot)
    axes.set_title("Picard Iteration Results for $\psi$ With High B")
    axes.set_xlabel("R Coordinate")
    axes.set_ylabel("Z Coordinate")

    fig2 = plt.figure()
    fig.set_figwidth(fig.get_figwidth() * 1)
    axes = fig2.add_subplot(1,1,1)
    dPsi_ser = numpy.loadtxt(os.path.join(path, "dPsi_over_steps_serial.txt"))
    num_steps_ser = dPsi_ser.shape[0]
    plot = axes.plot(numpy.linspace(1, num_steps_ser+1, num_steps_ser), dPsi_ser)
    dPsi_par = numpy.loadtxt(os.path.join(path, "dPsi_over_steps_parallel.txt"))
    num_steps_par = dPsi_par.shape[0]
    plot = axes.plot(numpy.linspace(1, num_steps_par+1, num_steps_par), dPsi_par)
    num_procs = len(glob.glob(os.path.join(path, "G-S_row_nonlinear_*_local.txt")))
    if (num_procs == 1):
        axes.set_title("Change in $\psi$ With Picard Cycle Count (Serial)")
    else:
        axes.set_title("Change in $\psi$ With Picard Cycle Count (Parallel: 6 Processes)")
    axes.set_xlabel("Cycle Number")
    axes.set_ylabel("d$\psi$")

    return None

if __name__ == '__main__':
    path = os.getcwd()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    X, Y, U = load_data(path)

    fig = plot_solution(X, Y, U)
    plt.show()