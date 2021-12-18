import os
import sys
import glob

import numpy
import matplotlib.pyplot as plt
import math

#This script plots the scalability analysis for a personal computer used in Section V. D
#The plot for scalability on Alfven was done using the same script with a few labels changed
#Only this script is provided since the two are essentially the same

def load_data(path):

    # Estimate number of processors
    num_procs = len(glob.glob(os.path.join(path, "G-S_*_procs_scaling.txt")))

    # Load all data
    data = []
    # N = numpy.empty(num_procs, dtype=int)
    for i in range(1, num_procs+1):
        data.append(numpy.loadtxt(os.path.join(path, "G-S_%s_procs_scaling.txt" % i)))
    
    data = numpy.array(data)

    return data, num_procs

def plot_solution(u, num_procs, s, p):
    fig = plt.figure()
    fig.set_figwidth(fig.get_figwidth() * 1)
    
    axes = fig.add_subplot(1, 1, 1)
    plot = axes.plot(range(1, num_procs+1), u, 'b')
    axes.set_title("Scaling Behavior on Personal Laptop")
    axes.set_xlabel("Process Count")
    axes.set_ylabel("Time (ms)")

    diff_min = 1e+30
    s_min = 1.0
    for s_0 in s:
        diffs = u - u[0] / (1./(s_0 + (1 - s_0) / p))
        diff = math.sqrt(sum(diffs**2.))
        if (diff < diff_min):
            diff_min = diff
            s_min = s_0


    axes.plot(p, u[0] / (1./(s_min + (1 - s_min) / p)), 'r')
    axes.legend(['Scaling Behavior', 'Amdahl: s = 0.22'])
    print('s_min = ' + str(s_min))
    print('diff_min = ' + str(diff_min))

    return None



if __name__ == '__main__':
    path = os.getcwd()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    u, num_procs = load_data(path)

    p = numpy.array([1,2,3,4,5,6])
    s = numpy.linspace(0.0,1.0,100)

    fig = plot_solution(u, num_procs, s, p)
    plt.show()