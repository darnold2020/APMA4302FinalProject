import os
import sys
import glob

import numpy
import matplotlib.pyplot as plt
import math

#This script generates the plot with errorbars comparing the negligible difference in completion time
#between the two partition approaches discussed in the Appendix

def plot_solution():
    fig = plt.figure()
    fig.set_figwidth(fig.get_figwidth() * 1)

    rowwise = [12481, 12325, 12097, 12527, 12206] #hard-coded results in milliseconds
    gridwise = [12616, 12367, 12662, 12114, 12233]

    urowwise = numpy.average(rowwise)
    ugrid = numpy.average(gridwise)

    urowwise_std = numpy.std(rowwise)
    ugridwise_std = numpy.std(gridwise)

    u = [urowwise, ugrid]
    u_std = [urowwise_std, ugridwise_std]
    
    axes = fig.add_subplot(1, 1, 1)
    plot = axes.errorbar(['row-wise', 'grid-wise'], u, yerr = u_std, marker = 'p', mfc = 'k', mec = 'k', linestyle = '', ecolor = 'k')
    axes.set_title("Time to Complete Nonlinear Simulation")
    axes.set_xlabel("Type of Partition")
    axes.set_ylabel("Time (ms)")

    return None

if __name__ == '__main__':
    path = os.getcwd()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    # u, num_procs = load_data(path)

    fig = plot_solution()
    plt.show()