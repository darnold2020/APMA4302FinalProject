import os
import sys
import glob

import numpy
import matplotlib.pyplot as plt
import math

import scipy
from scipy.special import jv, jn_zeros

#plot the linear computed solution, analytic solution, error field between the two, and error convergence behavior

def load_data(path, m):

    # Estimate number of processors
    num_procs = len(glob.glob(os.path.join(path, "G-S_row_linear_%s_*.txt" % m)))

    # Load all data
    data = []
    rank_N = numpy.empty(num_procs, dtype=int)
    # N = numpy.empty(num_procs, dtype=int)
    for i in range(num_procs):
        data.append(numpy.loadtxt(os.path.join(path, "G-S_row_linear_%s_%s.txt" % (m, i))))
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
    axes.set_title("Computed Solution")
    axes.set_xlabel("R Coordinate")
    axes.set_ylabel("Z Coordinate")

    m = 100

    r = numpy.linspace(0.0,1.0,m+2)
    z = numpy.linspace(-0.5,0.5,m+2)
    delta_X = 1.0 / (m+1)
    onesVec = numpy.ones(m)

    #grab R and Z values inside ghost boundary cells, which are pinned to 0
    R = r[1:-1]
    Z = z[1:-1]
    R = r
    Z = z

    [Rmesh, Zmesh] = numpy.meshgrid(R, Z)

    #eval_spheromak code requires the total length in the R and Z direction, along with R,Z mesh from earlier
    R0 = 1.0
    H0 = 1.0

    #calculate and return the analytic B and Psi fields
    B_an, Psi_an = eval_spheromak(Rmesh, Zmesh, R0, H0)

    fig2 = plt.figure()
    axes = fig2.add_subplot(1,1,1)
    cont = axes.contourf(Rmesh, Zmesh, abs(Psi_an), 20, cmap = 'RdBu_r')
    fig2.colorbar(cont)

    axes.set_title('Magnitude of $\psi$ In R-Z Plane')
    axes.set_xlabel('R Coordinate')
    axes.set_ylabel('Z Coordinate')

    fig3 = plt.figure()
    axes = fig3.add_subplot(1,1,1)
    cont = axes.contourf(Rmesh, Zmesh, abs(Psi_an - u), 20, cmap = 'RdBu_r')
    fig3.colorbar(cont)

    return None

def eval_spheromak(R, Z, R0, H0):
    # Compute normalization values for given geometry
    # Note: Assumes symmetry in z-direction (ie. -min(Z) = max(Z) = H0/2.0)
    kr = jn_zeros(1,1)/R0
    kz = numpy.pi/H0
    lam = numpy.sqrt(kr*kr + kz*kz)
    # print(lam)
    # Compute fields on R, Z grid
    B = numpy.zeros((3, R.shape[0], R.shape[1]))
    Psi = numpy.zeros(R.shape)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            ar = kr*R[i,j]
            az = kz*(Z[i,j]-H0/2.0)
            if ar == 0.0:
                tmp = 0.5
            else:
                tmp = jv(1,ar)/ar
            B[0,i,j] = kz*R[i,j]*numpy.cos(az)*tmp
            B[1,i,j] = -lam*R[i,j]*numpy.sin(az)*tmp
            B[2,i,j] = -jv(0,ar)*numpy.sin(az)
            Psi[i,j] = -numpy.power(R[i,j],2)*numpy.sin(az)*tmp # CJH
            #Normalizing to Psi = 1
    psi_max = Psi.max(axis=0).max() # CJH
    if psi_max < 1.E-8: # CJH
        psi_max = Psi.min(axis=0).min() # CJH
    return B / psi_max, Psi / psi_max

def plot_errors(Psi_diffs, ms, delta_Xs):
    fig10 = plt.figure()
    axes = fig10.add_subplot(1,1,1)
    axes.loglog(delta_Xs, find_C(Psi_diffs, delta_Xs, 2) * delta_Xs ** 2.)
    axes.loglog(delta_Xs, find_C(Psi_diffs[0], delta_Xs[0], 2) * delta_Xs ** 2., 'r--')
    axes.loglog(delta_Xs, find_C(Psi_diffs[0], delta_Xs[0], 1) * delta_Xs ** 1., 'g--')
    num_procs = len(glob.glob(os.path.join(path, "G-S_row_linear_%s_*.txt" % m)))
    if (num_procs == 1):
        axes.set_title('$\Psi$ Convergence (Serial Case)')
    else:
        axes.set_title('$\Psi$ Convergence (Parallel Case)')
    axes.set_ylabel('Error (Inf Norm)')
    axes.set_xlabel('$\Delta$ X')
    axes.legend(['Solution', 'Second Order', 'First Order'])
    return None

def find_C(errors, delta_x, order):
    return numpy.exp(numpy.log(errors) - order * numpy.log(delta_x))

if __name__ == '__main__':
    path = os.getcwd()
    if len(sys.argv) > 1:
        path = sys.argv[1]

    Psi_diffs = []
    delta_Xs = []

    ms = [25, 35, 50, 75, 100]

    for m in ms:
        r = numpy.linspace(0.0,1.0,m+2)
        z = numpy.linspace(-0.5,0.5,m+2)
        delta_X = 1.0 / (m+1)
        onesVec = numpy.ones(m)
        #grab R and Z values inside ghost boundary cells, which are pinned to 0
        R = r
        Z = z
        R0 = 1.0
        H0 = 1.0
        [Rmesh, Zmesh] = numpy.meshgrid(R, Z)
        B_an, Psi_an = eval_spheromak(Rmesh, Zmesh, R0, H0)
        X, Y, Psi = load_data(path, m)
        Psi_diff = numpy.linalg.norm((Psi.reshape((m+2)**2) - Psi_an.reshape((m+2)**2)), ord = numpy.inf)
        Psi_diffs.append(Psi_diff)
        delta_Xs.append(delta_X)

    Psi_diffs = numpy.array(Psi_diffs)
    delta_Xs = numpy.array(delta_Xs)
    fig = plot_solution(X, Y, Psi)
    fig10 = plot_errors(Psi_diffs, ms, delta_Xs)
    plt.show()

