// #include <omp.h>
#include "mpi.h"

#include <iostream>
#include <fstream>
using namespace std;

#include <math.h>
#include <chrono>

int main(int argc, char *argv[])
{
    auto start = chrono::high_resolution_clock::now();
    //constants
    double const pi = 3.1415926535;
    double const mu_0 = 4.0 * pi * 1e-7;
    double const B = -6. * 50000000.;

    //iteration and problem parameters
    int const MAX_ITER = pow(2, 16), PRINT_INT = 1;
    int N, k, n; //k for Picard iteration, n for Jacobi iteration index
    double dX, tolerance, dPsi_prev_max, dPsi_old_max;
    double term1, term2, RHS;
    double C;
    double Psi_max;
    double r;

    //MPI variables (process specific)
    int num_procs, rank, tag, rank_N, start_index_h, end_index_h, start_index_v, end_index_v;
    double dPsi_prev_max_proc, dPsi_old_max_proc, Psi_max_proc;

    MPI_Status status;
    MPI_Request request;

    //initialize MPI as in homework
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    N = 50;
    tolerance = 1e-8;
    dX = 1. / ((double) (N + 1.));

    int sqrt_num_procs;
    sqrt_num_procs = sqrt(num_procs);

    rank_N = N / sqrt(num_procs);
    start_index_h = (rank % sqrt_num_procs) * rank_N + 1;
    end_index_h = fmin(((rank % sqrt_num_procs) + 1) * rank_N, N);
    start_index_v = (rank / sqrt_num_procs) * rank_N + 1;
    end_index_v = fmin(((rank / sqrt_num_procs) + 1) * rank_N, N);
    //redo rank_N into rank_N_h and rank_N_v?

    //add send and receive buffers to work arrays
    double *send_buffer = new double[rank_N];
    double *recv_buffer = new double[rank_N];
    double **Psi = new double*[rank_N + 2];
    double **Psi_old = new double*[rank_N + 2];
    double **Psi_prev = new double*[rank_N + 2];
    double **R = new double*[rank_N + 2];
    for (int i = 0; i < N + 2; ++i)
    {
        Psi[i] = new double[rank_N + 2]; //now there are only rank_N rows of Psi per process
        Psi_old[i] = new double[rank_N + 2];
        Psi_prev[i] = new double[rank_N + 2];
        R[i] = new double[rank_N + 2];
    }

    //set initial guesses before Picard iteration
    C = 1.0;
    for (int i = 0; i < rank_N + 2; ++i)
    {
        r = dX * (double) (i + start_index_h);
        for (int j = 0; j < rank_N + 2; ++j)
        {
            Psi[i][j] = 1.0;
            R[i][j] = r;
        }
    }

    //may need to set boundary conditions (Dirichlet = 0)
    //set top
    if (rank / sqrt_num_procs == 0)
    {
        for (int i = 0; i < rank_N + 2; ++i)
        {
            Psi[i][0] = 0.0;
        }
    }
    //set bottom
    if (rank / sqrt_num_procs == sqrt_num_procs - 1)
    {
        for (int i = 0; i < rank_N + 2; ++i)
        {
            Psi[i][rank_N+1] = 0.0;
        }
    }
    //set left
    if (rank % sqrt_num_procs == 0)
    {
        for (int j = 0; j < rank_N + 2; ++j)
        {
            Psi[0][j] = 0.0;
        }
    }
    //set right
    if (rank % sqrt_num_procs == sqrt_num_procs - 1)
    {
        for (int j = 0; j < rank_N + 2; ++j)
        {
            Psi[rank_N+1][j] = 0.0;
        }
    }

    //copy into old versions of solution
    for (int i = 0; i < rank_N + 2; ++i)
    {
        for (int j = 0; j < rank_N + 2; ++j)
        {
            Psi_old[i][j] = Psi[i][j];
            Psi_prev[i][j] = Psi[i][j];
        }
    }

    //begin Picard iteration
    k = 0;
    while (k <= MAX_ITER)
    {
        k++;
        dPsi_prev_max_proc = 0.0;
        dPsi_prev_max = 0.0;
        n = 0;
            
        //enter Jacobi loop to solve matrix problem
        while (n <= MAX_ITER)
        {
            n++;
            dPsi_old_max_proc = 0.0;
            dPsi_old_max = 0.0;
            Psi_max_proc = 0.0;

            for (int i = 1; i < rank_N + 1; ++i)
            {
                for (int j = 1; j < rank_N + 1; ++j)
                {
                    term1 = mu_0 * R[i][j] * R[i][j] * B * Psi_prev[i][j] * Psi_prev[i][j] * (1./3. * Psi_prev[i][j] - 1./2.);
                    term2 = C * C * Psi_prev[i][j];
                    RHS = term1 + term2;
                    Psi[i][j] = (1./4.) * (Psi_old[i+1][j] + Psi_old[i-1][j] + Psi_old[i][j+1] + Psi_old[i][j-1]) - (dX / (8. * R[i][j])) * (Psi_old[i+1][j] - Psi_old[i-1][j]) + (dX * dX / 4.) * (RHS);
                    dPsi_old_max_proc = fmax(dPsi_old_max_proc, fabs(Psi[i][j] - Psi_old[i][j]));
                    if (Psi[i][j] > Psi_max_proc)
                    {
                        Psi_max_proc = Psi[i][j];
                    }
                }
            }

            MPI_Allreduce(&dPsi_old_max_proc, &dPsi_old_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD);

            if (n % PRINT_INT == 0 && rank == 0)
                cout << "After " << n << " iterations, dPsi_old_max = " << dPsi_old_max << "\n";

            if (dPsi_old_max < tolerance)
                break;

            //copy data into Psi_old
            for (int i = 1; i < rank_N + 1; ++i)
            {
                for (int j = 1; j < rank_N + 1; ++j)
                {
                    Psi_old[i][j] = Psi[i][j];
                }
            }

            //if Jacobi loop unbroken, communicate data between processes, similar to row-wise Jacobi scheme from class
            //send data down (tag = 1)
            if (rank / sqrt_num_procs != sqrt_num_procs - 1)
            {
                for (int i = 1; i < rank_N + 1; ++i)
                {
                    send_buffer[i-1] = Psi[i][rank_N];
                }
                MPI_Isend(send_buffer, rank_N, MPI_DOUBLE_PRECISION, rank + sqrt_num_procs, 1, MPI_COMM_WORLD, &request);
            }

            //send data up (tag = 2)
            if (rank / sqrt_num_procs != 0) 
            {
                for (int i = 1; i < rank_N + 1; ++i)
                {
                    send_buffer[i-1] = Psi[i][1];
                }
                MPI_Isend(send_buffer, rank_N, MPI_DOUBLE_PRECISION, rank - sqrt_num_procs, 2, MPI_COMM_WORLD, &request);
            }

            //send data right (tag = 3)
            if (rank % sqrt_num_procs != sqrt_num_procs - 1)
            {
                for (int i = 1; i < rank_N + 1; ++i)
                {
                    send_buffer[i-1] = Psi[rank_N][i];
                }
                MPI_Isend(send_buffer, rank_N, MPI_DOUBLE_PRECISION, rank + 1, 3, MPI_COMM_WORLD, &request);
            }
            
            //send data left (tag = 4)
            if (rank % sqrt_num_procs != 0)
            {
                for (int i = 1; i < rank_N + 1; ++i)
                {
                    send_buffer[i-1] = Psi[1][i];
                }
                MPI_Isend(send_buffer, rank_N, MPI_DOUBLE_PRECISION, rank - 1, 4, MPI_COMM_WORLD, &request);
            }

            //receive data from below (tag = 2)
            if (rank / sqrt_num_procs != sqrt_num_procs - 1)
            {
                MPI_Recv(recv_buffer, rank_N, MPI_DOUBLE_PRECISION, rank + sqrt_num_procs, 2, MPI_COMM_WORLD, &status);
                for (int i = 1; i < rank_N + 1; ++i)
                {
                    Psi_old[i][rank_N+1] = recv_buffer[i-1];
                }
            }

            //receive data from above (tag = 1)
            if (rank / sqrt_num_procs != 0)
            {
                MPI_Recv(recv_buffer, rank_N, MPI_DOUBLE_PRECISION, rank - sqrt_num_procs, 1, MPI_COMM_WORLD, &status);
                for (int i = 1; i < rank_N + 1; ++i)
                {
                    Psi_old[i][0] = recv_buffer[i-1];
                }
            }

            //receive data from right (tag = 4)
            if (rank % sqrt_num_procs != sqrt_num_procs - 1)
            {
                MPI_Recv(recv_buffer, rank_N, MPI_DOUBLE_PRECISION, rank + 1, 4, MPI_COMM_WORLD, &status);
                for (int i = 1; i < rank_N + 1; ++i)
                {
                    Psi_old[rank_N+1][i] = recv_buffer[i-1];
                }
            }
            
            //receive data from left (tag = 3)
            if (rank % sqrt_num_procs != 0)
            {
                MPI_Recv(recv_buffer, rank_N, MPI_DOUBLE_PRECISION, rank - 1, 3, MPI_COMM_WORLD, &status);
                for (int i = 1; i < rank_N + 1; ++i)
                {
                    Psi_old[0][i] = recv_buffer[i-1];
                }
            }
        }

        MPI_Allreduce(&Psi_max_proc, &Psi_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD);
        for (int i = 1; i < rank_N + 1; ++i)
        {
            for (int j = 1; j < rank_N + 1; ++j)
            {
                Psi[i][j] = Psi[i][j] / Psi_max;
            }
        }
        C = C / (sqrt(Psi_max));

        for (int i = 1; i < rank_N + 1; ++i)
        {
            for (int j = 1; j < rank_N + 1; ++j)
            {
                dPsi_prev_max_proc = fmax(dPsi_prev_max_proc, fabs(Psi[i][j] - Psi_prev[i][j]));
            }
        }

        MPI_Allreduce(&dPsi_prev_max_proc, &dPsi_prev_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD);

        if (k % PRINT_INT == 0 && rank == 0)
            cout << "After " << k << " iterations, dPsi_prev_max = " << dPsi_prev_max << "\n";
        
        if (dPsi_prev_max < tolerance)
            break;
        //if Picard loop unbroken, copy Psi into Psi_prev
        for (int i = 1; i < rank_N + 1; ++i)
        {
            for (int j = 1; j < rank_N + 1; ++j)
            {
                Psi_prev[i][j] = Psi[i][j];
            }
        }

        //send data down (tag = 5)
        if (rank / sqrt_num_procs != sqrt_num_procs - 1) //exclude first rank
        {
            for (int i = 1; i < rank_N + 1; ++i)
            {
                send_buffer[i-1] = Psi_prev[i][rank_N];
            }
            MPI_Isend(send_buffer, rank_N, MPI_DOUBLE_PRECISION, rank + sqrt_num_procs, 5, MPI_COMM_WORLD, &request);
        }

        //send data up (tag = 6)
        if (rank / sqrt_num_procs != 0) //exclude last rank
        {
            for (int i = 1; i < rank_N + 1; ++i)
            {
                send_buffer[i-1] = Psi_prev[i][1];
            }
            MPI_Isend(send_buffer, rank_N, MPI_DOUBLE_PRECISION, rank - sqrt_num_procs, 6, MPI_COMM_WORLD, &request);
        }

        //send data right (tag = 7)
        if (rank % sqrt_num_procs != sqrt_num_procs - 1)
        {
            for (int i = 1; i < rank_N + 1; ++i)
            {
                send_buffer[i-1] = Psi_prev[rank_N][i];
            }
            MPI_Isend(send_buffer, rank_N, MPI_DOUBLE_PRECISION, rank + 1, 7, MPI_COMM_WORLD, &request);
        }

        //send data left (tag = 8)
        if (rank % sqrt_num_procs != 0)
        {
            for (int i = 1; i < rank_N + 1; ++i)
            {
                send_buffer[i-1] = Psi_prev[1][i];
            }
            MPI_Isend(send_buffer, rank_N, MPI_DOUBLE_PRECISION, rank - 1, 8, MPI_COMM_WORLD, &request);
        }

        // cout << rank << " here\n";
        //receive data from below (tag = 6)
        if (rank / sqrt_num_procs != sqrt_num_procs - 1)
        {
            MPI_Recv(recv_buffer, rank_N, MPI_DOUBLE_PRECISION, rank + sqrt_num_procs, 6, MPI_COMM_WORLD, &status);
            for (int i = 1; i < rank_N + 1; ++i)
            {
                Psi_prev[i][rank_N+1] = recv_buffer[i-1];
            }
        }

        //receive data from above (tag = 5)
        if (rank / sqrt_num_procs != 0)
        {
            MPI_Recv(recv_buffer, rank_N, MPI_DOUBLE_PRECISION, rank - sqrt_num_procs, 5, MPI_COMM_WORLD, &status);
            for (int i = 1; i < rank_N + 1; ++i)
            {
                Psi_prev[i][0] = recv_buffer[i-1];
            }
        }

        //receive data from left (tag = 7)
        if (rank % sqrt_num_procs != 0)
        {
            MPI_Recv(recv_buffer, rank_N, MPI_DOUBLE_PRECISION, rank - 1, 7, MPI_COMM_WORLD, &status);
            for (int i = 1; i < rank_N + 1; ++i)
            {
                Psi_prev[0][i] = recv_buffer[i-1];
            }
        }

        //receive data from right (tag = 8)
        if (rank % sqrt_num_procs != sqrt_num_procs - 1)
        {
            MPI_Recv(recv_buffer, rank_N, MPI_DOUBLE_PRECISION, rank + 1, 8, MPI_COMM_WORLD, &status);
            for (int i = 1; i < rank_N + 1; ++i)
            {
                Psi_prev[rank_N+1][i] = recv_buffer[i-1];
            }
        }
    }

    //Check output and write out results
    //Check for failure like example code
    if (k >= MAX_ITER)
    {
        if (rank == 0)
        {
            cout << "*** Captain! Picard failed to converge!\n";
            cout << "Reached dPsi_prev_max = " << dPsi_prev_max << "\n";
            cout << "Tolerance was = " << tolerance << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    string file_name = "G-S_square_" + to_string(rank) + "_local.txt";
    ofstream fp(file_name);

    if (rank % sqrt_num_procs != sqrt_num_procs - 1 && rank / sqrt_num_procs != sqrt_num_procs - 1)
    {
        for (int i = 0; i < rank_N + 1; ++i)
        {
            for (int j = 0; j < rank_N + 1; ++j)
            {
                fp << Psi[i][j] << " ";
            }
            fp << "\n";
        }
    }
    else if (rank % sqrt_num_procs != sqrt_num_procs - 1 && rank / sqrt_num_procs == sqrt_num_procs - 1)
    {
        for (int i = 0; i < rank_N + 1; ++i)
        {
            for (int j = 1; j < rank_N + 2; ++j)
            {
                fp << Psi[i][j] << " ";
            }
            fp << "\n";
        }
    }
    else if (rank % sqrt_num_procs == sqrt_num_procs - 1 && rank / sqrt_num_procs != sqrt_num_procs - 1)
    {
        for (int i = 1; i < rank_N + 2; ++i)
        {
            for (int j = 0; j < rank_N + 1; ++j)
            {
                fp << Psi[i][j] << " ";
            }
            fp << "\n";
        }
    }
    else
    {
        for (int i = 1; i < rank_N + 2; ++i)
        {
            for (int j = 1; j < rank_N + 2; ++j)
            {
                fp << Psi[i][j] << " ";
            }
            fp << "\n";
        }
    }


    fp.close();

    MPI_Finalize();

    auto stop = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    if (rank == 0)
    {
        cout << duration.count() << " milliseconds to run.\n";
        string time_name = "G-S_square_" + to_string(num_procs) + "_procs.txt";
        ofstream fp(time_name);
        fp << duration.count();
        fp.close();
    }

    return 0;
}