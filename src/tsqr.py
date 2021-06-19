"""
@author: Milinda Fernado
@institute: Oden Institute, University of Texas, Austin. 
@brief: Performs parallel distributed memory A = QR factorization
of a tall-skinny matrix. 
https://www.cs.cornell.edu/~arb/papers/mrtsqr-bigdata2013.pdf 
"""

"""
Note: Since we are not using indirect QR factorization we don't need to
the assumption that A is full rank, I am not sure why the paper assumes that. 
"""

import numpy as np
from mpi4py import MPI
import argparse
import scipy 
import cupy as cp
from time import perf_counter as time

'''
@brief: Performs exclusive scan on array
'''
# def scan(x,mode='e'):
#     x  = list(x)
#     xs = list(x)
#     if(mode=='e'):
#         xs[0] = 0
#         for i in range(1,len(x)):
#             xs[i] = xs[i-1] + x[i-1]
#     else:
#         print("scan mode not supported")
#     return xs

    

'''
@brief generates a partitioned random matrix
matrix is partitioned row wise
@param m - global number of rows
@param n - global number of 
'''
def partitioned_rand_mat(m,n,comm):
    rank = comm.Get_rank()
    npes = comm.Get_size()

    rb = ((rank) * m ) // npes
    re = ((rank+1)  * m ) //npes
    num_rows= re-rb
    return np.random.rand(num_rows,n)

'''
@brief Gather a row-wise partitioned matrix. 
@param Ar: row wise partitioned matrix. 
@param comm: MPI communicator
Note: Assumes that matrix is row-wise partitioned. 
'''
def gather_mat(Ar,comm,root=0):
    rank  = comm.Get_rank()
    rl    = Ar.shape[0]
    rows  = comm.reduce(rl,root=0,op=MPI.SUM)
    cols = Ar.shape[1]

    ss_counts = rl*cols
    ss_counts = comm.allgather(ss_counts)

    A = None

    if(rank == root):
        A = np.zeros((rows,cols),dtype=Ar.dtype)
        #print("rows %d cols %d" %(rows,cols))
    
    comm.Gatherv(Ar,(A,ss_counts),root=root)
    return A

'''
@brief row-wise scatter matrix A onto all ranks.
@param A : gathered matrix
@param comm : MPI Communicator
@param root : root where gathered matrix A lives
'''
def scatter_mat(A,comm,root=0):
    rank = comm.Get_rank()
    npes = comm.Get_size()

    rc = None
    data_type = None

    if(rank==root):
        rc=[A.shape[0], A.shape[1]]
        data_type = A.dtype
    
    data_type=comm.bcast(data_type,root=root)
    rc=comm.bcast(rc,root=root)

    rows = rc[0]
    cols = rc[1]

    # print(rc)
    # print(data_type)
    
    ## row begin and end for the scattering
    rb = (rank * rows)//npes
    re = ((rank+1)*rows)//npes

    ss_counts = (re-rb)*cols
    ss_counts = comm.allgather(ss_counts)

    Ar = np.zeros(((re-rb),cols),dtype=data_type)
    comm.Scatterv((A,ss_counts),Ar,root=root)
    return Ar


'''
@brief Performs tall-skinny recursive QR factorization. 
Let A be mxn matrix, m>>n and compute A= Q*R where, 
Q is mxn orthonomal matrix, and R is a nxn matrix. 

Ar : partitioned matrix of m/p x n
To avoid recursive calling, currently we assume that, 
np x n can be gathered in a single processor.

Root processor for scatter/gather assume to be rank 0
'''

'''
@brief : check the computed QR without gather to a single process.

'''

def check_result(Ar,Qr,Rr,comm):

    rank = comm.Get_rank()
    npes = comm.Get_size()

    # check the A=QR
    is_prod_valid = np.allclose(Ar,np.matmul(Qr,Rr))
    is_prod_valid = comm.reduce(is_prod_valid,root=0,op=MPI.LAND)

    # Check orthonormal without gathering the matrix (does distributed MatMatMult and we don't need to store the result)
    # compute the Q^T x Q in the distributed manner. 
    #QTQ = np.zeros((Qr.shape[1],Qr.shape[1]),dtype=Qr.dtype)
    vec_temp   = np.zeros((1,Qr.shape[1]), dtype=Qr.dtype)
    vec_temp_g = np.zeros((1,Qr.shape[1]), dtype=Qr.dtype)
    eye_i_vec  = np.zeros((1,Qr.shape[1]), dtype=Qr.dtype)
    #QrT = Qr.transpose()

    is_Q_valid = True
    for col in range(0,Qr.shape[1]):
        # each col in the resultant matrix is the span of the QrT cols with col coefficients of the Qr matrix followed by a reduction operator. 
        vec_temp[0,:]    = 0.0
        eye_i_vec[0,:]   = 0.0
        eye_i_vec[0,col] = 1.0
        for i in range(0,Qr.shape[0]):
            vec_temp[0,:] = vec_temp[0,:] + (Qr[i,:] * Qr[i,col])
            
        comm.Reduce(vec_temp,vec_temp_g,root=0,op=MPI.SUM)
        if(not rank):
            if(not np.allclose(vec_temp_g,eye_i_vec)):
                print("Q is not orthonomal: %d ^th row is : %s" %(col,vec_temp_g))
                is_Q_valid = False
                break
    
    # Check R is upper triangular
    is_upper_R = np.allclose(Rr, np.triu(Rr))
    is_upper_R = comm.reduce(is_upper_R,root=0,op=MPI.LAND)

    is_valid =  (is_prod_valid and is_Q_valid and is_upper_R)
    return is_valid

def tsqr(Ar,comm):

    rank = comm.Get_rank()
    npes = comm.Get_size()

    Q2=None
    R2=None

    # local QR computation. 
    [Qr,Rr] =np.linalg.qr(Ar,mode='reduced')
    
    rows = Rr.shape[0]
    cols = Rr.shape[1]
    rows=comm.allreduce(rows,op=MPI.SUM)
    Rr_global_sz = rows*cols
    #print(Rr_global_sz)

    # round 2 qr with gather, 
    # Note : if Rr cannot be gatthered to 
    R = gather_mat(Rr,comm)
    if(rank==0):
        [Q2,R2] = np.linalg.qr(R,mode='reduced')
        

    #bcast computed R2, R = Q2 x R2
    R2 = comm.bcast(R2)

    # scatter the Q2 and compute the corrected Q for A. 
    Q2r = scatter_mat(Q2,comm)
    #print("Q2r : %s" %Q2r)
    Q = np.matmul(Qr,Q2r)
    # if(rank ==0):
    #     print("Q : %s" %Q)
    return [Q,R2]


def tsqr_gpu(Ar, comm):

    rank = comm.Get_rank()
    npes = comm.Get_size()

    Q=None
    Q2=None
    R=None

    num_devices=cp.cuda.runtime.getDeviceCount()
    dev_id = rank % num_devices
    cp.cuda.Device(dev_id).use()
    #print(num_devices)
    #print("rank: %d using %d" %(rank,cp.cuda.Device(dev_id).id))

    timer = {"H2D":0,"D2H":0, "COMPUTE":0}

    # H2D transfer
    start = time()
    A_GPU = cp.array(Ar)
    cp.cuda.stream.get_current_stream().synchronize()
    end = time()
    timer["H2D"] += (end-start)
    
    start = time()
    [Q1, R1] = cp.linalg.qr(A_GPU,mode='reduced')
    end = time()
    timer["COMPUTE"] += (end-start)

    # D2H transfer only R1 for the second pass of QR
    start = time()
    R1 = cp.asnumpy(R1)
    cp.cuda.stream.get_current_stream().synchronize()
    end = time()
    timer["D2H"] += (end-start)

    # gather R1 to rank 0 (root) processor
    R1g = gather_mat(R1,comm)
    if(not rank):
        # dump R1 again to a Root proc. device
        start = time()
        R1g_GPU   = cp.array(R1g)
        cp.cuda.stream.get_current_stream().synchronize()
        end = time()
        timer["H2D"] += (end-start)
        
        # compute second QR
        start = time()
        [Q2, R2] = cp.linalg.qr(R1g_GPU,mode='reduced')
        end = time()
        timer["COMPUTE"] += (end-start)

        start = time()
        Q2  = cp.asnumpy(Q2)
        R   = cp.asnumpy(R2)
        cp.cuda.stream.get_current_stream().synchronize()
        end = time()
        timer["D2H"] += (end-start)

    R = comm.bcast(R)
    Q2r = scatter_mat(Q2,comm)
    
    start = time()
    Q2r_GPU = cp.array(Q2r)
    cp.cuda.stream.get_current_stream().synchronize()
    start = time()
    timer["H2D"] += (end-start)

    start = time()
    Q = cp.matmul(Q1,Q2r_GPU)
    end = time()
    timer["COMPUTE"] += (end-start)

    start = time()
    Q = cp.asnumpy(Q)
    cp.cuda.stream.get_current_stream().synchronize()
    end  = time()
    timer["H2D"] += (end-start)

    #print("Q\n",Q)
    #print("R\n",R)
    return [Q,R,timer]


def tsqr_driver(comm):

    rank = comm.Get_rank()
    npes = comm.Get_size()

    if MODE == "MPI":
        for iter in range(0,WARMUP + ITERS):
            # generate a partitioned random matrix. 
            Ar = partitioned_rand_mat(NROWS,NCOLS,comm)
            
            # Only to debug
            #A = gather_mat(A,comm)
            # if(not rank):
            #     print(A)
            #Ar = scatter_mat(A,comm)
            #print("rank %d : \n mat %s" %(rank,Ar))

            start = time()
            [Qr,Rr] = tsqr(Ar,comm)
            end =time()
            
            if ((iter >= WARMUP)):
                t_dur = end-start
                t_min = comm.reduce(t_dur,root=0,op=MPI.MIN)
                t_max = comm.reduce(t_dur,root=0,op=MPI.MAX)
                if(not rank):
                    print(f'(t_min,t_max):\t\"({t_max},{t_min})\"\n', end='')
            
            #print("rank %d \n Q%s \n R %s" %(rank,Qr,Rr))
            #Qg = gather_mat(Q,comm)
            #if(rank==0):
            #    print(A-np.dot(Qg,R))

            if CHECK_RESULT:
                is_valid = check_result(Ar,Qr,Rr,comm)
                if(not rank):
                    if is_valid:
                        print("\nCorrect result!\n")
                    else:
                        print("%***** ERROR: Incorrect final result!!! *****%")

    elif MODE == "MPI+CUDA":
        
        for iter in range(0,WARMUP+ITERS):
            
            # gen. partitioned matrix. 
            Ar = partitioned_rand_mat(NROWS,NCOLS,comm)

            start = time()
            [Qr,Rr,_] = tsqr_gpu(Ar,comm)
            end =time()
            
            if ((iter >= WARMUP)):
                t_dur = end-start
                t_min = comm.reduce(t_dur,root=0,op=MPI.MIN)
                t_max = comm.reduce(t_dur,root=0,op=MPI.MAX)
                if(not rank):
                    print(f'(t_min,t_max):\t\"({t_max},{t_min})\"\n', end='')
                    print("internal_timers %s"%_)
            
            if CHECK_RESULT:
                is_valid = check_result(Ar,Qr,Rr,comm)
                if(not rank):
                    if is_valid:
                        print("\nCorrect result!\n")
                    else:
                        print("%***** ERROR: Incorrect final result!!! *****%")
        
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rows", help="Number of rows for input matrix; must be >> cols", type=int, default=5000)
    parser.add_argument("-c", "--cols", help="Number of columns for input matrix", type=int, default=100)
    parser.add_argument("-i", "--iterations", help="Number of iterations to run experiment. If > 1, first is ignored as warmup.", type=int, default=1)
    parser.add_argument("-w", "--warmup", help="Number of warmup runs to perform before iterations.", 
    type=int, default=0)
    parser.add_argument("-M", "--mode", help="execution mode: MPI, MPI+GPU",type=str,default="MPI")
    parser.add_argument("-K", "--check_result", help="Checks final result on CPU", action="store_true")
    args = parser.parse_args()

    
    NROWS=args.rows
    NCOLS=args.cols
    ITERS = args.iterations
    WARMUP = args.warmup
    MODE = args.mode
    CHECK_RESULT = args.check_result
    

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    npes = comm.Get_size()

    if(not rank):
        print('%**********************************************************************************************%\n')
        print('Config: rows=', NROWS, ' cols=', NCOLS, ' iterations=', ITERS, ' warmup=', WARMUP, ' check_result=', CHECK_RESULT, sep='', end='\n\n')

    comm.barrier()
    tsqr_driver(comm)

    

        



