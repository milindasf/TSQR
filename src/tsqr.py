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
#import cupy as cp
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--rows", help="number of rows", default=1000, type=int, dest="M")
    parser.add_argument("-n", "--cols", help="number of cols", default=20, type=int,  dest="N")
    args = parser.parse_args()

    M=args.M
    N=args.N

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    npes = comm.Get_size()

    A = partitioned_rand_mat(M,N,comm)
    A = gather_mat(A,comm)
    # if(not rank):
    #     print(A)

    Ar = scatter_mat(A,comm)
    #print("rank %d : \n mat %s" %(rank,Ar))

    [Qr,Rr] = tsqr(Ar,comm)
    #print("rank %d \n Q%s \n R %s" %(rank,Qr,Rr))
    #Qg = gather_mat(Q,comm)
    #if(rank==0):
    #    print(A-np.dot(Qg,R))
    is_valid = check_result(Ar,Qr,Rr,comm)
    if(not rank):
        print("QR result is valid :  %s " %is_valid)



