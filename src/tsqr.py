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


'''
@brief: Performs exclusive scan on array
'''
def scan(x,mode='e'):
    x  = list(x)
    xs = list(x)
       
    if(mode=='e'):
        xs[0] = 0
        for i in range(1,len(x)):
            xs[i] = xs[i-1] + x[i-1]
    else:
        print("scan mode not supported")
    return xs
    

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
def tsqr(Ar,comm):

    rank = comm.Get_rank()
    npes = comm.Get_size()

    Q2=None
    R2=None

    # local QR computation. 
    [Qr,Rr] =np.linalg.qr(Ar)
    
    rows = Rr.shape[0]
    cols = Rr.shape[1]
    rows=comm.allreduce(rows,op=MPI.SUM)
    Rr_global_sz = rows*cols
    #print(Rr_global_sz)

    # round 2 qr with gather, 
    # Note : if Rr cannot be gatthered to 
    R = gather_mat(Rr,comm)
    if(rank==0):
        [Q2,R2] = np.linalg.qr(R)

    #bcast computed R2, R = Q2 x R2
    R2 = comm.bcast(R2)

    # scatter the Q2 and compute the corrected Q for A. 
    Q2r = scatter_mat(Q2,comm)
    Q = np.dot(Qr,Q2r)
    return [Q,R2]



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
#print("rank %d \n Q%s \n R %s" %(rank,Q,R))
#Qg = gather_mat(Q,comm)
#if(rank==0):
#    print(A-np.dot(Qg,R))
is_valid = np.allclose(Ar,np.dot(Qr,Rr))
is_valid = comm.allreduce(is_valid,op=MPI.LAND)
if(not rank):
    print("QR result is valid :  %s " %is_valid)



