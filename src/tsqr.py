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

from re import A
import numpy as np
from mpi4py import MPI
import argparse
from numpy.core.fromnumeric import sort
import scipy 
import cupy as cp
from time import perf_counter as time
import concurrent.futures
import queue
import os as os


class profile_t:

    def __init__(self,name):
        self.name = name
        self.seconds=0
        self.snap=0
        self._pri_time =0
        self.iter =0

    def __add__(self,o):
        assert(self.name==o.name)
        self.seconds+=o.seconds
        self.snap+=o.snap
        self.iter+=o.iter
        return self

    def start(self):
        self._pri_time = time()
    
    def stop(self):
        self.seconds-=self._pri_time
        self.snap=-self._pri_time

        self._pri_time = time()

        self.seconds +=self._pri_time
        self.snap  += self._pri_time
        self.iter+=1
    
    def reset(self):
        self.seconds=0
        self.snap=0
        self._pri_time =0
        self.iter =0

    


def accumulate_counters(t1,t2,is_sorted=False):
    
    if(not is_sorted):
        t1=sort(t1,key=lambda w: w.name)
        t2=sort(t2,key=lambda w: w.name)
    
    assert (len(t1)== len(t2))
    for i in range(len(t1)):
        assert(t1[i].name ==  t2[i].name)
        t1[i].seconds += t2[i].seconds
        t1[i].snap    += t2[i].snap
        t1[i].iter    += t2[i].iter

    return
        
'''
Computes the row-wise partition bounds. 
'''
def row_partition_bounds(nrows,rank,npes):
    assert(nrows >=npes)
    rb = ((rank) * nrows ) // npes
    re = ((rank+1)  * nrows ) //npes
    return [rb,re]


'''
@brief generates a partitioned random matrix
matrix is partitioned row wise
@param m - global number of rows
@param n - global number of 
'''
def partitioned_rand_mat(m,n,comm):
    rank = comm.Get_rank()
    npes = comm.Get_size()
    [rb,re]=row_partition_bounds(m,rank,npes)
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
    [rb,re]=row_partition_bounds(rows,rank,npes)
    
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

def shared_mem_check_result(Ar,Qr,Rr):
    is_prod_valid = np.allclose(Ar,np.matmul(Qr,Rr))
    is_Q_valid = np.allclose(np.eye(Qr.shape[1]),np.matmul(np.transpose(Qr),Qr))
    is_upper_R = np.allclose(Rr, np.triu(Rr))

    is_valid =  (is_prod_valid and is_Q_valid and is_upper_R)
    return is_valid

'''
Pure MPI only TSQR without recursive splitting, 
Recursive splitting will require recursive communicator
splitting, that can be done, but require some work and 
repartitioning of the matrices. 
'''
def tsqr(Ar,comm):

    rank = comm.Get_rank()
    npes = comm.Get_size()

    Q2=None
    R2=None

    t_compute = profile_t("compute")
    t_comm    = profile_t("comm")

    # local QR computation. 
    t_compute.start()
    [Qr,Rr] =np.linalg.qr(Ar,mode='reduced')
    t_compute.stop()

    # rows = Rr.shape[0]
    # cols = Rr.shape[1]
    # rows=comm.allreduce(rows,op=MPI.SUM)
    # Rr_global_sz = rows*cols
    # #print(Rr_global_sz)

    # round 2 qr with gather, 
    # for smaller n assumes, Rr can be gathered to a single process, 
    # Memory cost : npes * (cols^2)
    # Otherwise we can do recursive split which is not implemented. 
    t_comm.start()
    R = gather_mat(Rr,comm)
    t_comm.stop()

    if(rank==0):
        t_compute.start()
        [Q2,R2] = np.linalg.qr(R,mode='reduced')
        t_compute.stop()
    
    #bcast computed R2, R = Q2 x R2
    t_comm.start()
    R = comm.bcast(R2)
    # scatter the Q2 and compute the corrected Q for A. 
    Q2r = scatter_mat(Q2,comm)
    t_comm.stop()

    t_compute.start()
    Q = np.matmul(Qr,Q2r)
    t_compute.stop()

    _perf_stats=[t_comm,t_compute]
    return [Q,R,_perf_stats]


'''
Performes GPU QR decomposition, 
timer values will get accumulated 
loc denotes the final location of the variable.
'H': For host, 'D': For device.  
'''
def block_gpu_qr(Ar,dev_id,loc={'A':'H','Q':'H','R':'H'}):
    
    t_h2d         = profile_t("H2D")
    t_d2h         = profile_t("D2H")
    t_kernel_gpu  = profile_t("kernel_gpu")

    cp.cuda.Device(dev_id).use()
    
    if(loc['A']=='H'):
        # H2D transfer
        t_h2d.start()
        A_GPU = cp.array(Ar)
        cp.cuda.stream.get_current_stream().synchronize()
        t_h2d.stop()
    else:
        A_GPU=Ar
    
    t_kernel_gpu.start()
    [Q1, R1] = cp.linalg.qr(A_GPU,mode='reduced')
    cp.cuda.get_current_stream().synchronize()
    t_kernel_gpu.stop()
    

    if (loc['Q']=='H' or loc['R']=='H'):

        t_d2h.start()
        if(loc['R']=='H'):
            R1 = cp.asnumpy(R1)
    
        if(loc['Q']=='H'):
            Q1 = cp.asnumpy(Q1)
        cp.cuda.stream.get_current_stream().synchronize()
        t_d2h.stop()
    
    _perf_stat= [t_h2d,t_kernel_gpu,t_d2h]
    return [Q1,R1,_perf_stat]

def block_gpu_matmult(Ar,Br, dev_id,loc={'A':'H','B':'H','C':'H'}):
    
    t_h2d         = profile_t("H2D")
    t_d2h         = profile_t("D2H")
    t_kernel_gpu  = profile_t("kernel_gpu")

    cp.cuda.Device(dev_id).use()

    A_GPU=Ar
    B_GPU=Br

    if(loc['A']=='H'  or loc['B']=='H'):
        t_h2d.start()
        if(loc['A']=='H'):
            A_GPU = cp.array(Ar)
        else:
            A_GPU=Ar

        if(loc['B']=='H'):
            B_GPU = cp.array(Br)
        else:
            B_GPU=Br

        cp.cuda.stream.get_current_stream().synchronize()
        t_h2d.stop()
    
    
    t_kernel_gpu.start()
    C_GPU = cp.matmul(A_GPU,B_GPU)
    cp.cuda.get_current_stream().synchronize()
    t_kernel_gpu.stop()

    if(loc['C']=='H'):
        t_d2h.start()
        C = cp.asnumpy(C_GPU)
        cp.cuda.stream.get_current_stream().synchronize()
        t_d2h.stop()
    else:
        C=C_GPU
    
    _perf_stat= [t_h2d,t_kernel_gpu,t_d2h]
    return [C,_perf_stat]

'''
Shared memory scatter
'''
def shared_mem_block_part(A,num_threads):

    [rows,cols] = A.shape
    A_blocked = list()
    for i in range(0,num_threads):
        [rb,re]=row_partition_bounds(rows,i,num_threads)
        A_blocked.append(A[rb:re,:])
    
    assert(len(A_blocked) == num_threads)
    return A_blocked

'''
Shared memory gather
'''
def shared_mem_unblock(A_blocked):
    return np.concatenate(A_blocked)

def tsqr_shared_mem_gpu_v1(A,num_threads):

    A_blocked = shared_mem_block_part(A,num_threads)
    loc={'A':'H','Q':'D','R':'H'}

    t_h2d         = [profile_t("H2D")]*num_threads
    t_d2h         = [profile_t("D2H")]*num_threads
    t_kernel_gpu  = [profile_t("kernel_gpu")]*num_threads

    result = list()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for t_id in range(num_threads):
            result.append(executor.submit(block_gpu_qr,A_blocked[t_id],t_id,loc))

    
    Q1_GPU  = list()
    R1_CPU  = list()
    r1_time = list()

    for m in result:
        #print(m.result())
        Q1_GPU.append (m.result()[0])
        R1_CPU.append (m.result()[1])
        r1_time.append(m.result()[2])
    
    R1=shared_mem_unblock(R1_CPU)
    loc={'A':'H','Q':'D','R':'H'}
    result =list()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for t_id in range(num_threads):
            result.append(executor.submit(block_gpu_qr,R1,t_id,loc))

    Q2_GPU  = list()
    R2_CPU  = list()
    r2_time = list()
    
    for m in result:
        Q2_GPU.append (m.result()[0])
        R = m.result()[1]
        r2_time.append(m.result()[2])

    #print(Q2_GPU)
    #print(R)
    loc={'A':'D','B':'D','C':'H'}
    [rows,cols] = R1.shape
    result=list()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for t_id in range(num_threads):
            # partition boundary. 
            [rb,re]=row_partition_bounds(rows,t_id,num_threads)
            result.append(executor.submit(block_gpu_matmult,Q1_GPU[t_id],Q2_GPU[t_id][rb:re,:],t_id,loc))
            
    r3_time=list()
    Q2_CPU=list()
    for m in result:
        Q2_CPU.append(m.result()[0])
        r3_time.append(m.result()[1])

    for i in range(num_threads):
        t_h2d[i] = r1_time[i][0] + r2_time[i][0] + r3_time[i][0]
        t_kernel_gpu[i] = r1_time[i][1] + r2_time[i][1] + r3_time[i][1]
        t_d2h[i] = r1_time[i][2] + r2_time[i][2] + r3_time[i][2]

    Q=shared_mem_unblock(Q2_CPU)
    _perf_stat = [t_h2d,t_kernel_gpu,t_d2h]
    return [Q,R,_perf_stat]


def tsqr_shared_mem_gpu_v2(A,num_threads):

    A_blocked = shared_mem_block_part(A,num_threads)
    loc={'A':'H','Q':'D','R':'H'}

    t_h2d         = [profile_t("H2D")]*num_threads
    t_d2h         = [profile_t("D2H")]*num_threads
    t_kernel_gpu  = [profile_t("kernel_gpu")]*num_threads
    t_t1_total    = [profile_t("t1_total")]*1
    t_t2_total    = [profile_t("t2_total")]*1
    t_t3_total    = [profile_t("t3_total")]*1
    

    result = list()
    t_t1_total[0].start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for t_id in range(num_threads):
            result.append(executor.submit(block_gpu_qr,A_blocked[t_id],t_id,loc))

    
    Q1_GPU  = list()
    R1_CPU  = list()
    r1_time = list()

    for m in result:
        #print(m.result())
        Q1_GPU.append (m.result()[0])
        R1_CPU.append (m.result()[1])
        r1_time.append(m.result()[2])
    t_t1_total[0].stop()
    
    R1=shared_mem_unblock(R1_CPU)
    loc={'A':'H','Q':'D','R':'H'}
    result =list()

    t_t2_total[0].start()
    [Q2,R] = np.linalg.qr(R1,mode='reduced')
    t_t2_total[0].stop()

    #print(Q2_GPU)
    #print(R)
    loc={'A':'D','B':'H','C':'H'}
    [rows,cols] = R1.shape
    result=list()
    t_t3_total[0].start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for t_id in range(num_threads):
            # partition boundary. 
            [rb,re]=row_partition_bounds(rows,t_id,num_threads)
            result.append(executor.submit(block_gpu_matmult,Q1_GPU[t_id],Q2[rb:re,:],t_id,loc))
            
    r3_time=list()
    Q2_CPU=list()
    for m in result:
        Q2_CPU.append(m.result()[0])
        r3_time.append(m.result()[1])
    t_t3_total[0].stop()
    
    for i in range(num_threads):
        t_h2d[i] = r1_time[i][0] + r3_time[i][0]
        t_kernel_gpu[i] = r1_time[i][1] + r3_time[i][1]
        t_d2h[i] = r1_time[i][2] + r3_time[i][2]

    Q=shared_mem_unblock(Q2_CPU)
    _perf_stat = [t_t1_total,t_t2_total,t_t3_total,t_h2d,t_kernel_gpu,t_d2h]
    return [Q,R,_perf_stat]


'''
Performs multi-gpu on multi-nodes with MPI parallelization. 
Can implement in various versions. 
More duplication less data movement version. 
'''
def tsqr_mpi_gpu_v1(Ar, comm):

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

    t_h2d         = profile_t("H2D")
    t_d2h         = profile_t("D2H")
    t_mpi_comm    = profile_t("mpi_comm")
    t_kernel_gpu  = profile_t("kernel_gpu")
    t_t1          = profile_t("qr1")
    t_t2          = profile_t("qr2")
    t_t3          = profile_t("mm")
    
    # perform blocked QR, 
    t_t1.start()
    [Q1,R1,ts]=block_gpu_qr(Ar,dev_id,loc={'A':'H','Q':'D','R':'H'})
    t_t1.stop()

    t_h2d += ts[0]
    t_kernel_gpu +=ts[1]
    t_d2h += ts[2]

    
    # gather R1 to rank 0 (root) processor
    t_mpi_comm.start()
    R1g = gather_mat(R1,comm)
    R1g = comm.bcast(R1g)
    t_mpi_comm.stop()
    
    t_t2.start()
    [Q2,R,ts]=block_gpu_qr(R1g,dev_id,loc={'A':'H','Q':'D','R':'H'})
    t_t2.stop()

    t_h2d += ts[0]
    t_kernel_gpu += ts[1]
    t_d2h += ts[2]

    [rb,re] = row_partition_bounds(R1g.shape[0],rank,npes)
    Q2r = Q2[rb:re,:]
    
    t_t3.start()
    [Q,ts]=block_gpu_matmult(Q1,Q2r,dev_id,loc={'A':'D','B':'D','C':'H'})
    t_t3.stop()

    t_h2d += ts[0]
    t_kernel_gpu += ts[1]
    t_d2h += ts[2]

    _pref_stat = [t_t1, t_t2, t_t3, t_mpi_comm, t_h2d, t_d2h, t_kernel_gpu]
    
    return [Q,R,_pref_stat]

'''
Performs multi-gpu on multi-nodes with MPI parallelization. 
Can implement in various versions. 
- Round 2 QR factorization happens in the GPU. (I think this should be good for larger np x n)
'''
def tsqr_mpi_gpu_v2(Ar, comm):

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

    t_h2d         = profile_t("H2D")
    t_d2h         = profile_t("D2H")
    t_mpi_comm    = profile_t("mpi_comm")
    t_kernel_gpu  = profile_t("kernel_gpu")
    t_t1          = profile_t("qr1")
    t_t2          = profile_t("qr2")
    t_t3          = profile_t("mm")
    
    # perform blocked QR, 
    t_t1.start()
    [Q1,R1,ts]=block_gpu_qr(Ar,dev_id,loc={'A':'H','Q':'D','R':'H'})
    t_t1.stop()
    
    t_h2d += ts[0]
    t_kernel_gpu +=ts[1]
    t_d2h += ts[2]

    
    # gather R1 to rank 0 (root) processor
    t_mpi_comm.start()
    R1g = gather_mat(R1,comm)
    t_mpi_comm.stop()
    
    t_t2.start()
    if(not rank):
        [Q2,R,ts]=block_gpu_qr(R1g,dev_id,loc={'A':'H','Q':'H','R':'H'})
        t_h2d += ts[0]
        t_kernel_gpu += ts[1]
        t_d2h += ts[2]
    t_t2.stop()
    
        
    t_mpi_comm.start()
    R = comm.bcast(R)
    Q2r = scatter_mat(Q2,comm)
    t_mpi_comm.stop()

    t_t3.start()
    [Q,ts]=block_gpu_matmult(Q1,Q2r,dev_id,loc={'A':'D','B':'H','C':'H'})
    t_t3.stop()

    t_h2d += ts[0]
    t_kernel_gpu += ts[1]
    t_d2h += ts[2]

    _pref_stat = [t_t1, t_t2, t_t3, t_mpi_comm, t_h2d, t_d2h, t_kernel_gpu ]
    
    return [Q,R,_pref_stat]


'''
MPI version 3, 
the second QR is done in the CPU. 
'''
def tsqr_mpi_gpu_v3(Ar, comm):

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

    t_h2d         = profile_t("H2D")
    t_d2h         = profile_t("D2H")
    t_mpi_comm    = profile_t("mpi_comm")
    t_kernel_gpu  = profile_t("kernel_gpu")
    t_t1          = profile_t("qr1")
    t_t2          = profile_t("qr2")
    t_t3          = profile_t("mm")
    
    # perform blocked QR, 
    t_t1.start()
    [Q1,R1,ts]=block_gpu_qr(Ar,dev_id,loc={'A':'H','Q':'D','R':'H'})
    t_t1.stop()

    t_h2d += ts[0]
    t_kernel_gpu +=ts[1]
    t_d2h += ts[2]

    
    # gather R1 to rank 0 (root) processor
    t_mpi_comm.start()
    R1g = gather_mat(R1,comm)
    R1g = comm.bcast(R1g)
    t_mpi_comm.stop()
    
    t_t2.start()
    #[Q2,R,ts]=block_gpu_qr(R1g,dev_id,loc={'A':'H','Q':'D','R':'H'})
    [Q2,R] = np.linalg.qr(R1g,mode='reduced')
    t_t2.stop()

    #t_h2d += ts[0]
    #t_kernel_gpu += ts[1]
    #t_d2h += ts[2]

    [rb,re] = row_partition_bounds(R1g.shape[0],rank,npes)
    Q2r = Q2[rb:re,:]
    
    t_t3.start()
    [Q,ts]=block_gpu_matmult(Q1,Q2r,dev_id,loc={'A':'D','B':'H','C':'H'})
    t_t3.stop()

    t_h2d += ts[0]
    t_kernel_gpu += ts[1]
    t_d2h += ts[2]

    _pref_stat = [t_t1, t_t2, t_t3, t_mpi_comm, t_h2d, t_d2h, t_kernel_gpu]
    
    return [Q,R,_pref_stat]


def tsqr_driver(comm):

    rank = comm.Get_rank()
    npes = comm.Get_size()

    t_header = False

    if MODE == "MPI":
        for iter in range(0,WARMUP + ITERS):
            np.random.seed(iter)
            # generate a partitioned random matrix. 
            Ar = partitioned_rand_mat(NROWS,NCOLS,comm)
            
            # Only to debug
            #A = gather_mat(A,comm)
            # if(not rank):
            #     print(A)
            #Ar = scatter_mat(A,comm)
            #print("rank %d : \n mat %s" %(rank,Ar))
            t_overall = profile_t("total")
            t_overall.start()
            [Qr,Rr,ts] = tsqr(Ar,comm)
            t_overall.stop()
            
            if ((iter >= WARMUP)):
                if(not t_header):
                    header="iter\ttotal_min\ttotal_max"
                    for t in ts:
                        header+="\t"+t.name+"_min"
                        header+="\t"+t.name+"_max"
                    
                    if(not rank):
                        print(header)
                    t_header=True
                
                t_dur = t_overall.seconds
                t_min = comm.reduce(t_dur,root=0,op=MPI.MIN)
                t_max = comm.reduce(t_dur,root=0,op=MPI.MAX)
                if(not rank):
                    t_= f"{iter}\t{t_min:.12f}\t{t_max:.12f}"
                    
                for t in ts:
                    t_dur = t.seconds
                    t_min = comm.reduce(t_dur,root=0,op=MPI.MIN)
                    t_max = comm.reduce(t_dur,root=0,op=MPI.MAX)
                    if(not rank):
                        t_= t_ + f"\t{t_min:.12f}\t{t_max:.12f}"

                
                if(not rank):
                    print(t_)
            
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
            np.random.seed(iter)
            # gen. partitioned matrix. 
            Ar = partitioned_rand_mat(NROWS,NCOLS,comm)

            t_overall = profile_t("total")
            t_overall.start()
            [Qr,Rr,ts] = tsqr_mpi_gpu_v1(Ar,comm)
            t_overall.stop()
            
            if ((iter >= WARMUP)):
                if(not t_header):
                    header="iter\ttotal_min\ttotal_max"
                    for t in ts:
                        header+="\t"+t.name+"_min"
                        header+="\t"+t.name+"_max"
                    
                    if(not rank):
                        print(header)
                    t_header=True
                
                t_dur = t_overall.seconds
                t_min = comm.reduce(t_dur,root=0,op=MPI.MIN)
                t_max = comm.reduce(t_dur,root=0,op=MPI.MAX)
                if(not rank):
                    t_= f"{iter}\t{t_min:.12f}\t{t_max:.12f}"
                    
                for t in ts:
                    t_dur = t.seconds
                    t_min = comm.reduce(t_dur,root=0,op=MPI.MIN)
                    t_max = comm.reduce(t_dur,root=0,op=MPI.MAX)
                    if(not rank):
                        t_= t_ + f"\t{t_min:.12f}\t{t_max:.12f}"

                
                if(not rank):
                    print(t_)
            
            if CHECK_RESULT:
                is_valid = check_result(Ar,Qr,Rr,comm)
                if(not rank):
                    if is_valid:
                        print("\nCorrect result!\n")
                    else:
                        print("%***** ERROR: Incorrect final result!!! *****%")

    elif MODE == "SM+CUDA":
        for iter in range(0,WARMUP+ITERS):
            np.random.seed(iter)
            # gen. partitioned matrix. 
            Ar = partitioned_rand_mat(NROWS,NCOLS,comm)

            t_overall = profile_t("total")
            t_overall.start()
            [Qr,Rr,ts]=tsqr_shared_mem_gpu_v2(Ar,NTHREADS)
            t_overall.stop()
            
            if ((iter >= WARMUP)):
                if(not t_header):
                    header="iter\ttotal\tqr1\tqr2\tmm"
                    for t in ts[3:]:
                        header+="\t"+t[0].name+"_min"
                        header+="\t"+t[0].name+"_max"
                    print(header)
                    t_header=True
                
                t_dur = t_overall.seconds
                t_min = t_dur
                t_max = t_dur
                t_= f"{iter}\t{t_dur:.12f}\t{ts[0][0].seconds:.12f}\t{ts[1][0].seconds:.12f}\t{ts[2][0].seconds:.12f}"
                    
                for t in ts[3:]:
                    t_min = min(t,key=lambda a: a.seconds).seconds
                    t_max = max(t,key=lambda a: a.seconds).seconds
                    t_= t_ + f"\t{t_min:.12f}\t{t_max:.12f}"

                print(t_)

            if CHECK_RESULT:
                is_valid = shared_mem_check_result(Ar,Qr,Rr)
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
    parser.add_argument("-p", "--placement", help="execution mode: MPI, MPI+GPU",type=str,default="MPI")
    parser.add_argument("-K", "--check_result", help="Checks final result on CPU", action="store_true")
    parser.add_argument("-t", "--threads", help="number of threads to use (MPI + Threads)",default=1,type=int)
    args = parser.parse_args()

    
    NROWS=args.rows
    NCOLS=args.cols
    ITERS = args.iterations
    WARMUP = args.warmup
    MODE = args.placement
    CHECK_RESULT = args.check_result
    NTHREADS = args.threads

    #os.environ["OMP_NUM_THREADS"] = str(NTHREADS)
    

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    npes = comm.Get_size()

    if(not rank):
        print('%**********************************************************************************************%\n')
        print('Config: rows=', NROWS, ' cols=', NCOLS, ' iterations=', ITERS, ' warmup=', WARMUP, ' check_result=', CHECK_RESULT,' MODE=', MODE,' THREADS=', NTHREADS, sep='', end='\n\n')

    comm.barrier()
    tsqr_driver(comm)

    

        



