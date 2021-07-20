# Parla + X for qr decomposition - taken from Parla code repo. 
import sys
import argparse
import numpy as np
import cupy as cp
from time import perf_counter as time
import os
from mpi4py import MPI

from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.function_decorators import specialized
from parla.tasks import *
from parla.ldevice import LDeviceSequenceBlocked, PartitionedTensor

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


def check_result_mpi(Ar,Qr,Rr,comm):

    rank = comm.Get_rank()
    npes = comm.Get_size()

    # check the A=QR
    is_prod_valid = np.allclose(Ar,np.matmul(Qr,Rr))
    is_prod_valid = comm.reduce(is_prod_valid,root=0,op=MPI.LAND)

    QTQ   = np.matmul(np.transpose(Qr),Qr)
    QTQ_g = np.zeros((Qr.shape[1],Qr.shape[1]))
    comm.Reduce(QTQ,QTQ_g,root=0,op=MPI.SUM)
    
    if(not rank):
        is_Q_valid = np.allclose(QTQ_g,np.eye(Qr.shape[1]))
    
    
    # Check R is upper triangular
    is_upper_R = np.allclose(Rr, np.triu(Rr))
    is_upper_R = comm.reduce(is_upper_R,root=0,op=MPI.LAND)

    is_valid =  (is_prod_valid and is_Q_valid and is_upper_R)
    return is_valid

def check_result(A, Q, R):
    # Check product
    is_correct_prod = np.allclose(np.matmul(Q, R), A)
    
    # Check orthonormal
    Q_check = np.matmul(Q.transpose(), Q)
    is_ortho_Q = np.allclose(Q_check, np.identity(NCOLS))
    
    # Check upper
    is_upper_R = np.allclose(R, np.triu(R))

    return is_correct_prod and is_ortho_Q and is_upper_R

# CPU QR factorization kernel
@specialized
def qr_block(block, taskid):
    t_qr_cpu[taskid].start()
    Q, R = np.linalg.qr(block)
    cp.cuda.get_current_stream().synchronize()
    t_qr_cpu[taskid].stop()
    return Q, R

# GPU QR factorization kernel and device-to-host transfer
@qr_block.variant(gpu)
def qr_block_gpu(block, taskid):
    # Run the kernel
    t_qr_gpu[taskid].start()
    gpu_Q, gpu_R = cp.linalg.qr(block)
    cp.cuda.get_current_stream().synchronize()
    t_qr_gpu[taskid].stop()
    
    # Transfer the data
    t_d2h[taskid].start()
    cpu_Q = cp.asnumpy(gpu_Q)
    cpu_R = cp.asnumpy(gpu_R)
    cp.cuda.get_current_stream().synchronize()
    t_d2h[taskid].stop()

    return cpu_Q, cpu_R

# CPU matmul kernel
@specialized
def matmul_block(block_1, block_2, taskid):
    t_mm_cpu[taskid].start()
    Q = block_1 @ block_2
    t_mm_cpu[taskid].stop()
    return Q

# GPU matmul kernel and device-to-host transfer
@matmul_block.variant(gpu)
def matmul_block_gpu(block_1, block_2, taskid):
    # Run the kernel
    t_mm_gpu[taskid].start()
    gpu_Q = cp.matmul(block_1, block_2)
    cp.cuda.get_current_stream().synchronize()
    t_mm_gpu[taskid].stop()

    
    # Transfer the data
    t_d2h[taskid].start()
    cpu_Q = cp.asnumpy(gpu_Q)
    cp.cuda.get_current_stream().synchronize()
    t_d2h[taskid].stop()

    return cpu_Q

async def tsqr_blocked(A, block_size):
    [nrows, ncols] = A.shape
    # Check for block_size > ncols
    assert ncols <= block_size, "Block size must be greater than or equal to the number of columns in the input matrix"

    # Calculate the number of blocks
    nblocks = (nrows + block_size - 1) // block_size # ceiling division
    mapper = LDeviceSequenceBlocked(nblocks, placement=A)
    A_blocked = mapper.partition_tensor(A) # Partition A into blocks
    
    # Initialize and partition empty array to store blocks (same partitioning scheme, share the mapper)
    Q1_blocked = mapper.partition_tensor(np.empty_like(A))
    R1 = np.empty([nblocks * ncols, ncols]) # Concatenated view
    # Q2 is allocated in t2
    Q = np.empty([nrows, ncols]) # Concatenated view

    # Create tasks to perform qr factorization on each block and store them in lists
    #t1_tot_start = time()
    t1_total.start()
    T1 = TaskSpace()
    for i in range(nblocks):
        # Block view to store Q1 not needed since it's not contiguous

        # Get block view to store R1
        R1_lower = i * ncols
        R1_upper = (i + 1) * ncols

        T1_MEMORY = None
        if PLACEMENT_STRING == 'gpu' or PLACEMENT_STRING == 'both':
            T1_MEMORY = int(4.2*A_blocked[i:i+1].nbytes) # Estimate based on empirical evidence

        @spawn(taskid=T1[i], placement=PLACEMENT, memory=T1_MEMORY, vcus=ACUS)
        def t1():
            #print("t1[", i, "] start on ", get_current_devices(), sep='', flush=True)

            # Copy the data to the processor
            t_h2d[i].start()
            A_block_local = A_blocked[i:i+1]
            cp.cuda.get_current_stream().synchronize()
            t_h2d[i].stop()
            
            # Run the kernel. (Data is copied back within this call; timing annotations are added there)
            Q1_blocked[i], R1[R1_lower:R1_upper] = qr_block(A_block_local, i)
            #print("t1[", i, "] end on ", get_current_devices(),  sep='', flush=True)

    await t1
    t1_total.stop()
    #t1_tot_end = time()
    #perf_stats.t1_tot = t1_tot_end - t1_tot_start

    # Perform intermediate qr factorization on R1 to get Q2 and final R
    t2_total.start()
    @spawn(dependencies=T1, placement=cpu)
    def t2():
        #print("\nt2 start", flush=True)

        # R here is the final R result
        # This step could be done recursively, but for small column counts that's not necessary
        Q2, R = np.linalg.qr(R1)

        # Q1 and Q2 must have an equal number of blocks, where Q1 blocks' ncols = Q2 blocks' nrows
        # Q2 is currently an (ncols * nblocks) x ncols matrix. Need nblocks of ncols rows each
        return Q2, R

    Q2, R = await t2
    t2_total.stop()
    
    # Partition Q2 (same partitioning scheme, share the mapper)
    Q2_blocked = mapper.partition_tensor(Q2)
    t3_total.start()
    # Create tasks to perform Q1 @ Q2 matrix multiplication by block
    T3 = TaskSpace()
    for i in range(nblocks):
        # Q1 is already in blocks

        # Get block view to store Q
        Q_lower = i * block_size # first row in block, inclusive
        Q_upper = (i + 1) * block_size # last row in block, exclusive

        T3_MEMORY = None
        if PLACEMENT_STRING == 'gpu' or PLACEMENT_STRING == 'both':
            T3_MEMORY = 4*Q1_blocked[i].nbytes # # This is a guess

        @spawn(taskid=T3[i], dependencies=[T1[i], t2], placement=PLACEMENT, memory=T3_MEMORY, vcus=ACUS)
        def t3():
            #print("t3[", i, "] start on ", get_current_devices(), sep='', flush=True)

            # Copy the data to the processor
            t_h2d[i].start()
            Q1_block_local = Q1_blocked[i]
            Q2_block_local = Q2_blocked[i:i+1]
            cp.cuda.get_current_stream().synchronize()
            t_h2d[i].stop()
            
            # Run the kernel. (Data is copied back within this call; timing annotations are added there)
            Q[Q_lower:Q_upper] = matmul_block(Q1_block_local, Q2_block_local, i)
            #print("t3[", i, "] end on ", get_current_devices(), sep='', flush=True)

    await T3
    t3_total.stop()
    return Q, R

async def tsqr_blocked_mpi(Ar,comm,block_size):

    rank = comm.Get_rank()
    npes = comm.Get_size()

    [nrows, ncols] = Ar.shape
    # Check for block_size > ncols
    assert ncols <= block_size, "Block size must be greater than or equal to the number of columns in the input matrix"

    # Calculate the number of blocks
    nblocks = (nrows + block_size - 1) // block_size # ceiling division
    mapper = LDeviceSequenceBlocked(nblocks, placement=Ar)
    A_blocked = mapper.partition_tensor(Ar) # Partition A into blocks
    
    # Initialize and partition empty array to store blocks (same partitioning scheme, share the mapper)
    Q1_blocked = mapper.partition_tensor(np.empty_like(Ar))
    R1 = np.empty([nblocks * ncols, ncols]) # Concatenated view
    # Q2 is allocated in t2
    Q = np.empty([nrows, ncols]) # Concatenated view

    # Create tasks to perform qr factorization on each block and store them in lists
    #t1_tot_start = time()
    t1_total.start()
    T1 = TaskSpace()
    for i in range(nblocks):
        # Block view to store Q1 not needed since it's not contiguous

        # Get block view to store R1
        R1_lower = i * ncols
        R1_upper = (i + 1) * ncols

        T1_MEMORY = None
        if PLACEMENT_STRING == 'gpu' or PLACEMENT_STRING == 'both':
            T1_MEMORY = int(4.2*A_blocked[i:i+1].nbytes) # Estimate based on empirical evidence

        @spawn(taskid=T1[i], placement=PLACEMENT, memory=T1_MEMORY, vcus=ACUS)
        def t1():
            #print("t1[", i, "] start on ", get_current_devices(), sep='', flush=True)

            # Copy the data to the processor
            t_h2d[i].start()
            A_block_local = A_blocked[i:i+1]
            cp.cuda.get_current_stream().synchronize()
            t_h2d[i].stop()
            
            # Run the kernel. (Data is copied back within this call; timing annotations are added there)
            Q1_blocked[i], R1[R1_lower:R1_upper] = qr_block(A_block_local, i)
            #print("t1[", i, "] end on ", get_current_devices(),  sep='', flush=True)

    await t1
    t1_total.stop()
    #t1_tot_end = time()
    #perf_stats.t1_tot = t1_tot_end - t1_tot_start
    t_comm_mpi.start()
    R1=gather_mat(R1,comm)
    t_comm_mpi.stop()

    # Perform intermediate qr factorization on R1 to get Q2 and final R
    t2_total.start()
    @spawn(dependencies=T1, placement=cpu)
    def t2():
        #print("\nt2 start", flush=True)
        Q2 = None
        R  = None
        # R here is the final R result
        # This step could be done recursively, but for small column counts that's not necessary
        if(not rank):
            Q2, R = np.linalg.qr(R1)

        # Q1 and Q2 must have an equal number of blocks, where Q1 blocks' ncols = Q2 blocks' nrows
        # Q2 is currently an (ncols * nblocks) x ncols matrix. Need nblocks of ncols rows each
        return Q2, R

    Q2, R = await t2
    t2_total.stop()
    comm.barrier()
    
    t_comm_mpi.start()
    Q2 = scatter_mat(Q2,comm)
    R   = comm.bcast(R)
    t_comm_mpi.stop()
    
    # Partition Q2 (same partitioning scheme, share the mapper)
    Q2_blocked = mapper.partition_tensor(Q2)
    t3_total.start()
    # Create tasks to perform Q1 @ Q2 matrix multiplication by block
    T3 = TaskSpace()
    for i in range(nblocks):
        # Q1 is already in blocks

        # Get block view to store Q
        Q_lower = i * block_size # first row in block, inclusive
        Q_upper = (i + 1) * block_size # last row in block, exclusive

        T3_MEMORY = None
        if PLACEMENT_STRING == 'gpu' or PLACEMENT_STRING == 'both':
            T3_MEMORY = 4*Q1_blocked[i].nbytes # # This is a guess

        @spawn(taskid=T3[i], dependencies=[T1[i], t2], placement=PLACEMENT, memory=T3_MEMORY, vcus=ACUS)
        def t3():
            #print("t3[", i, "] start on ", get_current_devices(), sep='', flush=True)

            # Copy the data to the processor
            t_h2d[i].start()
            Q1_block_local = Q1_blocked[i]
            Q2_block_local = Q2_blocked[i:i+1]
            cp.cuda.get_current_stream().synchronize()
            t_h2d[i].stop()
            
            # Run the kernel. (Data is copied back within this call; timing annotations are added there)
            Q[Q_lower:Q_upper] = matmul_block(Q1_block_local, Q2_block_local, i)
            #print("t3[", i, "] end on ", get_current_devices(), sep='', flush=True)

    await T3
    t3_total.stop()
    return Q, R



# !!! Don't use this. Force all the tasks to GPU. (Not Used)
async def tsqr_blocked_puregpu(A, block_size):
    Q1 = [None] * NGPUS
    R1 = PartitionedTensor([None] * NGPUS) # CAVEAT: PartitionedTensor with None holes can be fragile! Be cautious!

    # Create tasks to perform qr factorization on each block and store them in lists
    t1_total.start()
    T1 = TaskSpace()
    for i in range(NGPUS):
        @spawn(taskid=T1[i], placement=A.base[i]) # NB: A[i] dumbly moves the block here!
        def t1():
            #print("t1[", i, "] start on ", get_current_devices(), sep='', flush=True)
            t_qr_gpu[i].start()
            Q1[i], R1[i] = cp.linalg.qr(A[i])
            cp.cuda.get_current_stream().synchronize()
            t_qr_gpu[i].stop()
            R1[i] = R1[i].flatten()
            A[i] = None # Free up memory
            #print("t1[", i, "] end on ", get_current_devices(),  sep='', flush=True)

    await t1
    t1_total.stop()

    # Perform intermediate qr factorization on R1 to get Q2 and final R
    t2_total.start()
    @spawn(dependencies=T1, placement=gpu)
    def t2():
        #print("\nt2 start", flush=True)
        # Gather to this device
        R1_reduced = np.empty(shape=(0, NCOLS))
        for dev in range(NGPUS):
            next = R1[dev]
            next = next.reshape(NCOLS, NCOLS)
            R1_reduced = cp.vstack((R1_reduced, next))
            R1[dev] = None # Free up memory

        cp.cuda.get_current_stream().synchronize()
        # R here is the final R result
        Q2, R = cp.linalg.qr(R1_reduced)
        cp.cuda.get_current_stream().synchronize()
        Q2 = Q2.flatten()
        return Q2, R

    Q2, R = await t2
    t2_total.stop()
    #print("t2 end\n", flush=True)

    mapper = LDeviceSequenceBlocked(NGPUS, placement=Q2)
    Q2p = mapper.partition_tensor(Q2)
    Q = [None] * NGPUS
    
    t3_total.start()
    # Create tasks to perform Q1 @ Q2 matrix multiplication by block
    T3 = TaskSpace()
    for i in range(NGPUS):
        @spawn(taskid=T3[i], dependencies=[T1[i], t2], placement=Q1[i])
        def t3():
            #print("t3[", i, "] start on ", get_current_devices(), sep='', flush=True)
            # Copy the data to the processor
            # Q1 and Q2 must have an equal number of blocks, where Q1 blocks' ncols = Q2 blocks' nrows
            # Q2 is currently an (ncols * nblocks) x ncols matrix. Need nblocks of ncols rows each
            t_h2d[i].start()
            Q2_local = Q2p[i]
            cp.cuda.get_current_stream().synchronize()
            Q2_local = Q2_local.reshape(NCOLS, NCOLS)
            t_h2d[i].stop()
            
            # Run the kernel. (Data is copied back within this call; timing annotations are added there)
            t_mm_gpu[i].start()
            Q[i] = cp.matmul(Q1[i], Q2_local)
            cp.cuda.get_current_stream().synchronize()
            t_mm_gpu[i].stop()

    await T3
    t3_total.stop()
    return Q, R

def main():
    @spawn()
    async def launch_tsqr(placement=cpu):
        t_header=False
        ts=[t_h2d,t_d2h,t_qr_cpu,t_qr_gpu,t_mm_cpu,t_mm_gpu]
        for iter in range(WARMUP + ITERS):
            np.random.seed(iter)
            Ar = partitioned_rand_mat(NROWS,NCOLS,comm)
            
            t1_total.reset()
            t2_total.reset()
            t3_total.reset()
            t_comm_mpi.reset()

            for tt in ts:
                for t in tt:
                    t.reset()
            
            t_overall = profile_t("total")
            t_overall.start()
            Qr, Rr = await tsqr_blocked_mpi(Ar,comm, BLOCK_SIZE)
            t_overall.stop()
            if ((iter >= WARMUP)):
                    if(not t_header):
                        header="iter\ttotal_min\ttotal_max\tmpi_comm_min\tmpi_comm_max\tqr1_min\tqr1_max\tqr2_min\tqr2_max\tmm_min\tmm_max"
                        for t in ts:
                            header+="\t"+t[0].name+"_min"
                            header+="\t"+t[0].name+"_max"

                        if(not rank):
                            print(header)

                        t_header=True
                    
                    t_    = f"{iter}"
                    t_dur = t_overall.seconds
                    t_min = comm.reduce(t_dur,root=0,op=MPI.MIN)
                    t_max = comm.reduce(t_dur,root=0,op=MPI.MAX)
                    if(not rank):
                        t_+=f"\t{t_min:.12f}\t{t_max:.12f}"
                    
                    t_dur = t_comm_mpi.seconds
                    t_min = comm.reduce(t_dur,root=0,op=MPI.MIN)
                    t_max = comm.reduce(t_dur,root=0,op=MPI.MAX)
                    if(not rank):
                        t_+=f"\t{t_min:.12f}\t{t_max:.12f}"
                    
                    t_dur = t1_total.seconds
                    t_min = comm.reduce(t_dur,root=0,op=MPI.MIN)
                    t_max = comm.reduce(t_dur,root=0,op=MPI.MAX)
                    if(not rank):
                        t_ += f"\t{t_min:.12f}\t{t_max:.12f}"
                    
                    t_dur = t2_total.seconds
                    t_min = comm.reduce(t_dur,root=0,op=MPI.MIN)
                    t_max = comm.reduce(t_dur,root=0,op=MPI.MAX)
                    if(not rank):
                        t_ += f"\t{t_min:.12f}\t{t_max:.12f}"
                    
                    t_dur = t3_total.seconds
                    t_min = comm.reduce(t_dur,root=0,op=MPI.MIN)
                    t_max = comm.reduce(t_dur,root=0,op=MPI.MAX)
                    if(not rank):
                        t_ += f"\t{t_min:.12f}\t{t_max:.12f}"

                    for tt in ts:
                        tt_min = min(tt,key=lambda a: a.seconds).seconds
                        tt_max = max(tt,key=lambda a: a.seconds).seconds

                        t_dur = tt_min
                        t_min = comm.reduce(t_dur,root=0,op=MPI.MIN)

                        t_dur = tt_max
                        t_max = comm.reduce(t_dur,root=0,op=MPI.MAX)

                        if(not rank):
                            t_+= f"\t{t_min:.12f}\t{t_max:.12f}"

                    if(not rank):
                        print(t_)
            
            # barrier for qr2
            comm.barrier()
            # Check the results
            if CHECK_RESULT:
                is_valid = check_result_mpi(Ar,Qr,Rr,comm)
                if(not rank):
                    if is_valid:
                        print("\nCorrect result!\n")
                    else:
                        print("%***** ERROR: Incorrect final result!!! *****%")
                



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rows", help="Number of rows for input matrix; must be >> cols", type=int, default=5000)
    parser.add_argument("-c", "--cols", help="Number of columns for input matrix", type=int, default=100)
    parser.add_argument("-b", "--block_size", help="Block size to break up input matrix; must be >= cols", type=int, default=500)
    parser.add_argument("-i", "--iterations", help="Number of iterations to run experiment. If > 1, first is ignored as warmup.", type=int, default=1)
    parser.add_argument("-w", "--warmup", help="Number of warmup runs to perform before iterations.", type=int, default=0)
    parser.add_argument("-t", "--threads", help="Sets OMP_NUM_THREADS", default='16')
    parser.add_argument("-g", "--ngpus", help="Sets number of GPUs to run on. If set to more than you have, undefined behavior", type=int, default='4')
    parser.add_argument("-p", "--placement", help="'cpu' or 'gpu' or 'both' or 'puregpu'", default='gpu')
    parser.add_argument("-K", "--check_result", help="Checks final result on CPU", action="store_true")
    parser.add_argument("--csv", help="Prints stats in csv format", action="store_true")

    args = parser.parse_args()

    # Set global config variables
    NROWS = args.rows
    NCOLS = args.cols
    BLOCK_SIZE = args.block_size
    ITERS = args.iterations
    WARMUP = args.warmup
    NTHREADS = args.threads
    NGPUS = args.ngpus
    PLACEMENT_STRING = args.placement
    CHECK_RESULT = args.check_result
    CSV = args.csv
    
    BLOCK_SIZE = NROWS // (1*NGPUS)
    NTASKS=(NROWS + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Set up PLACEMENT variable
    if PLACEMENT_STRING == 'cpu':
        PLACEMENT = cpu
        ACUS = None
    elif PLACEMENT_STRING == 'gpu':
        PLACEMENT = [gpu(i) for i in range(NGPUS)]
        ACUS = None
    elif PLACEMENT_STRING == 'both':
        PLACEMENT = [cpu(0)] + [gpu(i) for i in range(NGPUS)]
        ACUS = 1
    elif PLACEMENT_STRING == 'puregpu':
        PLACEMENT = [gpu(i) for i in range(NGPUS)]
        ACUS = None
        BLOCK_SIZE = int(NROWS / NGPUS)
    else:
        print("Invalid value for placement. Must be 'cpu' or 'gpu' or 'both' or 'puregpu'")
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    npes = comm.Get_size()

    t1_total   = profile_t("t1_total") 
    t2_total   = profile_t("t2_total")
    t3_total   = profile_t("t3_total")
    t_comm_mpi = profile_t("mpi_comm")

    t_h2d    = [profile_t("h2d")]    * NTASKS
    t_d2h    = [profile_t("d2h")]    * NTASKS
    t_qr_cpu = [profile_t("qr_cpu")] * NTASKS
    t_qr_gpu = [profile_t("qr_gpu")] * NTASKS
    t_mm_cpu = [profile_t("mm_cpu")] * NTASKS
    t_mm_gpu = [profile_t("mm_gpu")] * NTASKS
    
    if(not rank):
        print('%**********************************************************************************************%\n')
        print("number of mpi tasks: ",npes)
        print('Config: rows=', NROWS, ' cols=', NCOLS, ' block_size=', BLOCK_SIZE, ' iterations=', ITERS, ' warmup=', WARMUP, \
            ' threads=', NTHREADS, ' ngpus=', NGPUS, ' placement=', PLACEMENT_STRING, ' check_result=', CHECK_RESULT, ' csv=', CSV, sep='')

    if PLACEMENT_STRING == 'puregpu':
        if(not rank):
            print('puregpu version chosen: block size automatically set to NROWS / NGPUS\n')

    with Parla():
        os.environ['OMP_NUM_THREADS'] = NTHREADS
        main()

    if(not rank):
        print('%**********************************************************************************************%\n')
