# Parla + X for qr decomposition - taken from Parla code repo. 
import sys
import argparse
import numpy as np
import cupy as cp
from time import perf_counter as time
import os
import logging

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
    #cpu_Q = cp.asnumpy(gpu_Q)
    cpu_Q = gpu_Q
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
    # milinda : no need the partition buffer we can keep the Q1 in the GPU
    Q1_blocked = [None]*nblocks #mapper.partition_tensor(np.empty_like(A))
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
        # if PLACEMENT_STRING == 'gpu' or PLACEMENT_STRING == 'both':
        #     T1_MEMORY = int(4.2*A_blocked[i:i+1].nbytes) # Estimate based on empirical evidence

        dev_id = i % NGPUS
        @spawn(taskid=T1[i], placement=PLACEMENT[dev_id], memory=T1_MEMORY, vcus=ACUS)
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
        ###@todo: Parla bug: refering this outside makes the array copy to the device. 
        # if PLACEMENT_STRING == 'gpu' or PLACEMENT_STRING == 'both':
        #     T3_MEMORY = 4*Q1_blocked[i].nbytes # # This is a guess

        dev_id = i % NGPUS
        @spawn(taskid=T3[i], dependencies=[T1[i], t2], placement=PLACEMENT[dev_id], memory=T3_MEMORY, vcus=ACUS)
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

def check_result(A, Q, R):
    # Check product
    is_correct_prod = np.allclose(np.matmul(Q, R), A)
    
    # Check orthonormal
    Q_check = np.matmul(Q.transpose(), Q)
    is_ortho_Q = np.allclose(Q_check, np.identity(NCOLS))
    
    # Check upper
    is_upper_R = np.allclose(R, np.triu(R))

    return is_correct_prod and is_ortho_Q and is_upper_R

def main():
    @spawn()
    async def test_tsqr_blocked(placement=cpu):
        t_header=False
        ts=[t_h2d,t_d2h,t_qr_cpu,t_qr_gpu,t_mm_cpu,t_mm_gpu]
        for iter in range(WARMUP + ITERS):
            np.random.seed(iter)
            A = np.random.rand(NROWS, NCOLS)
            
            t1_total.reset()
            t2_total.reset()
            t3_total.reset()

            for tt in ts:
                for t in tt:
                    t.reset()
            
            t_overall = profile_t("total")
            t_overall.start()
            Q, R = await tsqr_blocked(A, BLOCK_SIZE)
            t_overall.stop()
            if ((iter >= WARMUP)):
                    if(not t_header):
                        header="iter\ttotal\tqr1\tqr2\tmm"
                        for t in ts:
                            header+="\t"+t[0].name+"_min"
                            header+="\t"+t[0].name+"_max"
                        print(header)
                        t_header=True
                    
                    t_dur = t_overall.seconds
                    t_min = t_dur
                    t_max = t_dur
                    t_= f"{iter}\t{t_dur:.12f}\t{t1_total.seconds:.12f}\t{t2_total.seconds:.12f}\t{t3_total.seconds:.12f}"

                    for tt in ts:
                        t_min = tt[0].seconds
                        t_max = tt[0].seconds
                        for t in tt:
                            if (t_min > t.seconds):
                                t_min = t.seconds
                            
                            if (t_max < t.seconds):
                                t_max = t.seconds
                            # t_min = min(t,key=lambda a: a.seconds).seconds
                            # t_max = max(t,key=lambda a: a.seconds).seconds
                        t_= t_ + f"\t{t_min:.12f}\t{t_max:.12f}"
                    print(t_)
            # Check the results
            if CHECK_RESULT:
                if check_result(A, Q, R):
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
    # uncomment to enable logging. 
    #logging.basicConfig(filename='tsqr_parla_sm.log', level=logging.DEBUG)
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
    
    t1_total = profile_t("t1_total") 
    t2_total = profile_t("t2_total")
    t3_total = profile_t("t3_total")
    t_h2d    = [profile_t("h2d")]    * NTASKS
    t_d2h    = [profile_t("d2h")]    * NTASKS
    t_qr_cpu = [profile_t("qr_cpu")] * NTASKS
    t_qr_gpu = [profile_t("qr_gpu")] * NTASKS
    t_mm_cpu = [profile_t("mm_cpu")] * NTASKS
    t_mm_gpu = [profile_t("mm_gpu")] * NTASKS
    
    print('%**********************************************************************************************%\n')
    print('Config: rows=', NROWS, ' cols=', NCOLS, ' block_size=', BLOCK_SIZE, ' iterations=', ITERS, ' warmup=', WARMUP, \
        ' threads=', NTHREADS, ' ngpus=', NGPUS, ' placement=', PLACEMENT_STRING, ' check_result=', CHECK_RESULT, ' csv=', CSV, sep='')

    if PLACEMENT_STRING == 'puregpu':
        print('puregpu version chosen: block size automatically set to NROWS / NGPUS\n')

    with Parla():
        #os.environ['OMP_NUM_THREADS'] = NTHREADS
        main()

    print('%**********************************************************************************************%\n')
