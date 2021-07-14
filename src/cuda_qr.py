"""
@package Performs collection of qr runs to get a sample runtime for a selected GPU. 
"""

import numpy as np
import cupy as cp
from time import perf_counter as time
import os
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rows", help="Number of rows for input matrix; must be >> cols", type=int, default=5000)
    parser.add_argument("-c", "--cols", help="Number of columns for input matrix", type=int, default=100)
    parser.add_argument("-i", "--iterations", help="Number of iterations to run experiment. If > 1, first is ignored as warmup.", type=int, default=1)
    parser.add_argument("-w", "--warmup", help="Number of warmup runs to perform before iterations.", type=int, default=0)
    #parser.add_argument("-b", "--block_size", help="Block size to break up input matrix; must be >= cols", type=int, default=500)
    #parser.add_argument("-t", "--threads", help="Sets OMP_NUM_THREADS", default='16')
    #parser.add_argument("-g", "--ngpus", help="Sets number of GPUs to run on. If set to more than you have, undefined behavior", type=int, default='4')
    #parser.add_argument("-p", "--placement", help="'cpu' or 'gpu' or 'both' or 'puregpu'", default='gpu')
    #parser.add_argument("-K", "--check_result", help="Checks final result on CPU", action="store_true")
    #parser.add_argument("--csv", help="Prints stats in csv format", action="store_true")

    args = parser.parse_args()

    # Set global config variables
    NROWS = args.rows
    NCOLS = args.cols
    ITERS = args.iterations
    WARMUP = args.warmup
    # NTHREADS = args.threads
    # NGPUS = args.ngpus
    # PLACEMENT_STRING = args.placement
    # CHECK_RESULT = args.check_result
    # CSV = args.csv
    
    t_header=False
    t_list=list()
    for iter in range(ITERS + WARMUP):
        np.random.seed(iter)
        A = np.random.rand(NROWS, NCOLS)
        t_overall = profile_t("total")
        t_overall.start()
        [Q,R,_ts] = block_gpu_qr(A,0,loc={'A':'H','Q':'H','R':'H'})
        t_overall.stop()
        
        if ((iter >= WARMUP)):
            if(not t_header):
                header="iter\ttotal"
                for t in _ts:
                    header+="\t"+t.name+""
                print(header)
                t_header=True
                
            t_dur = t_overall.seconds
            t_min = t_dur
            t_max = t_dur
            t_= f"{iter}\t{t_dur:.12f}"
               
            for t in _ts:
                t_min = t.seconds
                t_= t_ + f"\t{t_min:.12f}"

            print(t_)
            t_list_iter =list()
            t_list_iter.append(t_overall)
            for t in _ts:
                t_list_iter.append(t)
            t_list.append(t_list_iter)
    
    H2D_bytes = NROWS*NCOLS * 8
    D2H_bytes = (NROWS*NCOLS + NCOLS**2)* 8
    GPU_FLOP   = (2*NROWS * (NCOLS**2) - (2/3) * (NCOLS**3))  

    H2D_BW    = 0
    D2H_BW    = 0
    QR_FLOPS  = 0 

    for i,t_iter in enumerate(t_list):
        t_total = t_iter[0]
        t_h2d   = t_iter[1]
        t_qr    = t_iter[2]
        t_d2h   = t_iter[3]

        assert(t_h2d.name == "H2D")
        assert(t_qr.name  == "kernel_gpu")
        assert(t_d2h.name == "D2H")

        H2D_BW   += H2D_bytes / (t_h2d.seconds)
        QR_FLOPS += GPU_FLOP  / (t_qr.seconds)
        D2H_BW   += D2H_bytes / (t_d2h.seconds)

    H2D_BW   /= len(t_list)
    QR_FLOPS /= len(t_list)
    D2H_BW   /= len(t_list)

    H2D_BW   /=(1024**3)
    D2H_BW   /=(1024**3)
    QR_FLOPS /=(1000**3)

    print("\nNROWS\tNCOLS\tH2D_BW(GB/sec)\tQR_FLOPS(GFlops/sec)\tD2H_BW(GB/sec)")
    print(f"{NROWS}\t{NCOLS}\t{H2D_BW}\t{QR_FLOPS}\t{D2H_BW}")

    



