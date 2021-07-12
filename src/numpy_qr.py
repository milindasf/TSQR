import numpy as np
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
        [Q,R] = np.linalg.qr(A,mode='reduced')
        t_overall.stop()
        
        if ((iter >= WARMUP)):
            if(not t_header):
                header="iter\ttotal"
                print(header)
                t_header=True
                
            t_dur = t_overall.seconds
            t_= f"{iter}\t{t_dur:.12f}"
            print(t_)

            t_list.append(t_overall)
               
    CPU_FLOP   = (NROWS * (NCOLS**2) - (2/3) * (NCOLS**3))  

    QR_FLOPS  = 0 

    for i,t_iter in enumerate(t_list):
        t_qr = t_iter
        QR_FLOPS += CPU_FLOP  / (t_qr.seconds)
        
    QR_FLOPS /= len(t_list)
    
    QR_FLOPS /=(1000**3)
    print("\nNROWS\tNCOLS\tQR_FLOPS(GFlops/sec)")
    print(f"{NROWS}\t{NCOLS}\t{QR_FLOPS}")