#!/bin/bash
#SBATCH -J parla           # Job name
#SBATCH -o parla.o%j       # Name of stdout output file
#SBATCH -e parla.e%j       # Name of stderr error file
#SBATCH -p rtx-dev          # Queue (partition) name
#SBATCH -N 2               # Total # of nodes (must be 1 for serial)
#SBATCH -n 2               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)
#SBATCH -A PHY21005       # Project/Allocation name (req'd if you have more than 1)

# Any other commands must follow all #SBATCH directives...
module load cuda
module list
pwd
date

# Launch serial code...

COLS=100
PLACEMENT=gpu
THREADS=4
I=1
W=2

export OMP_NUM_THREADS=4

ROWS=1000
ibtun python3 tsqr_parla_mpi.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W

ROWS=10000
ibrun python3 tsqr_parla_mpi.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W


ROWS=100000
ibrun python3 tsqr_parla_mpi.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W 


ROWS=1000000
ibrun python3 tsqr_parla_mpi.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W


#ROWS=10000000
#ibrun python3 tsqr_parla_mpi.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W






#PLACEMENT=SM+CUDA
#ROWS=1000
#python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W

#ROWS=10000
#python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W

#ROWS=100000
#python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W

#ROWS=1000000
#python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W

#PLACEMENT=MPI+CUDA
#THREADS=1
#export OMP_NUM_THREADS=1

#ROWS=1000
#ibrun -np 4 python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W

#ROWS=10000
#ibrun -np 4 python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W

#ROWS=100000
#ibrun -np 4 python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W


#ROWS=1000000
#ibrun -np 4 python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W
