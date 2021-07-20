#!/bin/bash
#SBATCH -J parla           # Job name
#SBATCH -o mpi_ws_1.o       # Name of stdout output file
#SBATCH -e mpi_ws_1.e       # Name of stderr error file
#SBATCH -p rtx          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 4               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)
#SBATCH -A PHY21005       # Project/Allocation name (req'd if you have more than 1)

# Any other commands must follow all #SBATCH directives...
module load cuda
module list
pwd
date

# Launch serial code...

COLS=100
PLACEMENT=MPI+CUDA
THREADS=1
I=1
W=2

export OMP_NUM_THREADS=1

ROWS=1000
ibrun python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W

ROWS=10000
ibrun python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W


ROWS=100000
ibrun python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W 


COLS=1000
PLACEMENT=MPI+CUDA
THREADS=1
I=1
W=2

export OMP_NUM_THREADS=1

ROWS=1000
ibrun python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W

ROWS=10000
ibrun python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W


ROWS=100000
ibrun python3 tsqr.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W 


