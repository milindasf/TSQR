#!/bin/bash
#SBATCH -J parla           # Job name
#SBATCH -o parla_ws_8.o       # Name of stdout output file
#SBATCH -e parla_ws_8.e       # Name of stderr error file
#SBATCH -p rtx          # Queue (partition) name
#SBATCH -N 8               # Total # of nodes (must be 1 for serial)
#SBATCH -n 8               # Total # of mpi tasks (should be 1 for serial)
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

ROWS=8000
ibrun python3 tsqr_parla_mpi.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W

ROWS=80000
ibrun python3 tsqr_parla_mpi.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W


ROWS=800000
ibrun python3 tsqr_parla_mpi.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W 



COLS=1000
PLACEMENT=gpu
THREADS=4
I=1
W=2

export OMP_NUM_THREADS=4

ROWS=8000
ibrun python3 tsqr_parla_mpi.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W

ROWS=80000
ibrun python3 tsqr_parla_mpi.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W


ROWS=800000
ibrun python3 tsqr_parla_mpi.py -r $ROWS -c $COLS -p $PLACEMENT -t $THREADS -i $I -w $W 


