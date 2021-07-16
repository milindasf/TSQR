#!/bin/bash
#SBATCH -J gpuQR           # Job name
#SBATCH -o gpu_QR.o%j       # Name of stdout output file
#SBATCH -e gpu_QR.e%j       # Name of stderr error file
#SBATCH -p rtx-dev          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)
#SBATCH -A PHY21005       # Project/Allocation name (req'd if you have more than 1)

# Any other commands must follow all #SBATCH directives...
module load cuda
module list
pwd
date

# Launch serial code...

PLACEMENT=gpu
THREADS=4
I=2
W=2



COLS=10
ROWS=1000
python3 cuda_qr.py -r $ROWS -c $COLS -i $I -w $W 

ROWS=10000
python3 cuda_qr.py -r $ROWS -c $COLS -i $I -w $W 

ROWS=100000
python3 cuda_qr.py -r $ROWS -c $COLS -i $I -w $W 

ROWS=1000000
python3 cuda_qr.py -r $ROWS -c $COLS -i $I -w $W 





COLS=100
ROWS=1000
python3 cuda_qr.py -r $ROWS -c $COLS -i $I -w $W 

ROWS=10000
python3 cuda_qr.py -r $ROWS -c $COLS -i $I -w $W 

ROWS=100000
python3 cuda_qr.py -r $ROWS -c $COLS -i $I -w $W 

ROWS=1000000
python3 cuda_qr.py -r $ROWS -c $COLS -i $I -w $W 



COLS=1000
ROWS=1000
python3 cuda_qr.py -r $ROWS -c $COLS -i $I -w $W 

ROWS=10000
python3 cuda_qr.py -r $ROWS -c $COLS -i $I -w $W 

ROWS=100000
python3 cuda_qr.py -r $ROWS -c $COLS -i $I -w $W 

ROWS=1000000
python3 cuda_qr.py -r $ROWS -c $COLS -i $I -w $W 



