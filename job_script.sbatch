#!/bin/bash
#SBATCH -n 1           
#SBATCH --time 24:00:00      
#SBATCH --mem-per-cpu=8000 
#SBATCH -J analysis1
#SBATCH -o analysis1.out
#SBATCH -e analysis1.err

module load stack/.2024-05-silent  gcc/13.2.0
module load python/3.11.6

python RNN_batch.py
