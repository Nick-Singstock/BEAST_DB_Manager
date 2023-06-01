#!/bin/bash
#SBATCH -J bader
#SBATCH --time=1:00:00
#SBATCH -o bader-%j.out
#SBATCH -e bader-%j.err
#SBATCH --nodes=1 --ntasks-per-node=36 
#SBATCH --partition=debug
#SBATCH --account=custws

source activate /home/twhittak/.conda-envs/myoctave
jbader_gc.py -f tinyout

