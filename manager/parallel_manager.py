#!/usr/bin/env python
"""
run parallel Perl jobs
Logic from Shankar
Author: Nick 
"""

import os
import subprocess
import numpy as np
from time import sleep
from sub_JDFTx import write as sub_write

opj = os.path.join
jdftx_python_dir = os.environ['JDFTx_manager_home']

def build_gpu_exeutable(jdftx_executable, gpus_per_job):
    # Don't need to add srun here because run_JDFTx.py and run_JDFTx_native.py will add it
    return f" -N 1 -n {gpus_per_job} {jdftx_executable}"

def sub_parallel(roots, cwd, nodes, cores_per_node, 
                time, procs = 2, testing = 'False',
                recursive = False, big_mem=False, singlepoint=False,
                gpus_per_job=1, native_ionic=False):
    print('Running '+str(len(roots))+ ' jobs in parallel on '+
          str(nodes)+f' nodes ({int(4/gpus_per_job)} jobs per node)')
    
    manager_home = os.environ['JDFTx_manager_home']
    if native_ionic:
        script = opj(manager_home, 'run_JDFTx_native.py')
    elif native_ionic == False:
        script = opj(manager_home, 'run_JDFTx.py')
    parallel_folder = './tmp_parallel'
    if not os.path.exists(parallel_folder):
        print('Making tmp_parallel/')
        os.mkdir(parallel_folder)
        
    # delete existing locks
    lock_folder = opj(parallel_folder, 'locks')
    if not os.path.exists(lock_folder):
        os.mkdir(lock_folder)
    locks = [opj(lock_folder, f) for f in os.listdir(lock_folder) if os.path.isfile(opj(lock_folder, f))]
    for lock_file in locks:
        subprocess.call('rm '+lock_file, shell=True) # TODO: make a better way to remove unused lock files 
    
    out = 'parallel'
    gpu = True # only available for Perl currently 
    
    if gpu:
        script += ' -g True'
        if gpus_per_job > 1:
            # If we are parallelizing states over gpus (ie multiple gpus per job), we need
            # to build a custom executable string to pass to run_JDFTx.py or run_JDFTx_native.py
            jdftx_executable = os.environ['JDFTx_GPU']
            jdftx_executable = build_gpu_exeutable(jdftx_executable, gpus_per_job)
            script += f" --executable \"{jdftx_executable}\""
            print(f"Using custom executable: {jdftx_executable}")
    if singlepoint: # add singlepoint tag for running singlepoints on whole dataset
        script += ' --singlepoint True'
    # tag to include regen attempt 
    #script += ' -r True'
    
    alloc = 'environ'
    # create parallel.sh header
    writelines = sub_write(nodes, cores_per_node, time, out, alloc, 'standard', script, 'False', 
                           procs, 'True', testing, big_mem, get_header = True)
    
    # assign vars
#    writelines += '\nexport SLURM_CPU_BIND="cores" \n'
#    writelines += 'export JDFTX_MEMPOOL_SIZE=32768  \n'   #update as appropriate
#    writelines += 'export MPICH_GPU_SUPPORT_ENABLED=1  \n\n'
    
    # add recursive logic if needed
    if recursive:
        writelines+='timeout 10 python '+os.path.join(jdftx_python_dir,'timer.py')+' > timer'+'\n\n'
    # add logic to launch parallel versions of para_managers (one per node)
    # n_managers is equal to 4 (gpus per node) divided by how many gpus
    # each job is using. So if we request 4 nodes (16 gpus) and want
    # two gpus per job, we need 8 managers.
    writelines += f'n_managers=$(({int(4/gpus_per_job)} '
    writelines += '* ${SLURM_JOB_NUM_NODES})) \n'
    writelines += 'echo "n_managers = $n_managers" \n'
    writelines += 'for i_manager in $(seq 1 ${n_managers}); do \n'
    writelines += '    python ' + opj(manager_home, 'parallel_manager.py') + ' & \n'
    writelines += 'done \n'
    writelines += 'wait \n'
    
    if recursive:
        writelines+='timeout 10 python '+os.path.join(jdftx_python_dir,'timer.py')+' > timer'+'\n\n'
    
    # writelines is now completed to be written as parallel.sh
    with open(opj(parallel_folder, out+'.sh'),'w') as f:
        f.write(writelines)
        
    # make roots_list.txt with roots needing to be run (from gc_manager)
    roottxt = ''
    for root in roots:
        roottxt += root + '\n'
    with open(opj(parallel_folder, 'root_list.txt'),'w') as f:
        f.write(roottxt)
    subprocess.call('chmod 755 '+opj(parallel_folder, 'root_list.txt'), shell=True)
        
    # make single job shell script
#    singlejob = ('#!/bin/bash \ntask_name="$1" \ntask_num="$2" \n\necho "Starting $task_name"'+
#                '\n\nsrun -N 1 -n 4 ./executable.sh ${task_name} ${task_num}'+
#                '\n\necho "Completed $task_name"')
    singlejob = ('#!/bin/bash \ntask_name="$1" \ntask_num="$2" \n\necho "Starting $task_name"'+
                 # UPDATE: changed to run from within dir so removed -d and {task_name} dependence 
                '\n\nexport FI_CXI_DEFAULT_VNI=$2'+
                '\npython ' + script + ' > out_file'#' -d ../${task_name} > ' # run run_JDFTx.py in calc dir
                 #+ '../${task_name}/out_file'
                 + '\n\necho "Completed $task_name"')
    with open(opj(parallel_folder, 'singlejob.sh'),'w') as f:
        f.write(singlejob)
    subprocess.call('chmod 755 '+opj(parallel_folder, 'singlejob.sh'), shell=True)
        
#    executable = ('#!/bin/bash \ntask_name="$1" \ntask_num="$2" \n\n' + 
#                  'echo "  Running ${task_name} on $(hostname) with ${SLURM_CPUS_PER_TASK}'+
#                  ' threads and cuda devs: ${CUDA_VISIBLE_DEVICES}"' + '\n' + 
#                  'python ' + script +' -d ../${task_name} > ' # run run_JDFTx.py in calc dir
#                  + '../${task_name}/out_file \n')
#    with open(opj(parallel_folder, 'executable.sh'),'w') as f:
#        f.write(executable)
#    subprocess.call('chmod 755 '+opj(parallel_folder, 'executable.sh'), shell=True)
    
    cwd = os.getcwd()
    os.chdir(opj(cwd,'tmp_parallel'))
    os.system('sbatch ' + out+'.sh')
    os.chdir(cwd)


def parallel_logic():
    # Load all jobs:
    task_list = [line.rstrip() for line in open("root_list.txt")]
    
    # Loop over available tasks:
    for i_task, task_name in enumerate(task_list):
        
        # add random delay to prevent separate gpus from reading at the same time
        sleep(np.random.rand() * 5)
        
        try:
            fp = open(f"locks/lock.{i_task}", "x")
        except FileExistsError:
            print(f"lock file {i_task} exists")
            continue  # another task got to it already
    
        cwd = os.getcwd() # this is the tmp_parallel dir where parallel.sh is called
        os.chdir(opj('..', task_name)) # need to switch to job folder for I/O functions
        os.system("bash " + opj(cwd, "singlejob.sh") + f" {task_name} {i_task}") # run job
        os.chdir(cwd) # go back to main dir before next job starts
        
        # Mark job as complete (to help identify incomplete jobs)
        fp.write(f"Completed {task_name}\n")
        fp.close()



if __name__ == '__main__':
    parallel_logic()

