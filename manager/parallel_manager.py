#!/usr/bin/env python
"""
run parallel Perl jobs
Logic from Shankar
Author: Nick 
"""

import os
from sub_JDFTx import write as sub_write


def sub_parallel(roots, cwd, nodes, cores_per_node, 
                time, procs = 2, testing = 'False'):

    script = os.path.join(os.environ['JDFTx_manager_home'], 'run_JDFTx.py')
    parallel_folder = './tmp_parallel'
    if not os.path.exists(parallel_folder):
        os.mkdir(parallel_folder)
    if not os.path.exists(os.path.join(parallel_folder, 'locks')):
        os.mkdir(os.path.join(parallel_folder, 'locks'))
    out = 'parallel'
    gpu = True # only available for Perl currently 
    
    if gpu:
        script += ' -g True'
    alloc = 'environ'
    # create parallel.sh header
    writelines = sub_write(nodes, cores_per_node, time, out, alloc, 'standard', script, 'False', 
                           procs, 'True', testing, get_header = True)
    
    # assign vars
    writelines += '\nexport SLURM_CPU_BIND="cores" \n'
    writelines += 'export JDFTX_MEMPOOL_SIZE=32768  \n'   #update as appropriate
    writelines += 'export MPICH_GPU_SUPPORT_ENABLED=1  \n\n'
    
    # add logic to launch parallel versions of para_managers (one per node)
    writelines += 'n_managers=${SLURM_JOB_NUM_NODES} \n'  # to run one task/node
    writelines += 'for i_manager in $(seq 1 ${n_managers}); do \n'
    writelines += '    python parallel_manager.py & \n'
    writelines += 'done \n'
    writelines += 'wait \n'
    
    # writelines is now completed to be written as parallel.sh
    with open(os.path.join(parallel_folder, out+'.sh'),'w') as f:
        f.write(writelines)
        
    # make roots_list.txt with roots needing to be run (from gc_manager)
    roottxt = ''
    for root in roots:
        roottxt += root + '\n'
    with open(os.path.join(parallel_folder, 'root_list.txt'),'w') as f:
        f.write(roottxt)
        
    # make single job shell script
    singlejob = ('#!/bin/bash \ntask_name="$1" \ntask_num="$2" \n\necho "Starting $task_name"'+
                '\n\nsrun -N 1 -n 4 ./executable.sh ${task_name} ${task_num}'+
                '\n\necho "Completed $task_name"')
    with open(os.path.join(parallel_folder, 'singlejob.sh'),'w') as f:
        f.write(singlejob)
        
    executable = ('#!/bin/bash \ntask_name="$1" \ntask_num="$2" \n\n' + 
                  
                  'python ' + script +' -d ../${task_name} > ' # TODO: this needs to be hardcode
                  + '../${task_name}/out_file \n')
    with open(os.path.join(parallel_folder, 'executable.sh'),'w') as f:
        f.write(executable)
    
    
    os.system('sbatch ' + os.path.join(parallel_folder, out+'.sh'))


def parallel_logic():
    # Load all jobs:
    task_list = [line.rstrip() for line in open("root_list.txt")]
    
    # Loop over available tasks:
    for i_task, task_name in enumerate(task_list):
        try:
            fp = open(f"locks/lock.{i_task}", "x")
        except FileExistsError:
            continue  # another task got to it already
    
        # Run job:
        os.system(f"bash ./singlejob.sh {task_name} {i_task}")
        
        # Mark job as complete (to help identify incomplete jobs)
        fp.write(f"Completed {task_name}\n")
        fp.close()



if __name__ == '__main__':
    parallel_logic()

