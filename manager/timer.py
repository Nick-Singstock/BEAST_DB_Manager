#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3/23/2021

@author: nisi6161
"""

import time
from datetime import datetime
import os
import subprocess

def write_time(append = True, filename = 'timer.txt'):
    dt = datetime.now()
    dt_string = dt.strftime("%d/%m/%Y %H:%M:%S")
    timer = str(time.time())
    time_text = dt_string + '\nTime: ' + timer + '\n'
    write(time_text, filename = filename)

def read(filename = 'timer.txt'):
    with open(filename,'r') as f:
        timer_text = f.read()
    return timer_text

def write(text, append = True, filename = 'timer.txt'):
    current = ''
    if append:
        current = read()
    with open(filename,'w') as f:
        f.write(current + '\n' + text)

def run(cmd):
    subprocess.call(cmd, shell=True)

def get_lines(text):
    return [x for x in text.split('\n') if x != '']

def main(codename = '', run_cmd = ''):
    cwd = os.getcwd()
    if run_cmd == '':
        run_cmd = 'sbatch submit.sh'
    short_queue_time = 4 * 60 * 60
    still_running_window = 2 * 60 # rerun if job is within 2 minutes of finishing
    
    if codename != '':
        timer_name = 'timer-'+codename+'.txt'
    else:
        timer_name = 'timer.txt'
    timer_exists = os.path.exists(os.path.join(cwd, timer_name))

    # scenario 1: first run, timer.txt not yet created
    if not timer_exists:
        write('Start \n', False, filename = timer_name)
        write_time(filename = timer_name)
        return

    else:
        time_txt = read()
        lines = get_lines(time_txt)
        if 'Time: ' in lines[-1]:
            # scenario 2/3: timer.txt created: job finishing, need to decide whether to rerun
            old_time = float(lines[-1].split()[1])
            current_time = time.time()
            delta = current_time - old_time
            if delta > short_queue_time - still_running_window:
                # rerun job
                write('Rerun\n', filename = timer_name)
                run(run_cmd)
                return
            else:
                # job complete
                write('Complete\n', filename = timer_name)
                return

        else:
            # scenario 4: timer.txt created, new job starting, need to print new start time
            write('Restart \n', filename = timer_name)
            write_time(filename = timer_name)
            return

# run timer manager function
main()
