#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:26:43 2022

@author: Nick Singstock
"""

import os
import subprocess

cwd = os.getcwd()

for root, folders, files in os.walk('calcs/'):

    print(root)
    #break
    
    if 'tinyout' not in files:
        continue
    #print('Delete wfns:', root)
    os.chdir(root)
    
    for file in ['wfns']:
        if file in files:
            print('Delete wfns:', root)
            subprocess.call('rm '+file, shell=True)
            #assert False
    os.chdir(cwd)
    #break