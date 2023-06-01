#!/usr/bin/env python

#built with python 3

# Jacob Clary 6/4/2019
# to add:

#script to do bader analysis for JDFTx calculation
#assumes JDFTx createVASP script works and bader is available 

import os, sys
import argparse 
import shutil
from shutil import copy2
import subprocess

def find_key(key_input, tempfile):
    #finds line where key occurs in stored input, last instance
    key_input = str(key_input)
    line = len(tempfile)                  #default to end
    for i in range(0,len(tempfile)):
        if key_input in tempfile[i]:
            line = i
    return line  #return int

def find_all_key(key_input, tempfile):
    #finds all lines where key occurs in in lines
    #should be combined with above function
    L = []     #default
    key_input = str(key_input)
    for i in range(0,len(tempfile)):
        if key_input in tempfile[i]:
            L.append(i)
    if not L:
        L = [len(tempfile)]
    return L  #returns list

def find_first_range_key(key_input, tempfile, startline=0, endline=-1, skip_pound = False):
    #finds all lines that exactly begin with key
    #should also be combined with above functions
    #good if parsing file where a word appears many times and just need extract specific
    #    lines that all start with the same string
    key_input = str(key_input)
    startlen = len(key_input)
    L = []

    if endline == -1:
        endline = len(tempfile)
    for i in range(startline,endline):
        line = tempfile[i]
        if skip_pound == True:
            for j in range(10):  #repeat to make sure no really weird formatting
                line = line.lstrip()
                line = line.lstrip('#')
        line = line[0:startlen]
        if line == key_input:
            L.append(i)
    if not L:
        L = [len(tempfile)]
    return L

def run_command(command):
    #run 'command' as a shell command
    #  'command' must be a list of strings
    subprocess.call(command,shell=False)

def run_command_file(command, filename):
    #run 'command' as a shell command and save output to filename
    #  'command' must be a list of strings
    with open(filename, "w") as outfile:
        subprocess.call(command, stdout = outfile)

def readfile(filename):
    #read a file into a list of strings
    f = open(filename,'r')
    tempfile = f.readlines()
    f.close()
    return tempfile

def writefile(filename,tempfile):
    #write tempfile (list of strings) to filename
    f = open(filename,'w')
    f.writelines(tempfile)
    f.close()

def readacf(acffile,Nat):
    #get atom electrons from ACF.dat
    tempfile = readfile(acffile)
    tempfile = [x.split() for x in tempfile]
    chg = [float(tempfile[x][4]) for x in range(2,2+Nat)]
    line = 2+Nat+1
    vacchg = float(tempfile[line][2])
    vacvol = float(tempfile[line+1][2])
    Nelec = float(tempfile[line+2][3])
    return chg,[vacchg,vacvol,Nelec]
    
def parse_args():
    #get arguments from command line
    parser = argparse.ArgumentParser(description="Script to do Bader analysis from JDFTx calculation")
    parser.add_argument("-f", type=str,default='out', help="output file to read (default: out)")
    parser.add_argument('-o', type=str,default='CHGCAR', help='name of CHGCAR file to write (default: CHGCAR)')
    parser.add_argument('-p', type=str,default='jdft', help='prefix of .n files (default: jdft)')
    parser.add_argument('-i', type=int,default=2,help='set Ninterp in createVASP (default: 2)')
    args = parser.parse_args()
    return args

def main():

    #store arguments
    args = parse_args()
    outfile = args.f
    chgcar = args.o
    prefix = args.p
    Ninterp = args.i

    #hardcode in the read of relevant info from out file
    tempfile = readfile(outfile)
    #get number atoms
    line = find_key('total atoms',tempfile)
    Nat = int(float(tempfile[line].split()[4]))
    #determine number of spins so whether need to read .n or .n_up and .n_dn files
    line = find_key('spintype',tempfile)
    spintype = tempfile[line].split()[1]
    #get neutral atom valence electrons
    endline = find_key('Setting up symmetries',tempfile)
    ionslines = find_first_range_key('ion ',tempfile,startline=0,endline=endline)
    ions = [tempfile[x].split()[1] for x in ionslines]
    atomlines = find_all_key('  Title:',tempfile)
    atoms = [tempfile[x].split()[1].replace('.','') for x in atomlines]

    vallines = find_all_key('valence electrons',tempfile)
    valelec = [float(tempfile[x].split()[4]) for x in vallines]
    atomtypes = dict(zip(atoms,valelec))
    if spintype == 'z-spin' or spintype == 'vector-spin':
        Nspin = 2
    elif spintype == 'no-spin' or spintype == 'spin-orbit':
        Nspin = 1
    else:
        print('Could not determine spin, assuming no spin')
        Nspin = 1 

    #make chgcars based on number of spins using jdftx utility script
    if Nspin == 1:
        #make VASP CHGCAR
        print('Making CHGCAR')
        com = ['createVASP',outfile,chgcar,prefix + '.n',str(Ninterp)]
        run_command(com)
        #run bader
        print('Run Bader on CHGCAR')
        com = ['bader',chgcar]
        run_command(com)

    else:
        #do same process for both spins
        print('Making CHGCAR_up')
        com = ['createVASP',outfile,chgcar+'_up','n_up',str(Ninterp)]
        run_command(com)
        print('Making CHGCAR_dn')
        com = ['createVASP',outfile,chgcar+'_dn','n_dn',str(Ninterp)]
        run_command(com)
        print('Run Bader on CHGCAR_up')
        com = ['bader',chgcar+'_up']
        run_command(com)
        shutil.move('ACF.dat','ACF.dat_up')
        print('Run Bader on CHGCAR_dn')
        com = ['bader',chgcar+'_dn']
        run_command(com)
        shutil.move('ACF.dat','ACF.dat_dn')

        #if multiple spins are present, add together spin up and spin down results into one file
        atomchgsup,totalchgsup = readacf('ACF.dat_up',Nat)
        atomchgsdn,totalchgsdn = readacf('ACF.dat_dn',Nat)
        totalchg = [atomchgsup[i]+atomchgsdn[i] for i in range(len(atomchgsup))]

        #write summed ACF.dat file
        print('Making final summed ACF.dat file')
        copy2('ACF.dat_up','ACF.dat')
        tempfile = readfile('ACF.dat')        
        tempfile[0] = '    #         X           Y           Z        CHARGE     OXIDATION STATE\n'
        for i in range(Nat):
            line = tempfile[i+2].split()[:4]
            line.append(str(totalchg[i]))
            line.append(str(atomtypes[ions[i]]-totalchg[i]))
            line = [float(x) for x in line]
            line[0] = int(line[0])
            tempfile[i+2] = '{:5d} {:11.4f} {:11.4f} {:11.4f} {:11.4f} {:15.4f}\n'.format(line[0],line[1],line[2],line[3],line[4],line[5])
        i = 2+Nat+1
        tempfile[i] = '    VACUUM CHARGE: {:20.4f}\n'.format(totalchgsup[0]+totalchgsdn[0])
        tempfile[i+1] = '    VACUUM VOLUME: {:20.4f}\n'.format(totalchgsup[1]+totalchgsdn[1])
        tempfile[i+2] = '    NUMBER OF ELECTRONS: {:14.4f}\n'.format(totalchgsup[2]+totalchgsdn[2])
        writefile('ACF.dat',tempfile)    


if __name__ == '__main__':
    main()


