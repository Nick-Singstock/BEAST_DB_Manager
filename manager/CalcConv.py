# Class to manage convergence logic
from os.path import exists as ope
from OutParser import OutParser

class CalcConv:
    def __init__(self, convergence_step, ctype):
        self.convergence_step = convergence_step
        self.ctype = ctype

    def get_parser(self, log_function, ctype=None):
        if ctype == None:
            ctype = self.ctype
        convergence_step = self.convergence_step
        if ope(f'parse_{convergence_step}'):
            # Current convergence step has been partially parsed

            print("parse option 1")
            with open(f'parse_{convergence_step}', 'r') as f:
                lines = f.readlines()
            # There is one number in the parse file which shows where the parser finished 
            # Reading the out file last time this calculation was run in the same convergence step
            last_parse_line = int(lines[0]) 
            log_function(f'parse_{convergence_step} found, current convergence step has already been partially parsed')
            parser = OutParser('out', start_line=last_parse_line, ctype=ctype)
            # Check if a calculation has been run since the last line written to the parse file.
            # This is relevant after the calculation ran and is being parsed
            calc_run = parser.check_calc_started()
            if calc_run == False:
                log_function(f'Beginning parsing after line {last_parse_line} in out file')
                return parser
            elif calc_run == True:
                log_function(f'Calculation has already been run since the last line written to the parse file')
                # Calculation has been run since the last line written to the parse file
                # so we need to start parsing from the beginning of the out file
                parser = OutParser('out', start_line=0, ctype=ctype)
                parser = parser.from_end_of_previous_step()
                new_start_line = parser.start_line
                log_function(f'Beginning parsing after line {new_start_line} in out file')
                print(f'new_start_line: {new_start_line}')
                return parser

        if ope(f'parse_{convergence_step - 1}'):
            print("parse option 2")
            # Previous convergence step was successfully finished and parsed,
            # so we read that file and start after that line

            with open(f'parse_{convergence_step -1}', 'r') as f:
                lines = f.readlines()
            # read in the last line of the previous parse file
            parse_line_from_prev_step = int(lines[0])
            parser = OutParser('out', start_line=parse_line_from_prev_step, ctype=ctype)
            calc_started = parser.check_calc_started()
            if calc_started == False:
                # calculation has not yet started so there is nothing to parse.
                # skip remaining steps for setting up a parser and return the parser as is
                print("calc has not started")
                return parser
            # re-initialize parser to start from the beginning of the JDFT run for this step
            parser = parser.from_end_of_previous_step()
            new_start_line = parser.start_line
            print(f'new_start_line: {new_start_line}')
            log_function(f'parse_{convergence_step -1 } found, previous convergence step was successfully finished and parsed')
            log_function(f'Beginning parsing after line {new_start_line} in out file')
            parser = OutParser('out', start_line=new_start_line, ctype=ctype)
        else:
            # Calculation is on first step so no parse file exists
            
            print("parse option 3")
            last_parse_line = 0
            log_function(f'parse_{convergence_step} not found, beginning parsing from beginning of out file')
            parser = OutParser('out', start_line=last_parse_line, ctype=ctype)
        return parser


    def check_convergence_from_parse_file_line(self, convergence_step):
        # Will check if the calculation converged on the current convergence step.
        # it initializes the outparser from the line in the convergence file.
        # If it encounters the Done! line, it returns True, because the calculation converged
        # If it encounters the JDFTx banner line, it returns False, because the calculation did not converge
        # before it was restarted

        with open(f'parse_{convergence_step}', 'r') as f:
            lines = f.readlines()
        last_line = int(lines[0])
        parser = OutParser('out', start_line=last_line, ctype=self.ctype)
        trimmed_text_lines = parser.trimmed_text_lines
        converged = False
        for line in trimmed_text_lines:
            if line.strip() == "Done!":
                converged = True
        return converged
    
    def check_parse_file_convergence(self, convergence_step):
        with open(f'parse_{convergence_step}', 'r') as f:
            lines = f.readlines()
        if lines[1].split()[1] == 'True':
            return True
        else:
            return False
    
    def check_parse_file_timeout(self, convergence_step):
        with open(f'parse_{convergence_step}', 'r') as f:
            lines = f.readlines()
        if lines[2].split()[1] == 'True':
            return True
        else:
            return False

    def calc_running(self):
        if ope('calc_running'):
            return True
        else:
            return False
    
    def check_step_convergence(self, convergence_step):
        if ope(f'parse_{convergence_step}'):
            return self.check_parse_file_convergence(convergence_step)
        else:
            return False