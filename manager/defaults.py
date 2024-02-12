# default arguments to gc_manager

# In order to be able to have the command line arguments 
# override the default values specified in the config.txt file,
# the defaults have to be turned off in the argument parser
# to keep track of which arguments the user manually specified.
# The defaults are then added back to the parser object after
# the user's arguments are parsed and the config arguments are
# added.

class DefaultArgs:
    def __init__(self):
        self.defaults = {
            'setup': 'False', 's': 'False', 
            'run_time': 12, 't': 12,
            'rerun_unconverged': 'True', 'u': 'True',
            'gpu': 'False', 'g': 'False',
            'make_new': 'True', 'm': 'True',
            'backup': 'False', 'b': 'False',
            'save_dos': 'False', 'sdos': 'False',
            'analyze_dos': 'False', 'ados': 'False',
            'read_all': 'False', 'ra': 'False',
            'analyze': 'False', 'a': 'False',
            'check_calcs': 'True', 'cc': 'True',
            'selective_dynamics': 'True', 'sd': 'True',
            'save': 'True', 'v': 'True',
            'run_new': 'True', 'rn': 'True',
            'add_adsorbed': 'True', 'ads': 'True',
            'add_desorbed': 'True', 'des': 'True',
            'add_molecules': 'True', 'mol': 'True',
            'make_neb': 'True', 'neb': 'True',
            'neb_climbing': 'True', 'nebc': 'True',
            'current_force': 'True', 'cf': 'True',
            'nodes': 1, 'n': 1,
            'cores': 32, 'c': 32,
            'short_recursive': 'False', 'r': 'False',
            'rhe_zeroed': 'False', 'rhe': 'False',
            'qos': 'False', 'q': 'False',
            'parallel': 1, 'p': 1,
            'use_convergence': 'True', 'conv': 'True',
            'kpoint_density': 1000, 'kptd': 1000,
            'kpoint_density_bulk': 1000, 'kptdb': 1000,
            'copy_electronic': 'False', 'elec': 'False',
            'smart_procs': 'True', 'sp': 'True',
            'calc_fixer': 'False', 'fix': 'False',
            'skip_fixer': 'False', 'skipfix': 'False',
            'full_rerun': 'False', 'fr': 'False',
            'clean_wfns': 'False', 'use_nb': 'True',
            'bundle_jobs': 'False', 'bundle': 'False',
            'use_nb': 'True', 'use_no_bias_structure': 'True',
            'big_mem': False, 'bm': False,
            'mean_dos': False, 'mdos': False,
            'singlepoint': False, 'sp': False,
            'gpus_per_job': 1, 'gpj': 1,
            'native_ionic': False, 'ni': False
        }
        # links connects the long and short versions of the arguments
        self.links = {
            'setup': 's', 's': 'setup',
            'run_time': 't', 't': 'run_time',
            'rerun_unconverged': 'u', 'u': 'rerun_unconverged',
            'gpu': 'g', 'g': 'gpu',
            'make_new': 'm', 'm': 'make_new',
            'backup': 'b', 'b': 'backup',
            'save_dos': 'sdos', 'sdos': 'save_dos',
            'analyze_dos': 'ados', 'ados': 'analyze_dos',
            'read_all': 'ra', 'ra': 'read_all',
            'analyze': 'a', 'a': 'analyze',
            'check_calcs': 'cc', 'cc': 'check_calcs',
            'selective_dynamics': 'sd', 'sd': 'selective_dynamics',
            'save': 'v', 'v': 'save',
            'run_new': 'rn', 'rn': 'run_new',
            'add_adsorbed': 'ads', 'ads': 'add_adsorbed',
            'add_desorbed': 'des', 'des': 'add_desorbed',
            'add_molecules': 'mol', 'mol': 'add_molecules',
            'make_neb': 'neb', 'neb': 'make_neb',
            'neb_climbing': 'nebc', 'nebc': 'neb_climbing',
            'current_force': 'cf', 'cf': 'current_force',
            'nodes': 'n', 'n': 'nodes',
            'cores': 'c', 'c': 'cores',
            'short_recursive': 'r', 'r': 'short_recursive',
            'rhe_zeroed': 'rhe', 'rhe': 'rhe_zeroed',
            'qos': 'q', 'q': 'qos',
            'parallel': 'p', 'p': 'parallel',
            'use_convergence': 'conv', 'conv': 'use_convergence',
            'kpoint_density': 'kptd', 'kptd': 'kpoint_density',
            'kpoint_density_bulk': 'kptdb', 'kptdb': 'kpoint_density_bulk',
            'copy_electronic': 'elec', 'elec': 'copy_electronic',
            'smart_procs': 'sp', 'sp': 'smart_procs',
            'calc_fixer': 'fix', 'fix': 'calc_fixer',
            'skip_fixer': 'skipfix', 'skipfix': 'skip_fixer',
            'full_rerun': 'fr', 'fr': 'full_rerun',
            'clean_wfns': 'clean_wfns', 'use_nb': 'use_nb',
            'bundle_jobs': 'bundle', 'bundle': 'bundle_jobs',
            'use_nb': 'use_nb', 'use_no_bias_structure': 'use_no_bias_structure',
            'big_mem': 'bm', 'bm': 'big_mem',
            'mean_dos': 'mdos', 'mdos': 'mean_dos',
            'singlepoint': 'sp', 'sp': 'singlepoint',
            'gpus_per_job': 'gpj', 'gpj': 'gpus_per_job',
            'native_ionic': 'ni', 'ni': 'native_ionic'
        }

    def set_defaults(self, args):
        '''
        args is a parser argparser object. The defaults will be added to the 
        parser object and returned. Also, any arguments that are linked in the
        argparser will be added to the parser object. 

        returns the same parser object with the defaults added
        '''
        args_dict = vars(args).copy()
        for key, value in args_dict.copy().items():
            linked_key = self.links[key]
            args_dict[linked_key] = value

        for key, value in self.defaults.items():
            if key in args_dict:
                pass
            elif key not in args_dict:
                args_dict[key] = value
        
        for key, value in args_dict.items():
            setattr(args, key, value)

        return args
            
            

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


# parser.add_argument('-gpj', '--gpus-per-job', help=('number of processes per job on gpus. Options are 1, 2, 4. '
#                                                             'Each job will be split into the specified number of tasks. '
#                                                             'This is useful for gaining memory headroom and speeding up calculations '
#                                                             'with more than one state.'),
#                     type=int, default=1, choices=[1,2,4])
# parser.add_argument('-ni', '--native_ionic', help=('whether to use the native_ionic script for running JDFTx.'
#                                                            'ionic optimization is handled through JDFTx and is much faster'
#                                                            'than ASE. However, it cannot do special constrained optimizations like NEBs'),
#                                                            type=bool, default=False)
# self.args = parser.parse_args()