This is the main control file for using gc_manager.py in a managed directory.
Please read this header section if you are unfamiliar with the manager script.

jdft manager was built to manage a large number of JDFTx surface calculations.
	This script will run unbiased bulk and surface calculations, add requested biases to surfaces,
	add adsorbate molecules to requested surface sites, create desorbed calculations and
	NEB pathways. It will also rerun unconverged jobs and compile and analyze converged
	calculations. All data is saved in the file results/all_data.json and can be used
	for further analysis with python. 

To begin, please add a surface POSCAR to a subfolder (e.g. SURF_01) in the calcs/surfs/ directory.
	To manage this calculation, please add calculation parameters below. 
	Note: New surfaces can be created from bulk structures with gen_surface.py

	An example of how to manage a surface is included below for a surface named SURF_01 
	which is to be converged first with no bias, then at 0 V, then at -0.1 and -0.5 V. 
	Note: No bias (None) must always be included. 
	Note: 0 V must be included in order to run additional biases.
	Note: biases should be listed in Volts (RHE). "target-mu" will be updated based on "pcm-variant".
	
	"=" tag indicates a new surface
	"+" tags refer to specific changes made to "inputs" file for respective surface.
		All nested calculations for the surface inherit the same tag changes. 
		Molecule specific changes can also be made separate from surface. pH can be set as a tag.
	"-" adds a molecule to the surface above the listed atoms. Can be 1) name of atom on surface, 
		2) specific atom index, or 3) 'All' positions on surface
	
	Keywords -- for molecules/adsorbates these include: 
		"Biases:" - List of all biases for adsorbate calcs, must be subset of surface biases. 
					Also needed for all surfaces. Must be separated by ", ". Biases in RHE by default.
		"Desorb:" - Creates desorbed state for NEB calcs at listed biases
		"NEB:" - Creates an NEB calc, needs following params:
				 1. Adsorbed state folder (e.g., 01). This should be the folder name in calcs/
				 2. Bias for NEB (e.g., -0.1 V). Must be within molecule biases list.
				 3. Images to run (e.g., 7)
				 4. fmax tag (e.g., 0.05)
	Note: Inclusion of a molecule to be an adsorbate will run the corresponding molecule in the 
	molecules/ folder at all requested biases in order to get binding energies.	

=SURF_01
Biases: [None, 0, -0.1, -0.5]
+kpoint-folding 4 4 1
+fmax 0.02

-MOL_01: [10, 11] 
Biases: [0, -0.1]
Desorb: [0, -0.1]

NEB: 01 -0.1 7 0.05
NEB: 01 0 7 0.05

-MOL_02: [10]
Biases: [-0.5]


----- CALCULATIONS BELOW -----





