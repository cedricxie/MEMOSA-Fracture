1.	Clone the repository of MEMOSA
There are two ways to do it:
a.	Check out from https://nanohub.org/tools/memosa/svn
b.	

Fork the repository at https://github.com/c-PRIMED/fvm
i.	Click fork button to make your own copy:
ii.	Use command to download to the cluster: git clone https://github.com/yourusername/fvm.git


2.	Clone the repository of Fracture Module for MEMOSA
a.	Fork the repository at https://github.com/cedricxie/MEMOSA-Fracture
b.	Different branches are for different material models, namely:
a.	
Explicit-Unsym-TranIsotropic
Transversely Isotropic
Implicit-Sym-Lambda-AllEigenvalue-Mu 
Symmetric
Explicit-Unsym-Kappa-AllEigenvalue-Mu
Amor’s Model
Explicit-Unsym-Lambda-PositiveEigenvalue-Mu
Miehe’s Model
Explicit-PositivePrincipalStress
Failure by positive principal stress

c.	Similar as 1.b.ii, 
i.	Either clone the files directly to the cluster, or
ii.	Use desktop client of github from https://desktop.github.com/ for easier access with GUI. And upload the code to cluster manually later.1.	3.	In order to switch between different material models:
a.	Overwrite the original codes at /src/fvm/src/modules/fvmbase/ with the codes inside folder FractureCode and StructureCode
b.	Compile the codes using command: ./make.py configname (Configurations are stored in the 'configs' subdirectory.)
c.	To test, use command: ./make.py --test configname


4.	Before running the simulations, do a comparison against the example code at homoCase/homoCase.py to make sure:
a.	the elastic energy is correctly calculated
b.	the variable SymFlag is set to be the right value
c.	it would be the best if you understand any other differences in the scripts
