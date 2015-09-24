#test ckchou
# DecayCascades

**DecayCascades**, a multiple diffusion networks model, is a generative model to deal with diffusion network inference problem.

We also provide several state of the arts of diffusion network model listed below.

* InfoPath
* MMRate
* MixCascades

Please refer to my paper for more details.

## Dependency
* Openmp
* SNAP 2.3 Library @ http://snap.stanford.edu/snap/download.html
* Gnuplot

Note:  
You should compile the SNAP core, cascdynetinf and kronecker before doing installation.  
  * SNAP core is in *\<SNAP Library Path\>/snap-core* directory.  
  * cascdynetinf and kronecker are in *\<SNAP Library Path\>/snap-adv* directory.  

After compilation put the compiled objective files i.e. Snap.o, cascdynetinf.o and kronecker.o into *lib* directory. 

## Installation

Get codes from Github.

`git clone https://github.com/plliao/DecayCascades.git`

`cd DecayCascades`

Set the SNAP library path in Makefile

`vim Makefile`

Find "SnapDirPath = ../Snap-2.3" in *Makefile* and set your SNAP Library path.

Create the required directories.

`mkdir obj bin`

Compile codes using *make* command.

`make -j 4`

The compiled programs are in the *bin* directory.

## Usage

Please check all model parameters listed in the corresponging cpp file. 

We provide an example in *exp.sh* script for you to refer.

Note that you should create the required directories before you run the script i.e. `mkdir plot result data`.

### Program Descriptions
#### Models
* InfoPath.cpp: main file of InfoPath model  
* MMRate.cpp: main file of MMRate model  
* MixCascades.cpp: main file of MixCascades model  
* DecayCascades.cpp: main file of DecayCascades model

#### Evaluations
* EvaluationAUC.cpp: PRC AUC evaluation main file  
* EvaluationMSE.cpp: MSE evaluation main file  
* EvaluationMultiple.cpp: multiple network evaluation main file  

#### Utility
* generate_DCnets.cpp: cascades and network generator using our diffusion model  
* DataMerger.cpp: a program to merge several cascades and network files into single file.

## Reference
1. *InfoPath*, Structure and Dynamics of Information Pathways in On-line Media, in WSDM 2013
2. *MMRate*, MMRate: inferring multi-aspect diffusion networks with multi-pattern cascades, in KDD 2014
3. *MixCascades*, Cluster cascades: Infer multiple underlying networks using diffusion data, in ASONAM 2014
4. *DecayCascades*, Uncovering Multiple Diffusion Networks Using the First-Hand Sharing Pattern
