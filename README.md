# DecayCascades

**DecayCascades**, a multiple diffusion networks model, is a generative model to deal with diffusion network inference problem.

We also provide several state of the arts of diffusion network models listed below.

1. InfoPath
2. MMRate
3. MixCascades

## Dependency
1. Openmp
2. SNAP 2.3 Library @ http://snap.stanford.edu/snap/download.html
3. Gnuplot

## Installation

Get codes from Github.

`git clone`

`cd DecayCascades`

Set the SNAP library path in Makefile

`vim Makefile`

Find "SnapDirPath = ../Snap-2.3" in *Makefile* and set your SNAP Library path.

Build the required directories.

`mkdir obj bin`

Compile codes using *make* command.

`make -j 4`

The compiled programs are in the *bin* directory.

## Usage

Please check all model parameters listed in the corresponging cpp file. 

We provide an example in *exp.sh* script for you to refer.

Note that you should make the required directories to run the script i.e. `mkdir plot result data`.

## Reference
1. *InfoPath*, Structure and Dynamics of Information Pathways in On-line Media in WSDM 2013
2. *MMRate*, MMRate: inferring multi-aspect diffusion networks with multi-pattern cascades in KDD 2014
3. *MixCascades*, Cluster cascades: Infer multiple underlying networks using diffusion data in ASONAM 2014
4. *DecayCascades*, Uncovering Multiple Diffusion Networks Using the First-Hand Sharing Pattern
