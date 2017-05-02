# FASTEN

**FASTEN** is a generative model to solve diffusion network inference problem.

Our model infers multiple diffusion networks using the first-hand sharing pattern.

We also provide several diffusion network models listed below.

* InfoPath
* MMRate
* MixCascades

Please refer to my paper for more details.

```
Uncovering Multiple Diffusion Networks Using the First-Hand Sharing Pattern
Pei-Lun Liao, Chung-Kuang Chou, and Ming-Syan Chen
Proceedings of the 2016 SIAM International Conference on Data Mining. 2016, 63-71 
```

## Dependency
* Openmp
* SNAP 2.3 Library @ http://snap.stanford.edu/snap/releases.html
* Gnuplot

Note:  
You should compile the `SNAP core`, `cascdynetinf.cpp` and `kronecker.cpp` before doing installation.  

  * `SNAP core` is in `<Your SNAP Library Path>/snap-core` directory.
  
    To compile `SNAP core`.
      ```
      cd <Your SNAP Library Path>/snap-core
      make
      ```
      
  * `cascdynetinf.cpp` and `kronecker.cpp` are in `<Your SNAP Library Path>/snap-adv` directory.  
  
    To compile `cascdynetinf.cpp` and `kronecker.cpp`
    
    ```
    cd <Your SNAP Library Path>/snap-adv
    g++ -c cascdynetinf.cpp -o cascdynetinf.o -I ../snap-core -I ../glib-core ../snap-core/Snap.o
    g++ -c kronecker.cpp -o kronecker.o -I ../snap-core -I ../glib-core ../snap-core/Snap.o
    ```

After compilation put the compiled objective files i.e. `Snap.o`, `cascdynetinf.o` and `kronecker.o` into `lib` directory.

## Installation

Get codes from Github.

`git clone https://github.com/plliao/FASTEN.git`

`cd FASTEN`

Set the SNAP library path in Makefile

`vim Makefile`

Find `SnapDirPath = ../Snap-2.3` in Makefile and set your SNAP Library path.

Create the required directories.

`mkdir obj bin`

Compile codes using `make` command.

`make -j 4`

The compiled programs are in the `bin` directory.

## Usage

Please check all model parameters listed in the corresponging cpp file. 

We provide an example in `exp.sh` script for you to refer.

To run the script

`./exp.sh`

Note that you should create the required directories before you run the script i.e. `mkdir plot result data`.

### Program Descriptions
#### Models
* InfoPath.cpp: main file of InfoPath model  
* MMRate.cpp: main file of MMRate model  
* MixCascades.cpp: main file of MixCascades model  
* FASTEN.cpp: main file of FASTEN model

#### Evaluations
* EvaluationAUC.cpp: PRC AUC evaluation file  
* EvaluationMSE.cpp: MSE evaluation file  
* EvaluationMultiple.cpp: multiple network evaluation file  

#### Utility
* generate_FASTEN_nets.cpp: cascades and network generator using our diffusion model  
* DataMerger.cpp: a program to merge several cascades and network files into single file.

## Reference
1. *InfoPath*, Structure and Dynamics of Information Pathways in On-line Media, at WSDM 2013
2. *MMRate*, MMRate: inferring multi-aspect diffusion networks with multi-pattern cascades, at KDD 2014
3. *MixCascades*, Cluster cascades: Infer multiple underlying networks using diffusion data, at ASONAM 2014
4. *FASTEN*, Uncovering Multiple Diffusion Networks Using the First-Hand Sharing Pattern, at SDM 2016
