# Banditos-2023


## Directory Structure
```
│
├── utils    
│   └── plots             <- Plots of meaningful results
│
├── scripts               <- Scripts for running the experiments
│    ├── part_1           <- Learning for Social Influence
│    ├── part_2           <- Learning for Matching 
│    ├── part_3           <- Learning for joint Social Influence and Matching
│    ├── part_4           <- Context generation 
│    ├── part_5           <- Learning in a Non stationary environment  
│    └── part_6           <- Adversarial environment
│
│
└── algorithms
    ├── environments      <- Classes for simulation of the real world
    ├── bandits           <- Classes for bandit algorithms
    └── optimization      <- Classes for optimization algorithms
```
## How to run the experiments
In the folder scripts you can find the scripts for running the experiments required by the 6 parts.  
Every script use the modules contained in the folder `algorithms`.  
A brief description of the experiment is contained in each of the scripts, and for running the experiments
you'll have to set some initial parameters (at the top of the file).
Before launching the experiments:  
```pip3 -r install requirements.txt```
