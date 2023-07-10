# Banditos-2023


## Directory Structure
```
│
├── data    
│   ├── plots             <- Plots of meaningful results
│   └── ???               <- ???
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
- The scripts `lin_ucb_regret_analysis.py` and `ts_regret_analysis.py` are the scripts for part 1, 
analysis of the regret in an influence maximization setting for both a Thomspon Sampling like bandit
and a linear UCB bandit.  
- The script `matching_ucb_regret_analysis.py` is the script for part 2, it's the analysis for the regret
in a matching setting for a UCB bandit.   
- The script `influence_matching_ucb_regret_analysis.py` is for part 3, we jointly use UCB bandits for learning
the influence probabilities and then the mean of the rewards in a matching setting.
