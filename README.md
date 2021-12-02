# Overview

This repository contains the data and code needed to reproduce the main results from our paper:

C.Gagne and P.Dayan, Peril, prudence and planning as risk, avoidance and worry. Journal of Mathematical Psychology (2021) 102617, https://doi.org/10.1016/j.jmp.2021.102617.

# Installations

The code has only a few requirements, which can installed by running:

`conda env create -f environment.yml`


# Code organization

The code is organized according to the sections of the paper.

Code for:
- calculating CVaR for a single choice (Figure 1) is in 'cvar_single_choice'.
- plotting the nCVaR/pCVaR tree (Figure 2) is in 'cvar_tree'
- evaluating nCVaR/pCVaR for a random policy (Figure 3) is in 'policy_evaluation'
- calculating the optimal nCVaR/pCVaR policies (Figures 4-5) is in 'policy_optimization'
- obtaining the optimal greedy replay sequences (Figures 6-7) are in 'replay optimization'

Policy evaluation, optimization, and replay shares code in the folder 'shared'. Here, you can find the nCVaR/pCVaR Bellman evaluation and optimality operators.

# Simulation results

All of the simulations have already been run and the results are stored in 'simulation_results'. The '.mat' files were created using code privately borrowed from Chow et al. 2015 (not contained in the repo); these files contain some information about the MDPs, which is reused by our scripts and also were used to check our implementation versus theirs. These should not be deleted.
