#!/usr/bin/env python
#%%

import numpy as np
import sys
import pickle
import os
import scipy
import scipy.stats
import argparse

import pCVaR_Prioritized_Value_Iteration_XY
from pCVaR_Prioritized_Value_Iteration_XY import pCVaR_Prioritized_Value_Iteration_XY

sys.path.append('../shared')
from scenarios import setup_params

def main():
    '''
    Example:

        python run_Prioritized_Value_Iteration_pCVaR_XY.py --example_name 1D --case 4 --ypr 5 --xeval 4 --iters 60 --subcase aS


    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--example_name',type=str,default='')
    parser.add_argument('--case',type=int,default=2)
    parser.add_argument('--subcase',type=str,default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ypr', type=int, default=11)
    parser.add_argument('--xeval', type=int, default=12)
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--special_init', type=str, default=None)
    args = parser.parse_args()

    # set random seed
    np.random.seed(int(args.seed))


    # Choose scenario
    example_name = args.example_name
    testcase = args.case
    yi_pr = args.ypr
    x_eval = args.xeval
    iters = args.iters
    subcase = args.subcase
    special_init = args.special_init
    p,maze = setup_params(testcase)

    pCVaR_Prioritized_Value_Iteration_XY(
                                       example_name,
                                       testcase,
                                       maze,
                                       yi_pr,
                                       x_eval,
                                       start_iter = 0,
                                       end_iter = iters,
                                       maxIter_inner = 100,
                                       converg_thresh_pr=0.05,#0.01,
                                       converg_thresh_nonpr=0.2,
                                       q_v_diff_eval_thresh=0.01,#0.001,
                                       init_me = True,
                                       parallel=True,
                                       verbose=False,
                                       subcase = subcase,
                                       special_init = special_init,
                                       set_values_as_rewards=True)



if __name__=='__main__':
    main()
