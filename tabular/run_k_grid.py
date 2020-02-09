"""
In  tabular MDP setting, evaluates the learning of optimal policy using different guidance discount factors
On-policy means we run episodes,
 in each episode we generate roll-outs/trajectories of current policy and run algorithm to improve the policy.
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse
from copy import deepcopy
import timeit, time
from tabular.main_control import run_main

try:
    from tabular.mdp_utils import MDP, SetMdpArgs
    from tabular.planing_utils import GetUniformPolicy, PolicyEvaluation, PolicyIteration, generalized_greedy
    from tabular.learning_utils import ModelEstimation, TD_Q_evaluation, TD_Q_evaluation_given_pol
    from common.utils import set_random_seed, create_result_dir, save_run_data, load_run_data, write_to_log, get_grid
except ImportError as error:
    raise ImportError(str(error) + ', try this:  1. Go to main project dir 2. $ python3 setup.py install ')

plt_params = {'font.size': 10,
          'lines.linewidth': 2, 'legend.fontsize': 16, 'legend.handlelength': 2,
          'pdf.fonttype':42, 'ps.fonttype':42,
          'axes.labelsize': 16, 'axes.titlesize': 16,
          'xtick.labelsize': 12, 'ytick.labelsize': 12}
plt.rcParams.update(plt_params)

# -------------------------------------------------------------------------------------------
#  Run mode
# -------------------------------------------------------------------------------------------

load_run_data_flag = False  # False/True If true just load results from dir, o.w run simulation
result_dir_to_load = './saved/2019_12_29_15_20_46'
save_PDF = False  # False/True - save figures as PDF file

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
# ----- Run Parameters ---------------------------------------------#
parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)',
                    default='')
parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)
parser.add_argument('--n_reps', type=int,  help='number of experiment repetitions',
                    default=1000)  # 1000

# ----- Model Parameters ---------------------------------------------#

parser.add_argument('--depth', type=int,  help='Length of trajectory',
                    default=10)
parser.add_argument('--gammaEval', type=float,  help='gammaEval',
                    default=0.99)
parser.add_argument('--sampling_type', type=str,  help='How to generate data',
                    default='Trajectories')  # 'Trajectories' |   'Generative'

# ----- Algorithm Parameters ---------------------------------------------#
parser.add_argument('--method', type=str,  help='RL Algorithm',
                    default='Expected_SARSA')  # 'Model_Based' | 'SARSA' | Expected_SARSA

parser.add_argument('--TD_Init_type', type=str,  help='How to initialize V',
                    default='R_max')  # 'zero' |   'random_0_1' |  'random_0_max' | 'R_max'

parser.add_argument('--n_TD_iter', type=float,  help='number of TD iterations',
                    default=500)  # 500 for RandomMDP, 5000 for GridWorld

args = parser.parse_args()


# Additional args:

# MDP definition ( see data_utils.SetMdpArgs)
args.mdp_def = {'type': 'RandomMDP', 'S': 10, 'A': 2, 'k': 5, 'reward_std': 0.1}
# args.mdp_def = {'type': 'GridWorld', 'N0': 4, 'N1': 4,  'reward_std': 0.1}
# args.mdp_def = {'type': 'GridWorld2', 'N0': 4, 'N1': 4,  'reward_std': 0.1}

#  how to create parameter grid:


args.learning_rate_def = {'type': 'a/(b+i_iter)', 'a': 50, 'b': 1000}      #  #  {'type': 'a/(b+i_iter)', 'a': 100, 'b': 1000}  |  {'type': 'const', 'alpha': 0.1}   | {'type': 'a/(b+i_sqrt(iter))', 'a': 100, 'b': 1000}



args.n_episodes = 5  # Number of episodes
args.n_traj_grid = [5]    # [1,2,4,10]  #  [5,10,20,50]     [1,2,4,10]   #  grid of number of trajectories to generate per episode
args.epsilon = 0.3


# -------------------------------------------------------------------------------------------
def run_simulation(args):
    import ray
    start_time = timeit.default_timer()
    create_result_dir(args)
    set_random_seed(args.seed)

    k_grid = np.arange(1, 6)
    n_grid = len(k_grid)
    no_reg_err_mean = np.zeros(n_grid)
    no_reg_err_std = np.zeros(n_grid)
    best_gamma_err_mean = np.zeros(n_grid)
    best_gamma_err_std = np.zeros(n_grid)
    best_l2_err_mean = np.zeros(n_grid)
    best_l2_err_std = np.zeros(n_grid)

    for i_k, k in enumerate(k_grid):
        args_run = deepcopy(args)
        args_run.mdp_def['k'] = k

        # Run gamma grid
        args_run.param_grid_def = {'type': 'gamma_guidance', 'spacing': 'linspace', 'start': 0.1, 'stop': 0.99, 'num': 50}
        alg_param_grid = get_grid(args_run.param_grid_def)
        info_dict = run_main(args_run, save_result=False)
        planing_loss_avg = info_dict['planing_loss_avg']
        planing_loss_std = info_dict['planing_loss_std']
        # Mark the best gamma:
        i_best = np.argmin(planing_loss_avg[0])
        best_gamma_err_mean[i_k] = planing_loss_avg[0][i_best]
        best_gamma_err_std[i_k] = planing_loss_std[0][i_best]

        args_run.param_grid_def = {'type': 'L2_factor', 'spacing': 'linspace', 'start': 0.0, 'stop': 0.01, 'num': 50}
        alg_param_grid = get_grid(args_run.param_grid_def)
        info_dict = run_main(args_run, save_result=False)
        planing_loss_avg = info_dict['planing_loss_avg']
        planing_loss_std = info_dict['planing_loss_std']
        # Mark the best gamma:
        i_best = np.argmin(planing_loss_avg[0])
        best_l2_err_mean[i_k] = planing_loss_avg[0][i_best]
        best_l2_err_std[i_k] = planing_loss_std[0][i_best]

        no_reg_err_mean[i_k] = planing_loss_avg[0][0]
        no_reg_err_std = planing_loss_std[0][0]
    # end for
    grid_results_dict = {'k_grid': k_grid, 'best_gamma_err_mean': best_gamma_err_mean,
                         'best_gamma_err_std': best_gamma_err_std,
                         'best_l2_err_mean': best_l2_err_mean, 'best_l2_err_std': best_l2_err_std,
                         'no_reg_err_mean': no_reg_err_mean, 'no_reg_err_std': no_reg_err_std}
    save_run_data(args, grid_results_dict)
    stop_time = timeit.default_timer()
    write_to_log('Total runtime: ' +
                 time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)), args)
    return grid_results_dict
# end  run_simulations
# -------------------------------------------------------------------------------------------

if load_run_data_flag:
    args, grid_results_dict = load_run_data(result_dir_to_load)
else:
   grid_results_dict = run_simulation(args)

k_grid = [int(k) for k in grid_results_dict['k_grid']]
best_gamma_err_mean = grid_results_dict['best_gamma_err_mean']
best_gamma_err_std = grid_results_dict['best_gamma_err_std']
best_l2_err_mean = grid_results_dict['best_l2_err_mean']
best_l2_err_std = grid_results_dict['best_l2_err_std']
no_reg_err_mean = grid_results_dict['no_reg_err_mean']
no_reg_err_std = grid_results_dict['no_reg_err_std']

ax = plt.figure().gca()
ci_factor = 1.96 / np.sqrt(args.n_reps)  # 95% confidence interval factor
plt.errorbar(k_grid, no_reg_err_mean, no_reg_err_std * ci_factor, label='No Regulrization')
plt.errorbar(k_grid, best_gamma_err_mean, best_gamma_err_std * ci_factor, label='Best Discount Regulrization')
plt.errorbar(k_grid, best_l2_err_mean, best_l2_err_std * ci_factor, label='Best L2 Regulrization')
plt.grid(True)
plt.legend()
plt.xlabel(r'Connectivity parameter $k$ ')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('Loss')
if save_PDF:
    plt.savefig(args.run_name + '.pdf', format='pdf', bbox_inches='tight')
else:
    plt.title('Loss +- 95% CI \n ' + str(args.run_name))

plt.show()
print('done')
