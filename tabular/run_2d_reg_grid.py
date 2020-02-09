"""
In  tabular MDP setting, evaluates the learning of optimal policy using different guidance discount factors
On-policy means we run episodes,
 in each episode we generate roll-outs/trajectories of current policy and run algorithm to improve the policy.
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
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
result_dir_to_load = './saved/2020_02_04_06_19_35'  # '2020_02_04_06_19_35' | '2020_02_03_21_45_36'
save_PDF = False  # False/True - save figures as PDF file
local_mode = False  # True/False - run non-parallel to get error messages and debugging

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
                    default='R_max')  # 'R_max' |   'random_0_1' |  'random_0_max' | 'R_max'

parser.add_argument('--n_TD_iter', type=float,  help='number of TD iterations',
                    default=5000)  # 500 for RandomMDP, 5000 for GridWorld

args = parser.parse_args()

# Additional args:

# MDP definition ( see data_utils.SetMdpArgs)
# args.mdp_def = {'type': 'RandomMDP', 'S': 10, 'A': 2, 'k': 5, 'reward_std': 0.1}
args.mdp_def = {'type': 'GridWorld', 'N0': 4, 'N1': 4,  'reward_std': 0.1}
# args.mdp_def = {'type': 'GridWorld2', 'N0': 4, 'N1': 4,  'reward_std': 0.1}

#  how to create parameter grid:


args.learning_rate_def = {'type': 'a/(b+i_iter)', 'a': 50, 'b': 1000}      #  #  {'type': 'a/(b+i_iter)', 'a': 100, 'b': 1000}  |  {'type': 'const', 'alpha': 0.1}   | {'type': 'a/(b+i_sqrt(iter))', 'a': 100, 'b': 1000}



args.n_episodes = 5  # Number of episodes
args.n_traj_grid = [5]    # [1,2,4,10]  #  [5,10,20,50]     [1,2,4,10]   #  grid of number of trajectories to generate per episode
args.epsilon = 0.3

args.l2_grid_def = {'type': 'L2_factor', 'spacing': 'linspace', 'start': 0.0, 'stop': 0.01, 'num': 11}
args.gam_grid_def = {'type': 'gamma_guidance', 'spacing': 'linspace', 'start': 0.09, 'stop': 0.99, 'num': 11}


# -------------------------------------------------------------------------------------------
def run_simulations(args, local_mode):
    import ray
    ray.init(local_mode=local_mode)
    start_time = timeit.default_timer()
    create_result_dir(args)
    set_random_seed(args.seed)

    l2_grid = np.around(get_grid(args.l2_grid_def), decimals=4)
    gam_grid = np.around(get_grid(args.gam_grid_def), decimals=4)
    grid_shape = (len(l2_grid), len(gam_grid))
    loss_avg = np.zeros(grid_shape)
    loss_std = np.zeros(grid_shape)

    run_idx = 0
    for i0 in range(grid_shape[0]):
        for i1 in range(grid_shape[1]):
          args_run = deepcopy(args)
          args_run.param_grid_def = {'type': 'L2_factor', 'spacing': 'list', 'list': [l2_grid[i0]]}
          args_run.default_gamma = gam_grid[i1]

          info_dict = run_main(args_run, save_result=False, plot=False)
          loss_avg[i0, i1] = info_dict['planing_loss_avg'][0]
          loss_std[i0, i1] = info_dict['planing_loss_std'][0]
          run_idx += 1
          print("Finished {}/{}".format(run_idx, loss_avg.size))
        # end for
    # end for
    grid_results_dict = {'l2_grid': l2_grid, 'gam_grid': gam_grid, 'loss_avg': loss_avg,
                         'loss_std': loss_std}
    save_run_data(args, grid_results_dict)
    stop_time = timeit.default_timer()
    write_to_log('Total runtime: ' +
                 time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)), args)
    return grid_results_dict

# -------------------------------------------------------------------------------------------
if load_run_data_flag:
    args, grid_results_dict = load_run_data(result_dir_to_load)
else:
    grid_results_dict = run_simulations(args, local_mode)
l2_grid = grid_results_dict['l2_grid']
gam_grid = grid_results_dict['gam_grid']
loss_avg = grid_results_dict['loss_avg']
loss_std = grid_results_dict['loss_std']

ci_factor = 1.96 / np.sqrt(args.n_reps)  # 95% confidence interval factor
max_deviate = 100. * np.max(loss_std * ci_factor /  loss_avg)
print('Max 95% CI relative to mean: ', max_deviate, '%')

with sns.axes_style("white"):
    ax = sns.heatmap(loss_avg,  cmap="YlGnBu", xticklabels=gam_grid, yticklabels=l2_grid,  annot=True)
    plt.xlabel(r'Guidance Discount Factor $\gamma$')
    plt.ylabel(r'$L_2$ Regularization Factor ')
    if save_PDF:
        plt.savefig(args.run_name + '.pdf', format='pdf', bbox_inches='tight')
    else:
        plt.title('Loss avg. Max 95% CI relative to mean: {}%\n {}'.
                  format(np.around(max_deviate, decimals=1), args.run_name))
    plt.show()
    print('done')
