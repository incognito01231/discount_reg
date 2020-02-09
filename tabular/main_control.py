"""
In  tabular MDP setting, evaluates the learning of optimal policy using different guidance discount factors
On-policy means we run episodes,
 in each episode we generate roll-outs/trajectories of current policy and run algorithm to improve the policy.
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse
import timeit, time

from tabular.mdp_utils import MDP, SetMdpArgs
from tabular.planing_utils import PolicyEvaluation, PolicyIteration
from tabular.learning_utils import ExpectedSARSA_Learning, ModelBasedLearning, SARSA_Learning
from common.utils import set_random_seed, create_result_dir, save_run_data, load_run_data, write_to_log, get_grid

plt_params = {'font.size': 10,
          'lines.linewidth': 2, 'legend.fontsize': 16, 'legend.handlelength': 2,
          'pdf.fonttype':42, 'ps.fonttype':42,
          'axes.labelsize': 16, 'axes.titlesize': 16,
          'xtick.labelsize': 12, 'ytick.labelsize': 12}
plt.rcParams.update(plt_params)

# -------------------------------------------------------------------------------------------
#  Run mode
# -------------------------------------------------------------------------------------------

local_mode = False  # True/False - run non-parallel to get error messages and debugging
save_PDF = False  # False/True - save figures as PDF file
# Option to load previous run results:
load_run_data_flag = False  # False/True If true just load results from dir, o.w run simulation
result_dir_to_load = './saved/2019_12_29_11_09_25'  #  2019_12_28_16_27_54 | 2019_12_28_11_06_17 | 2019_12_29_08_44_52 | 2019_12_29_11_09_25

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
parser.add_argument('--default_gamma', type=float,  help='The default guidance discount factor (if None use gammaEval)',
                    default=None)
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

#  how to create parameter grid:
args.param_grid_def = {'type': 'gamma_guidance', 'spacing': 'linspace', 'start': 0.1, 'stop': 0.99, 'num': 50}
# args.param_grid_def = {'type': 'L2_factor', 'spacing': 'linspace', 'start': 0.0, 'stop': 0.01, 'num': 50}
# args.param_grid_def = {'type': 'L2_factor', 'spacing': 'linspace', 'start': 0.0, 'stop': 0.002, 'num': 50}
# args.param_grid_def = {'type': 'L1_factor', 'spacing': 'linspace', 'start': 0.0, 'stop': 1.2, 'num': 50}

args.learning_rate_def = {'type': 'a/(b+i_iter)', 'a': 50, 'b': 1000}      #  #  {'type': 'a/(b+i_iter)', 'a': 100, 'b': 1000}  |  {'type': 'const', 'alpha': 0.1}   | {'type': 'a/(b+i_sqrt(iter))', 'a': 100, 'b': 1000}

args.n_episodes = 5  # Number of episodes
args.n_traj_grid = [1, 2, 4, 10]  #  RandomMDP [1,2,4,10] #  GridWorld  [5,10,15,20]   #  grid of number of trajectories to generate per episode
args.epsilon = 0.3
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
def run_simulations(args, save_result, local_mode):
    import ray
    ray.init(local_mode=local_mode, ignore_reinit_error=True),
    # A Ray remote function.
    # Runs a single repetition of the experiment
    @ray.remote
    def run_rep(i_rep, alg_param_grid, n_traj_grid, args_r):
        traj_grid_len = len(n_traj_grid)
        n_grid = len(alg_param_grid)

        # runs a single repetition of the experiment
        loss_rep = np.zeros((traj_grid_len, n_grid))

        # default values
        gammaEval = args_r.gammaEval
        if args_r.default_gamma is None:
            gamma_guidance = gammaEval
        else:
            gamma_guidance = args_r.default_gamma
        l2_factor = None
        l1_factor = None

        # Generate MDP:
        M = MDP(args_r)

        # Optimal policy for the MDP:
        pi_opt, V_opt, Q_opt = PolicyIteration(M, gammaEval)

        for i_grid, alg_param in enumerate(alg_param_grid):

            if args_r.param_grid_def['type'] == 'L2_factor':
                l2_factor = alg_param
            elif args_r.param_grid_def['type'] == 'L1_factor':
                l1_factor = alg_param
            elif args_r.param_grid_def['type'] == 'gamma_guidance':
                gamma_guidance = alg_param
            else:
                raise ValueError('Unrecognized args.grid_type')

            for i_n_traj, n_traj in enumerate(args_r.n_traj_grid):  # grid of number of trajectories to generate
                if args_r.method == 'Expected_SARSA':
                    pi_t = ExpectedSARSA_Learning(args_r, M, n_traj, gamma_guidance, l2_factor, l1_factor)
                elif args_r.method == 'Model_Based':
                    pi_t = ModelBasedLearning(args_r, M, n_traj, gamma_guidance)
                elif args_r.method == 'SARSA':
                    pi_t = SARSA_Learning(args_r, M, n_traj, gamma_guidance)
                else:
                    raise ValueError('unrecognized method')
                # Evaluate performance of policy:
                V_t, _ = PolicyEvaluation(M, pi_t, gammaEval)
                loss_rep[i_n_traj, i_grid] = (np.abs(V_opt - V_t)).mean()
            # end for i_n_traj
        #  end for i_grid
        return loss_rep

    # end run_rep
    # --------------------------------------------------
    start_time = timeit.default_timer()
    if save_result:
        create_result_dir(args)
    set_random_seed(args.seed)

    n_reps = args.n_reps
    alg_param_grid = get_grid(args.param_grid_def)
    n_grid = alg_param_grid.shape[0]
    traj_grid_len = len(args.n_traj_grid)
    planing_loss = np.zeros((n_reps, traj_grid_len, n_grid))

    # ----- Run simulation in parrnell process---------------------------------------------#
    loss_rep_id_lst = []
    for i_rep in range(n_reps):
        # returns objects ids:
        planing_loss_rep_id = run_rep.remote(i_rep, alg_param_grid, args.n_traj_grid, args)
        loss_rep_id_lst.append(planing_loss_rep_id)
    # -----  get the results --------------------------------------------#
    for i_rep in range(n_reps):
        loss_rep = ray.get(loss_rep_id_lst[i_rep])
        write_to_log('Finished: {} out of {} reps'.format(i_rep + 1, n_reps), args)
        planing_loss[i_rep] = loss_rep
    # end for i_rep
    info_dict = {'planing_loss_avg': planing_loss.mean(axis=0), 'planing_loss_std': planing_loss.std(axis=0),
                 'alg_param_grid': alg_param_grid}
    if save_result:
        save_run_data(args, info_dict)
    stop_time = timeit.default_timer()
    write_to_log('Total runtime: ' +
                 time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)), args)
    return info_dict
# end  run_simulations
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def run_main(args, save_result=True, load_run_data_flag=False, result_dir_to_load='', save_PDF=False, plot=True, local_mode=False):

    SetMdpArgs(args)
    if load_run_data_flag:
        args, info_dict = load_run_data(result_dir_to_load)
    else:
        info_dict = run_simulations(args, save_result, local_mode)
    planing_loss_avg = info_dict['planing_loss_avg']
    planing_loss_std = info_dict['planing_loss_std']
    alg_param_grid = info_dict['alg_param_grid']
    n_reps = args.n_reps

    # ----- Plot figures  ---------------------------------------------#
    if plot:
        ax = plt.figure().gca()
        ci_factor = 1.96/np.sqrt(n_reps)  # 95% confidence interval factor
        for i_n_traj, n_traj in enumerate(args.n_traj_grid):  # grid of number of trajectories to generate
            total_samples = args.depth *  n_traj
            plt.errorbar(alg_param_grid, planing_loss_avg[i_n_traj], yerr=planing_loss_std[i_n_traj] * ci_factor,
                         marker='.', label='{} '.format(total_samples))
            # Mark the best gamma:
            i_best = np.argmin(planing_loss_avg[i_n_traj])
            plt.scatter(alg_param_grid[i_best], planing_loss_avg[i_n_traj][i_best], marker='*', s=400)
        plt.grid(True)
        if args.param_grid_def['type'] == 'L2_factor':
            plt.xlabel(r'$L_2$ Regularization Factor ')
        elif args.param_grid_def['type'] == 'L1_factor':
            plt.xlabel(r'$L_1$ Regularization Factor ')
        elif args.param_grid_def['type'] == 'gamma_guidance':
            plt.xlabel(r'Guidance Discount Factor $\gamma$')
        else:
            raise ValueError('Unrecognized args.grid_type')
        plt.ylabel('Loss')
        plt.legend()
        # plt.xlim([0.4,0.8])
        # ax.set_yticks(np.arange(0., 9., step=1.))
        if save_PDF:
            plt.savefig(args.run_name + '.pdf', format='pdf', bbox_inches='tight')
        else:
            plt.title('Loss +- 95% CI \n ' + str(args.run_name))
                      #'\n' r'Average absolute estimation error of $V^*(s)$ +- 95% CI'
        plt.show()
    # end if plot
    print('done')
    return info_dict
# end run_main
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    info_dict = run_main(args, save_result=True, load_run_data_flag=load_run_data_flag,
                         result_dir_to_load=result_dir_to_load, save_PDF=save_PDF, local_mode=local_mode)
