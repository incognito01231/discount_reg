
from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse
import timeit, time

from tabular.mrp_utils import MRP, SetMrpArgs, calc_mixing_time
from tabular.planing_utils import GetUniformPolicy, PolicyEvaluation, PolicyIteration, generalized_greedy, evaluate_value_estimation
from tabular.learning_utils import ModelEstimation, TD_Q_evaluation, TD_Q_evaluation_given_pol, LSTDQ, LSTD, LSTD_Nested, batch_TD_value_evaluation, LSTD_Nested_Standard
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
result_dir_to_load = './saved/2020_01_28_23_14_23'

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
                    default=5000) # 1000
# ----- Model Parameters ---------------------------------------------#

parser.add_argument('--depth', type=int,  help='Length of trajectory',
                    default=5)

parser.add_argument('--gammaEval', type=float,  help='gammaEval',
                    default=0.99)
parser.add_argument('--default_gamma', type=float,  help='The default guidance discount factor (if None use gammaEval)',
                    default=None)
parser.add_argument('--default_l2_proj', type=float,
                    default=1e-4)
parser.add_argument('--sampling_type', type=str,  help='How to generate data',
                    default='Trajectories')  # 'Trajectories' |   'Generative'

parser.add_argument('--evaluation_loss_type', type=str,  help='how to evaluate loss',
                    default='rankings_kendalltau')  #  'rankings_kendalltau' | 'L2_normalized | correction_scaling' |  'L1_normalized' | 'L2_normalized' | 'Lmax_normalized' | 'greedy_pol_V' | ''greedy_V_L1' | 'Bellman_Err' | 'greedy_V_L_infty' | Values_SameGamma


# # Parameters for iter-TD alg
# parser.add_argument('--TD_Init_type', type=str,  help='How to initialize V',
#                     default='R_max')  # 'zero' |   'random_0_1' |  'random_0_max' | 'R_max'
# # Parameters for iter-TD alg
# parser.add_argument('--n_TD_iter', type=float,  help='number of TD iterations',
#                     default=5000)  #

args = parser.parse_args()


# Additional args:

args.alg_type = 'LSTD_Nested_Standard'  # 'LSTD' | 'LSTD_Nested' | batch_TD_value_evaluation | LSTD_Nested_Standard

# Parameters for iter-TD alg
args.learning_rate_def = {'type': 'a/(b+i_iter)', 'a': 50, 'b': 1000}      #  #  {'type': 'a/(b+i_iter)', 'a': 100, 'b': 1000}  |  {'type': 'const', 'alpha': 0.1}   | {'type': 'a/(b+i_sqrt(iter))', 'a': 100, 'b': 1000}

# MDP definition ( see data_utils.SetMdpArgs)
# args.mrp_def = {'type': 'ToyMRP', 'p01': 0.5, 'p10': 0.5,  'reward_std': 0.1}
args.mrp_def = {'type': 'Chain', 'p_left': 0.5, 'length': 9,  'mean_reward_range': (0, 1), 'reward_std': 0.1}


#  how to create parameter grid:
args.param_grid_def = {'type': 'gamma_guidance', 'spacing': 'linspace', 'start': 0.1, 'stop': 0.99, 'num': 20}
# args.param_grid_def = {'type': 'l2_fp', 'spacing': 'linspace', 'start': 0.0001, 'stop': 0.1, 'num': 50}
# args.param_grid_def = {'type': 'l2_proj', 'spacing': 'linspace', 'start': 0.0001, 'stop': 0.001, 'num': 20}
# args.param_grid_def = {'type': 'l2_factor', 'spacing': 'linspace', 'start': 0.0001, 'stop': 0.01, 'num': 20}

args.initial_state_distrb_type = 'uniform'  # 'uniform' | 'middle'
args.default_n_trajectories = 5

# args.config_grid_def = {'type': 'n_trajectories', 'spacing': 'list', 'list': [1, 2, 3]}
# args.config_grid_def = {'type': 'trajectory_len', 'spacing': 'list', 'list': [10, 20, 30]}
args.config_grid_def = {'type': 'p_left', 'spacing': 'linspace', 'start': 0.1, 'stop': 0.5, 'num': 20}

# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
def run_simulations(args, save_result, local_mode):
    import ray
    ray.init(local_mode=local_mode)
    # A Ray remote function.
    # runs a single repetition of the experiment
    @ray.remote  # (num_cpus=0.2)  # specify how much resources the process needs
    def run_rep(i_rep, alg_param_grid, config_grid, args):
        nS = args.nS
        if args.initial_state_distrb_type == 'middle':
            args.initial_state_distrb = np.zeros(nS)
            args.initial_state_distrb[nS // 2] = 1.
        elif args.initial_state_distrb_type == 'uniform':
            args.initial_state_distrb = np.ones(nS) / nS

        initial_state_distrb = args.initial_state_distrb
        n_grid = alg_param_grid.shape[0]
        n_configs = args.n_configs
        loss_rep = np.zeros((n_configs, n_grid))

        # default values
        gammaEval = args.gammaEval
        if args.default_gamma is None:
            gamma_guidance = gammaEval
        else:
            gamma_guidance = args.default_gamma
        l2_fp = 1e-5
        l2_proj = args.default_l2_proj

        for i_config in range(args.n_configs):  # grid of n_configs

            n_traj = args.default_n_trajectories
            if args.config_grid_def['type'] == 'n_trajectories':
                n_traj = config_grid[i_config]
            elif args.config_grid_def['type'] == 'trajectory_len':
                args.depth = config_grid[i_config]
            elif args.config_grid_def['type'] == 'p_left':
                args.mrp_def['p_left'] = config_grid[i_config]

            # Generate MDP:
            M = MRP(args)

            for i_grid, alg_param in enumerate(alg_param_grid):

                # grid values:
                if args.param_grid_def['type'] == 'l2_proj':
                    l2_proj = alg_param
                elif args.param_grid_def['type'] == 'l2_fp':
                    l2_fp = alg_param
                elif args.param_grid_def['type'] == 'gamma_guidance':
                    gamma_guidance = alg_param
                elif args.param_grid_def['type'] == 'l2_factor':
                    l2_fp = alg_param
                    l2_proj = alg_param
                else:
                    raise ValueError('Unrecognized args.grid_type')

                if args.alg_type not in ['LSTD_Nested', 'LSTD_Nested_Standard']\
                        and args.param_grid_def['type'] == 'l2_fp':
                    raise Warning(args.alg_type + ' does not use l2_fp !!!')

                V_true = np.linalg.solve((np.eye(nS) - gammaEval * M.P), M.R)

                # Generate data:
                data = M.SampleData(n_traj, args.depth, p0=initial_state_distrb, reward_std=args.reward_std,
                                    sampling_type=args.sampling_type)

                # value estimation:
                if args.alg_type == 'LSTD':
                    V_est = LSTD(data, gamma_guidance, args, l2_factor=l2_proj)
                elif args.alg_type == 'LSTD_Nested':
                    V_est = LSTD_Nested(data, gamma_guidance, args, l2_proj, l2_fp)

                elif args.alg_type == 'LSTD_Nested_Standard':
                    V_est = LSTD_Nested_Standard(data, gamma_guidance, args, l2_proj, l2_fp)
                elif args.alg_type == 'batch_TD_value_evaluation':
                    V_est = batch_TD_value_evaluation(data, gamma_guidance, args, l2_factor=l2_proj)
                else:
                    raise ValueError('Unrecognized args.grid_type')
                loss_type = args.evaluation_loss_type
                pi = None
                eval_loss = evaluate_value_estimation(loss_type, V_true, V_est, M, pi, gammaEval, gamma_guidance)
                loss_rep[i_config, i_grid] = eval_loss
            # end for i_grid
        #  end for i_config

        # Get optimal gamma per config\p_l:
        gamma_best_rep = np.zeros(n_configs)
        for i_config in range(args.n_configs):  # grid of n_configs
            i_best = np.argmin(loss_rep[i_config])
            gamma_best_rep[i_config] = alg_param_grid[i_best]
        return loss_rep, gamma_best_rep
    # end run_rep

    start_time = timeit.default_timer()
    if save_result:
        create_result_dir(args)
    set_random_seed(args.seed)

    n_reps = args.n_reps
    alg_param_grid = get_grid(args.param_grid_def)
    n_grid = alg_param_grid.shape[0]

    config_grid = get_grid(args.config_grid_def)
    n_configs = len(config_grid)
    args.n_configs = n_configs

    planing_loss = np.zeros((n_reps, n_configs, n_grid))
    gamma_best = np.zeros((n_reps, n_configs))

    # ----- Run simulation in parrnell process---------------------------------------------#
    out_id_id_lst = []
    for i_rep in range(n_reps):
        # returns objects ids:
        out_id = run_rep.remote(i_rep, alg_param_grid, config_grid, args)
        out_id_id_lst.append(out_id)
    # -----  get the results --------------------------------------------#
    for i_rep in range(n_reps):
        loss_rep, gamma_best_rep = ray.get(out_id_id_lst[i_rep])
        write_to_log('Finished: {} out of {} reps'.format(i_rep + 1, n_reps), args)
        planing_loss[i_rep] = loss_rep
        gamma_best[i_rep] = gamma_best_rep
    # end for i_rep
    info_dict = {'planing_loss_avg': planing_loss.mean(axis=0), 'planing_loss_std': planing_loss.std(axis=0),
                 'alg_param_grid': alg_param_grid, 'config_grid': config_grid, 'gamma_best':gamma_best}
    if save_result:
        save_run_data(args, info_dict)
    stop_time = timeit.default_timer()
    write_to_log('Total runtime: ' +
                 time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)), args)
    return info_dict
# end  run_simulations
# -------------------------------------------------------------------------------------------
def run_main(args, save_result=True, load_run_data_flag=False, result_dir_to_load='', save_PDF=False, plot=True):
    SetMrpArgs(args)
    if load_run_data_flag:
        args, info_dict = load_run_data(result_dir_to_load)
    else:
        info_dict = run_simulations(args, save_result, local_mode)
    planing_loss_avg = info_dict['planing_loss_avg']
    planing_loss_std = info_dict['planing_loss_std']
    alg_param_grid = info_dict['alg_param_grid']
    config_grid =  info_dict['config_grid']
    gamma_best = info_dict['gamma_best']
    n_reps = args.n_reps

    # ----- Plot figures  ---------------------------------------------#
    if plot:
        ax = plt.figure().gca()
        ci_factor = 1.96 / np.sqrt(n_reps)  # 95% confidence interval factor
        gamma_best_avg = gamma_best.mean(axis=0)
        gamma_best_std =  gamma_best.std(axis=0)
        mixing_times = np.zeros(args.n_configs)
        for i_config in range(args.n_configs):  # grid of number of trajectories to generate
            args.mrp_def['p_left'] = config_grid[i_config]
            M = MRP(args)
            mixing_times[i_config] = np.around(calc_mixing_time(M.P), decimals=3)

        plt.figure()
        plt.errorbar(config_grid, gamma_best_avg, yerr=gamma_best_std * ci_factor)
        plt.grid(True)
        plt.xlabel(r': $p_l=$')
        plt.ylabel('Optimal Discount')

        plt.figure()
        plt.errorbar(mixing_times, gamma_best_avg, yerr=gamma_best_std * ci_factor)
        plt.grid(True)
        plt.xlabel('Mixing Time')
        plt.ylabel('Optimal Discount')

        if save_PDF:
            plt.savefig(args.run_name + '.pdf', format='pdf', bbox_inches='tight')
        else:
            plt.title('Loss +- 95% CI, Legend: {} \n '.format(args.config_grid_def['type']) + str(args.run_name))
            # '\n' r'Average absolute estimation error of $V^*(s)$ +- 95% CI'
        plt.show()
    # end if plot
    print('done')
    return info_dict
# end  run_main
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    info_dict = run_main(args, save_result=True, load_run_data_flag=load_run_data_flag,
                         result_dir_to_load=result_dir_to_load, save_PDF=save_PDF)



