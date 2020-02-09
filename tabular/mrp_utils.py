from __future__ import division, absolute_import, print_function
import numpy as np
from common.utils import sample_discrete

# Markov Reward Process

# ------------------------------------------------------------------------------------------------------------~
def SetMrpArgs(args):
    mrp_type = args.mrp_def['type']
    if mrp_type == 'ToyMRP':
        args.nS = 2
        args.reward_std = args.mrp_def['reward_std']
    elif mrp_type == 'Chain':
        args.nS = args.mrp_def['length']
        args.reward_std = args.mrp_def['reward_std']
    else:
        raise ValueError('Invalid mrp_type')



# ------------------------------------------------------------------------------------------------------------~
class MRP(): # Markov Reward Process
    def __init__(self, args):
        """
          Parameters:

          Returns:

          """
        nS = args.nS  # number of states
        P = np.zeros((nS, nS))
        mrp_type = args.mrp_def['type']

        if mrp_type == 'ToyMRP':
            p01 = args.mrp_def['p01']
            p00 = 1 - p01
            p10 = args.mrp_def['p10']
            p11 = 1 - p10

            P[0, 0] = p00
            P[0, 1] = p01
            P[1, 0] = p10
            P[1, 1] = p11

            R = np.random.rand(nS)

        elif mrp_type == 'Chain':
            p_left = args.mrp_def['p_left']
            p_right = 1 - p_left
            nS = args.mrp_def['length']
            P = np.zeros((nS, nS))
            for i in range(nS):
                if i < nS - 1:
                    P[i, i+1] = p_right
                else:
                    P[i,i] = p_right
                if i > 0:
                    P[i, i - 1] = p_left
                else:
                    P[i, i] = p_left

            mean_reward_range =  args.mrp_def['mean_reward_range']
            R = mean_reward_range[0] + (mean_reward_range[1] - mean_reward_range[0]) * np.random.rand(nS)
            # R = np.random.rand(nS)
        else:
            raise ValueError('Invalid mrp_type')
        self.R = R
        self.P = P
        self.nS = nS
        self.type = args.mrp_def['type']

    def SampleData(self, n, depth, p0=None, reward_std=0.1, sampling_type='Trajectories'):
        """
        # generate n trajectories

        Parameters:
        """
        R = self.R
        P = self.P
        nS = self.nS
        if p0 is None:
            p0 = np.ones(nS) / nS  # uniform
        data = []
        if sampling_type == 'Trajectories':
            for i_traj in range(n):
                data.append([])
                # sample initial state:
                s = sample_discrete(p0)

                for t in range(depth):
                    # TODO: in gridWorld use table
                    # Until t==depth, sample a~pi(.|s), s'~P(.|s,a), r~R(s,a)
                    s_next = sample_discrete(P[s, :])
                    r = R[s] + np.random.randn(1)[0] * reward_std
                    data[i_traj].append((s, r, s_next))
                    s = s_next

        elif sampling_type == 'Generative':
            for i_traj in range(n):
                data.append([])
                for t in range(depth):
                    s = sample_discrete(np.ones(nS) / nS)
                    r = R[s] + np.random.randn(1)[0] * reward_std
                    s_next = sample_discrete(P[s, :])
                    data[i_traj].append((s, r, s_next))
        else:
            raise ValueError('Unrecognized data_type')
        return data


# ------------------------------------------------------------------------------------------------------------~


def calc_mixing_time(P):
    evals, evecs = np.linalg.eig(P)
    evals.sort()
    mixing_time = evals[-1]-evals[-2] # spectral gap , note: evals[-1] always == 1
    return mixing_time