from __future__ import division, absolute_import, print_function
import numpy as np


# Markov Decision Process
# ------------------------------------------------------------------------------------------------------------~
def SetMdpArgs(args):
    mdp_type = args.mdp_def['type']
    if mdp_type == 'RandomMDP':
        args.nS = args.mdp_def['S']  # number of states
        args.nA = args.mdp_def['A']  # number of actions
        args.k = args.mdp_def['k']  # Number of non-zero entries in each row  of transition-matrix
        args.reward_std = args.mdp_def['reward_std']
    elif mdp_type == 'GridWorld':
        args.nS = args.mdp_def['N0'] * args.mdp_def['N1']
        args.nA = 5
        args.reward_std = args.mdp_def['reward_std']
    elif mdp_type == 'GridWorld2':
        args.nS = args.mdp_def['N0'] * args.mdp_def['N1']
        args.nA = 4
        args.reward_std = args.mdp_def['reward_std']
    else:
        raise ValueError('Invalid mdp_type')


#------------------------------------------------------------------------------------------------------------~
class MDP(): # Markov Desicion Process
    def __init__(self, args):
        """
          Randomly generate an MDP


          Parameters:

          Returns:
          P: [nS x nA x nS] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
          R: [nS x nA] mean rewards matrix R
          """
        nS = args.nS  # number of states
        nA = args.nA  # number of actions
        P = np.zeros((nS, nA, nS))
        mdp_type = args.mdp_def['type']

        if mdp_type == 'RandomMDP':
            # For each state-action pair (s; a), the distribution over the next state,  P_{s,a,s'}=P(s'|s,a), is determined by choosing k  non-zero entries uniformly from
            #  all nS states, filling these k entries with values uniformly drawn from [0; 1], and finally normalizing
            k = args.k  # Number of non-zero entries in each row  of transition-matrix
            for a in range(nA):
                for i in range(nS):
                    nonzero_idx = np.random.choice(nS, k, replace=False)
                    for j in nonzero_idx:
                        P[i, a, j] = np.random.rand(1)
                    P[i, a, :] /= P[i, a, :].sum()
            R = np.random.rand(nS, nA) # rewards means

        elif mdp_type == 'GridWorld':
            # based on "How to Combine Tree-Search Methods in Reinforcement Learning", Efroni '19
            # state transition is deterministic
            N0 = args.mdp_def['N0']
            N1 = args.mdp_def['N1']
            action_set = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]]
            assert args.nA == len(action_set)
            P = np.zeros((nS, nA, nS))
            next_state_table = np.empty((nS, nA), dtype=np.int)

            # Create reward means:
            R_image = np.random.rand(nS) - 0.5  # draw reward means ~ U[-0.5,0.5]
            # add goal state with r = 1
            s_goal = np.random.randint(nS)
            R_image[s_goal] = 1.0
            R = np.zeros((nS,nA))
            for s in range(nS):
                # set same reward mean for all actions in each state
                R[s, :] = R_image[s]
            # set state transitions
            for s0 in range(N0):
                for s1 in range(N1):
                    s = s0 + s1 * N0
                    for a, shift in enumerate(action_set):
                        s_next0 = s0 + shift[0]
                        s_next1 = s1 + shift[1]
                        if 0 <= s_next0 < N0 and 0 <= s_next1 < N1:
                            # if the move is legal
                            s_next = s_next0 + s_next1 * N0
                        else:
                            # otherwise stay in place
                            s_next = s
                        next_state_table[s, a] = s_next
                        P[s, a, s_next] = 1
            self.next_state_table = next_state_table
        elif mdp_type == 'GridWorld2':
            # based on "How to Combine Tree-Search Methods in Reinforcement Learning", Efroni '19
            # state transition is deterministic
            N0 = args.mdp_def['N0']
            N1 = args.mdp_def['N1']
            action_set = [[-1, 0], [1, 0], [0, 1], [0, -1]]
            assert args.nA == len(action_set)
            P = np.zeros((nS, nA, nS))
            next_state_table = np.empty((nS, nA), dtype=np.int)

            # Create reward means:
            R_image = np.random.rand(nS) - 1.0  # draw reward means ~ U[-1,0]
            # add goal state with r = +1 , and no exit
            s_goal = np.random.randint(nS)
            R_image[s_goal] = 1.0
            R = np.zeros((nS, nA))
            for s in range(nS):
                # set same reward mean for all actions in each state
                R[s, :] = R_image[s]
            # set state transitions
            for s0 in range(N0):
                for s1 in range(N1):
                    s = s0 + s1 * N0
                    for a, shift in enumerate(action_set):
                        s_next0 = s0 + shift[0]
                        s_next1 = s1 + shift[1]
                        if s != s_goal and 0 <= s_next0 < N0 and 0 <= s_next1 < N1:
                            # if the move is legal
                            s_next = s_next0 + s_next1 * N0
                        else:
                            # otherwise stay in place
                            s_next = s
                        next_state_table[s, a] = s_next
                        P[s, a, s_next] = 1
            self.next_state_table = next_state_table
        else:
            raise ValueError('Invalid mdp_type')
        self.R = R
        self.P = P
        self.nA = nA
        self.nS = nS
        self.type = args.mdp_def['type']

    # ------------------------------------------------------------------------------------------------------------~
    def SampleData(self, pi, n, depth, p0=None, reward_std=0.1, sampling_type='Trajectories'):
        """
        # generate n trajectories

        Parameters:
        P: [nS x nA x nS] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
        R: [nS x nA] mean rewards matrix R
        pi: [nS x nA]  matrix representing  pi(a|s)
        n: number of trajectories to generate
        depth: Length of trajectory
        p0 (optional) [nS] matrix of initial state distribution (default:  uniform)
        Returns:
        data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
        """
        R = self.R
        P = self.P
        nS = self.nS
        nA = self.nA
        if p0 is None:
            p0 = np.ones(nS) / nS  # uniform
        data = []
        if sampling_type == 'Trajectories':
            for i_traj in range(n):
                data.append([])
                # sample initial state:
                s = sample_discrete(p0)
                a = sample_discrete(pi[s, :])
                # Until t==depth, sample a~pi(.|s), s'~P(.|s,a), r~R(s,a)
                for t in range(depth):
                    if self.type == 'GridWorld':
                        # deterministic transition
                        s_next = self.next_state_table[s, a]
                    else:
                        s_next = sample_discrete(P[s, a, :])
                    a_next = sample_discrete(pi[s, :])
                    r = R[s, a] + np.random.randn(1)[0] * reward_std
                    data[i_traj].append((s, a, r, s_next, a_next))
                    s = s_next
                    a = a_next
        elif sampling_type == 'Generative':
            for i_traj in range(n):
                data.append([])
                for t in range(depth):
                    s = sample_discrete(np.ones(nS) / nS)
                    a = sample_discrete(np.ones(nA) / nA)
                    r = R[s, a] + np.random.randn(1)[0] * reward_std
                    s_next = sample_discrete(P[s, a, :])
                    a_next = sample_discrete(pi[s, :])
                    data[i_traj].append((s, a, r, s_next, a_next))
        else:
            raise ValueError('Unrecognized data_type')
        return data
#------------------------------------------------------------------------------------------------------------~
def sample_discrete(probs):
    """
    Samples a discrete distribution
     Parameters:
        probs - probabilities over {0,...K-1}
    Returns:
        drawn random integer from  {0,...K-1} according to probs
    """
    K = probs.size
    return np.random.choice(K, size=1, p=probs)[0]

