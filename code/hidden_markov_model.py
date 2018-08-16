#!/usr/bin/env python3

import numpy as np

class HMM:
    def __init__(self, n_states, vocab_size):
        """
        initialize HMM with transition probabilities
        and emission probabilities uniformly

            n_states:   number of states
            vocab_size: number of possible observation types
        symbols here follow Jurafsky and Martin NLP book
        """
        # a set of N states
        self.Q = [np.ones(vocab_size)/vocab_size for i in range(n_states)]

        # transition probability matrix
        self.A = np.ones((n_states, n_states))/n_states
        # special start and end states
        self.q0 = np.ones(n_states)/n_states
        self.qF = np.ones(n_states)/n_states

        self.vocab_size = vocab_size
        self.n_states = n_states

    def __len__(self):
        return self.n_states

    def forward_backward(self, observations):
        """
        This funtion runs forward-backward at the same time
        """
        T = len(observations)
        N = self.n_states
        f_matrix = np.zeros((N, T))
        b_matrix = np.zeros((N, T))

        for s in range(N):
            f_matrix[s, 0] = self.q0[s] * self.Q[s][observations[0]]
            b_matrix[s, T-1] = self.qF[s]

        for t in range(1, T):
            for s in range(N):
                for s_prime in range(N):
                    prev = f_matrix[s_prime, t-1]
                    transition = self.A[s_prime, s]
                    emission = self.Q[s][observations[t]]
                    f_matrix[s, t] += prev * transition * emission

                    prev = b_matrix[s_prime, T-t]
                    transition = self.A[s, s_prime]
                    emission = self.Q[s_prime][observations[T-t]]
                    b_matrix[s, (T-1)-t] += prev * transition * emission

        return f_matrix, b_matrix

    def forward(self, observations):
        T = len(observations)
        N = self.n_states
        f_matrix = np.zeros((N, T))

        # initialization
        for s in range(N):
            f_matrix[s, 0] = self.q0[s] * self.Q[s][observations[0]]

        # recursion
        for t in range(1, T):
            for s in range(N):
                for s_prime in range(N):
                    prev = f_matrix[s_prime, t-1]
                    transition = self.A[s_prime, s]
                    emission = self.Q[s][observations[t]]
                    f_matrix[s, t] += prev * transition * emission
        return f_matrix

    def backward(self, observations):
        T = len(observations)
        N = self.n_states
        b_matrix = np.zeros((N, T))

        # initialization
        for s in range(N):
            b_matrix[s, T-1] = self.qF[s]

        # recursion
        for t in range(T-2, -1, -1):
            for s in range(N):
                for s_prime in range(N):
                    prev = b_matrix[s_prime, t+1]
                    transition = self.A[s, s_prime]
                    emission = self.Q[s_prime][observations[t+1]]

                    b_matrix[s, t] += prev * transition * emission
        return b_matrix

    def viterbi(self, observations):
        T = len(observations)
        N = self.n_states
        viterbi = np.zeros((N, T))

        # initialization
        for s in range(N):
            viterbi[s, 0] = self.q0[s] * self.Q[s][observations[0]]

        # recursion
        for t in range(1, T):
            for s in range(N):
                to_s = []
                for s_prime in range(N):
                    prev = viterbi[s_prime, t-1]
                    transition = self.A[s_prime, s]
                    emission = self.Q[s][observations[t]]

                    to_s.append(prev * transition * emission)
                viterbi[s, t] = max(to_s)
        return viterbi

    def decode(self, observations):
        viterbi = self.viterbi(observations)
        # backtrace
        return viterbi.argmax(axis=0)

    def compute_likelihood(self, observations, matrix=None):
        if not isinstance(matrix, np.ndarray):
            matrix = self.forward(observations)

        likelihood = 0
        for s in range(self.n_states):
            prev = matrix[s, len(observations)-1]
            transition = self.qF[s]
            likelihood += prev * transition
        return likelihood

    def train(self, train_data, stop=0.001):
        prev_l = 0
        while True:
            cur_l = 0
            for o_seq in train_data:
                N = self.n_states
                T = len(o_seq)
                V = self.vocab_size

                gamma = np.zeros((N, T, V))
                xi = np.zeros((T-1, N, N))
                # This is specially for xi for "from start" and "to end"
                xi_sp = np.zeros((2, N))

                f_matrix, b_matrix = self.forward_backward(o_seq)
                L = self.compute_likelihood(o_seq, matrix=f_matrix)

                # E-step: compute gamma and xi
                for t in range(T):
                    for j in range(N):
                        # compute gamma for time t, state j
                        alpha = f_matrix[j, t]
                        beta = b_matrix[j, t]
                        o = o_seq[t]
                        gamma[j, t, o] = alpha * beta / L

                        # Skip xi when t is the last observation
                        if t == T-1:
                            continue

                        for i in range(N):
                            # compute xi for time t, state i and j
                            alpha = f_matrix[i, t]
                            transition = self.A[i, j]
                            emission = self.Q[j][o_seq[t+1]]
                            beta = b_matrix[j, t+1]
                            xi[t, i, j] = (
                                    alpha * transition * emission * beta / L)
                # E-step for special states
                # Could be integrated into the above block
                # by adding "<start>" "<end>" symbols to observations
                for j in range(N):
                    from_start = self.q0[j]
                    emission = self.Q[j][o_seq[0]]
                    beta = b_matrix[j, 0]
                    xi_sp[0][j] = from_start * emission * beta / L

                    alpha = f_matrix[j, T-1]
                    to_end = self.qF[j]
                    xi_sp[1][j] = to_end * alpha / L

                # M-step: compute new parameters
                # update emission probabilities
                for j in range(N):
                    # exp number of observation emitted from state j
                    exp_total = np.sum(gamma[j])
                    for v in range(V):
                        # exp number of v emitted from state j
                        exp_v = np.sum(gamma[j, :, v])
                        self.Q[j][v] = exp_v / exp_total

                for j in range(N):
                    # exp number of transition from j to any
                    exp_total = np.sum(xi[:, j, :]) + xi_sp[1][j]
                    for i in range(N):
                        # exp number of transition from j to i
                        exp_ji = np.sum(xi[:, j, i])

                        self.A[j, i] = exp_ji / exp_total

                    # transition from start to j
                    self.q0[j] = xi_sp[0, j] / np.sum(xi_sp[0])

                    # transition from j to end
                    self.qF[j] = xi_sp[1, j] / exp_total

                cur_l += self.compute_likelihood(o_seq)
            cur_l /= len(train_data)
            if abs(cur_l - prev_l) < stop:
                print('stopping criterion met')
                break
            prev_l = cur_l
