import numpy as np


class DynLinEnvironment:

    def __init__(self, a, b, c, d, n, p, m, horizon, noise, out_noise,
                 n_trials, output_mapping):
        assert horizon > 0 and n > 0 and p > 0 and m > 0
        assert a.shape == (n, n) and b.shape == (n, p) \
               and c.shape == (m, n) and d.shape == (m, p)
        if noise is not None:
            assert noise.shape == (n_trials, horizon, n)
        if out_noise is not None:
            assert out_noise.shape == (n_trials, horizon, m)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.n = n
        self.p = p
        self.m = m
        self.all_noise = noise
        self.all_out_noise = out_noise
        self.horizon = horizon
        self.n_trials = n_trials
        self.output_mapping = output_mapping
        self.t = None
        self.state = None
        self.noise = None
        self.out_noise = None
        self.reset(0)

    def step(self, action):
        assert action.ndim == 2 and action.shape == (
            self.p, 1), 'error in action input'
        output = self.c @ self.state + self.d @ action + \
            self.out_noise[self.t, :].reshape(self.m, 1)
        self.state = self.a @ self.state + self.b @ action + \
            self.noise[self.t, :].reshape(self.n, 1)
        self.t = self.t + 1
        return self.output_mapping @ output

    def reset(self, i_trials):
        assert 0 <= i_trials < self.n_trials, 'trial not available'
        self.state = np.zeros((self.n, 1))
        self.t = 0
        self.noise = self.all_noise[i_trials, :, :]
        assert self.noise.ndim == 2 and self.noise.shape == (
            self.horizon, self.n), 'error in noise'
        self.out_noise = self.all_out_noise[i_trials, :, :]
        assert self.out_noise.ndim == 2 and self.out_noise.shape == (
            self.horizon, self.m), 'error in output_noise'
# creating a new state-independent class of environments 
class StateIndepDynLinEnvironment:

    def __init__(self, a, b, c, d, g, n, p, m, horizon, noise, out_noise, n_arms,
                 n_trials, output_mapping):
        assert horizon > 0 and n > 0 and p > 0 and m > 0
        assert a.shape == (n, n) and b.shape == (n,p) and c.shape == (m, n) and d.shape == (m, p)
        if noise is not None:
            assert noise.shape == (n_trials, horizon, n)
        if out_noise is not None:
            assert out_noise.shape == (n_trials, horizon, m)
        self.a = a
        self.b = b 
        self.c = c
        self.d = d
        self.g = g # d X n (or p x n) matrix
        self.n = n
        self.p = p
        self.m = m
        self.all_noise = noise
        self.all_out_noise = out_noise
        self.horizon = horizon
        self.n_arms = n_arms
        self.n_trials = n_trials
        self.output_mapping = output_mapping
        self.t = None
        self.state = None
        self.noise = None
        self.out_noise = None
        self.reset(0)

    def step(self, action, actions_played): #add actions_played insted of actions.
        assert action.ndim == 2 and action.shape == (
            self.p, 1), 'error in action input'
        assert actions_played.shape == (self.horizon, self.p)

        h=np.zeros((self.horizon,self.p)) # Markov parameter matrix(something like this)
        a_s=np.zeros((self.horizon,self.n)) # Diagonal A-matrix values raised to a s-1 power.

        H_index = len(actions_played) 

        a_indices=np.arange(H_index)

        a_s[1:H_index, :]=np.power(np.diagonal(self.a),a_indices[1:,None]-1)

        h = (self.g @ a_s.T).T

        h[0, :]=np.diagonal(self.d) # remember that h{0}=theta

        actions_played = actions_played[::-1]

        output = np.sum(h * actions_played) + self.out_noise[self.t, :].reshape(self.m, 1) 

        self.state = self.a @ self.state + self.b @ action + \
            self.noise[self.t, :].reshape(self.n, 1)
        self.t = self.t + 1
        return self.output_mapping @ output

    def reset(self, i_trials):
        assert 0 <= i_trials < self.n_trials, 'trial not available'
        self.state = np.zeros((self.n, 1))
        self.t = 0
        self.noise = self.all_noise[i_trials, :, :]
        assert self.noise.ndim == 2 and self.noise.shape == (
            self.horizon, self.n), 'error in noise'
        self.out_noise = self.all_out_noise[i_trials, :, :]
        assert self.out_noise.ndim == 2 and self.out_noise.shape == (
            self.horizon, self.m), 'error in output_noise'
        