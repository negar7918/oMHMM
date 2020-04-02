import numpy as np
from hmmlearn import hmm
from scipy import stats
from sklearn.metrics.cluster import v_measure_score

from oMHMM.tests.simulations.scenario_1 import spamhmm_orthogonal_with_hyperparam_init_C


class PoissonHMM(hmm._BaseHMM):

    # Overriding the parent
    def __init__(self, rates, *args, **kwargs):
        hmm._BaseHMM.__init__(self, *args, **kwargs)
        self.rates = rates

    # Overriding the parent
    def _generate_sample_from_state(self, state, random_state=None):
        return stats.poisson(self.rates[state]).rvs()


def generate_hmm(rate, num_of_states, trans, start_prob, num_of_samples):
    model = PoissonHMM(rate, n_components=num_of_states)
    model.startprob_ = np.array(start_prob)
    model.transmat_ = np.array(trans)
    Y, C = model.sample(num_of_samples)
    print("Y values:")
    print(Y)
    print("C values:")
    print(C)
    return C, Y


def generate_data(n_data, weights, rates, seq_len, trans, start_prob, n_states):
    len_0 = 0
    n_clusters = trans.shape[0]
    data = np.zeros((n_data, seq_len))
    pi = np.zeros((n_data, n_clusters))
    states = np.zeros((n_data, seq_len))
    for i in range(n_data):
        # pick a cluster id and create data from this cluster
        k = np.random.choice(n_clusters, size=1, p=weights)[0]
        for l in range(seq_len):
            c, x = generate_hmm(rates[k], n_states, trans[k], start_prob[k], 1)
            data[i, l] = x
            states[i, l] = c
        if k == 0:
            len_0 += 1
        pi[i, k] = 1

    return data, len_0, pi

v_measures = []
v_measures_oMHMM = []
seed = 32
np.random.seed(seed)

num_of_cells, seq_len, num_of_states = 50, 800, 4
trans_1 = [[0, 0, .07, .93], [0, .003, .007, .99], [0, 0, .06, .94], [0, 0, .02, .98]]
trans_2, start_1, start_2 = [[0, .002, .99, .008], [0, 0, .95, .05], [0, 0, .92, .08], [0, 0, .87, .13]], [0, 0, .1, .9], [0, 0, .9, .1]
trans = np.concatenate((trans_1, trans_2), axis=0).reshape(2, num_of_states, num_of_states)
start_prob = np.concatenate((start_1, start_2), axis=0).reshape(2, num_of_states)

rates_1 = np.random.uniform(low=80, high=100, size=(num_of_states,))
rates_2 = np.random.uniform(low=80, high=100, size=(num_of_states,))
rates = np.concatenate((rates_1, rates_2), axis=0).reshape(2, num_of_states)

data, len_0, pi = generate_data(num_of_cells, [.5, .5], rates, seq_len, trans, start_prob, num_of_states)

y = np.zeros(num_of_cells, dtype=int)
M = 2
lengths = (np.ones(num_of_cells, dtype=int) * seq_len).tolist()

hyperparameters = [0, .1, .5, 1]
for hyperparam in hyperparameters:
    mhmm = spamhmm_orthogonal_with_hyperparam_init_C.SpaMHMM(hyperparam, epsilon=0, n_nodes=1,
                                                             mix_dim=M,
                                                             n_components=num_of_states,
                                                             n_features=seq_len,
                                                             graph=None,
                                                             n_iter=100,
                                                             verbose=True,
                                                             name='mhmm')
    mhmm.fit(data.flatten()[:, np.newaxis], y, lengths)
    pi_nk2, transitions = mhmm._compute_mixture_posteriors(data.flatten()[:, np.newaxis], y, lengths)
    predicted_cluster2 = []
    label2 = []
    for n in range(num_of_cells):
        cell = np.float64(pi_nk2[n])
        truth = np.float64(pi[n])
        predicted_cluster2 = np.append(predicted_cluster2, np.where(cell == max(cell))[0][0])
        label2 = np.append(label2, np.where(truth == max(truth))[0][0])
    print(label2)
    print(predicted_cluster2)
    v_measure = v_measure_score(label2, predicted_cluster2)
    v_measures_oMHMM = np.append(v_measures_oMHMM, v_measure)
    print(transitions.reshape(M, num_of_states, num_of_states))
    print(np.exp(pi_nk2))


print(v_measures_oMHMM)
print('oMHMM v_measure: {}'.format(np.max(v_measures_oMHMM)))
print('hyperparameter {}'.format(hyperparameters[np.where(v_measures_oMHMM == np.max(v_measures_oMHMM))[0][-1]]))



