import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from hmmlearn import hmm
from scipy import stats
from sklearn.metrics.cluster import v_measure_score

from oMHMM.tests.simulations.scenario_2 import spamhmm_orthogonal_3clust_init_C


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
    data1 = np.zeros((n_data, seq_len))
    data2 = np.zeros((n_data, seq_len))
    data3 = np.zeros((n_data, seq_len))
    pi = np.zeros((n_data, n_clusters))
    states1 = np.zeros((n_data, seq_len))
    states2 = np.zeros((n_data, seq_len))
    states3 = np.zeros((n_data, seq_len))
    for i in range(n_data):
        # pick a cluster id and create data from this cluster
        k = np.random.choice(n_clusters, size=1, p=weights)[0]
        for l in range(seq_len):
            if k == 0:
                c, x = generate_hmm(rates[k], n_states, trans[k], start_prob[k], 1)
                data1[i, l] = x
                states1[i, l] = c
            elif k ==1:
                c, x = generate_hmm(rates[k], n_states, trans[k], start_prob[k], 1)
                data2[i, l] = x
                states2[i, l] = c
            else:
                c, x = generate_hmm(rates[k], n_states, trans[k], start_prob[k], 1)
                data3[i, l] = x
                states3[i, l] = c

        if k == 0:
            len_0 += 1
        pi[i, k] = 1

    return data1, data2, data3, len_0, pi


v_measures = []
v_measures_oMHMM = []
seed = 32
np.random.seed(seed)

num_of_cells, seq_len, num_of_states = 100, 200, 4
trans_1 = [[0, 0, .07, .93], [0, .003, .007, .99], [0, 0, .06, .94], [0, 0, .02, .98]]
trans_2, start_1, start_2 = [[0, .002, .99, .008], [0, 0, .95, .05], [0, 0, .92, .08], [0, 0, .87, .13]], [0, 0, .1,
                                                                                                           .9], [0,
                                                                                                                 0,
                                                                                                                 .9,
                                                                                                                .1]
start_3 = [0, .5, 0, .5]
trans_3 = [[0, .3, 0, .7], [0, .4, 0, .6], [0, .3, 0, .7], [0, .3, .02, .68]]
trans = np.concatenate((trans_1, trans_2, trans_3), axis=0).reshape(3, num_of_states, num_of_states)
start_prob = np.concatenate((start_1, start_2, start_3), axis=0).reshape(3, num_of_states)

rates_1 = np.array([1, 3, 9, 27])
rates_2 = np.array([1, 3, 9, 27])
rates_3 = np.array([1, 3, 9, 27])
rates = np.concatenate((rates_1, rates_2, rates_3), axis=0).reshape(3, num_of_states)

data1, data2, data3, len_0, pi = generate_data(num_of_cells, [.3, .3, .4], rates, seq_len, trans, start_prob,
                                               num_of_states)

data = np.concatenate((data1, data2, data3), axis=0)

y = np.zeros(num_of_cells, dtype=int)
M = 3
lengths = (np.ones(num_of_cells, dtype=int) * seq_len).tolist()

hyperparameters = [0, .1, .5, 1]
for hyperparam in hyperparameters:
    mhmm = spamhmm_orthogonal_3clust_init_C.SpaMHMM(hyperparam=hyperparam, epsilon=.0000001, n_nodes=1,
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



