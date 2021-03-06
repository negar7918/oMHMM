import numpy as np
from sklearn.base import BaseEstimator
from hmmlearn import hmm
from hmmlearn.base import ConvergenceMonitor
from hmmlearn.utils import normalize, log_normalize, iter_from_X_lengths
from scipy.misc import logsumexp
import pickle
import time
import warnings
import math
from oMHMM import maximizer, rand_initialization


def normalize(a, axis=None):
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


class RegulizedGaussianHMM(hmm.GaussianHMM):

    def __init__(self, similarity, other_trans, epsilon, hyperparam, *args, **kwargs):
        hmm.GaussianHMM.__init__(self, *args, **kwargs)
        self.similarity = similarity
        self.other_trans = other_trans
        self.hyperparam = hyperparam
        self.epsilon = epsilon

    def normalize(self, a, axis=None):
        a_sum = a.sum(axis)
        if axis and a.ndim > 1:
            # Make sure we don't divide by zero.
            a_sum[a_sum == 0] = 1
            shape = list(a.shape)
            shape[axis] = 1
            a_sum.shape = shape

        a /= a_sum


    def regularize_trans(self, a, hyperparam, epsilon):
        return maximizer.maximize_new(a, self.other_trans * hyperparam, epsilon)

    def _do_mstep(self, stats):
        if 's' in self.params:
            startprob_ = self.startprob_prior - 1. + stats['start']
            self.startprob_ = np.where(self.startprob_ == 0.0,
                                       self.startprob_, startprob_)
            self.normalize(self.startprob_)
        if 't' in self.params:
            transmat_ = self.transmat_prior - 1. + stats['trans']
            self.transmat_ = np.where(self.transmat_ == 0.0,
                                      self.transmat_, transmat_)

            self.normalize(self.transmat_, axis=1) #used to avoid computational issue in cvxpy
            self.transmat_ = self.regularize_trans(self.transmat_, self.hyperparam, self.epsilon)
            self.normalize(self.transmat_, axis=1)


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class SpaMHMM(BaseEstimator):
    def __init__(self, hyperparam, epsilon,  n_nodes, mix_dim, n_components, n_features, graph=None,
                 emission='gaussian', n_iter=10, tol=1e-2, n_iter_mstep=100,
                 lr_mstep=1e-3, rho1=0.9, rho2=0.99,
                 verbose=False, name='spamhmm'):
        super(SpaMHMM, self).__init__()

        self.n_nodes = n_nodes
        self.mix_dim = mix_dim
        self.n_components = n_components
        self.n_features = n_features
        self.graph = graph
        self.emission = emission
        self.n_iter = n_iter
        self.tol = tol
        self.n_iter_mstep = n_iter_mstep
        self.lr_mstep = lr_mstep
        self.rho1 = rho1
        self.rho2 = rho2
        self.verbose = verbose
        self.name = name
        self.hyperparam = hyperparam
        self.epsilon = epsilon

    def init_params(self, X):
        '''
        Parameters initialization.
        '''
        if type(X) == list:
            X = np.concatenate(X)

        self.mixCoefUnNorm = np.random.rand(self.n_nodes, self.mix_dim) + 1e-9
        self.mixCoef = np.reshape(np.array([.5, .5]), (1, 2))

        startProb = np.exp(np.random.randn(self.mix_dim, self.n_components))
        normalize(startProb, axis=1)

        transProb = np.exp(np.random.randn(self.mix_dim, self.n_components,
                                           self.n_components))
        normalize(transProb, axis=2)

        self.time_ = 1
        self.first_moment_ = np.zeros_like(self.mixCoef)
        self.second_moment_ = np.zeros_like(self.mixCoef)

        if self.emission == 'gaussian':
            self.mixModels = [RegulizedGaussianHMM(n_components=self.n_components,
                                              covariance_type='diag', similarity=1,
                                              other_trans=np.zeros((self.n_components, self.n_components)), hyperparam = self.hyperparam, epsilon=self.epsilon)
                              for i in range(self.mix_dim)]

            for m in range(self.mix_dim):
                self.mixModels[m]._init(X)
                seed = 1
                trans = rand_initialization.generate_transitions_random(seed, 2, self.n_components, .1)
                self.mixModels[0].transmat_ = trans[0]
                self.mixModels[1].transmat_ = trans[1]

        else:
            raise NotImplementedError('{} emission is not implemented'
                                      .format(self.emission))

    def scores_per_seq(self, X, y, lengths=None):
        '''
        Computes the log-likelihood for each sequence in X coming from nodes y.
        Inputs:
            X - np.array of size (n_samples, n_features).
            y - np.int of size n_sequences, whose entries are in the range
                [0, n_nodes-1].
            lengths - list containing the lengths of each individual sequence
                      in X, with size n_sequences.
        Outputs:
            log_likelihood - np.array of size n_sequences.
        '''
        if type(X) == list:
            lengths = [x.shape[0] for x in X]
            X = np.concatenate(X)
            y = np.array(y)

        N = y.shape[0]

        log_likelihood = np.zeros(N)
        for seq_idx, (i, j) in enumerate(iter_from_X_lengths(X, lengths)):
            ll_per_comp = np.zeros(self.mix_dim)
            for m in range(self.mix_dim):
                if self.mixCoef[y[seq_idx], m] == 0.:
                    continue

                ll_per_comp[m] = self.mixModels[m].score(X[i:j, :])

            nonzero_idx = (self.mixCoef[y[seq_idx], :] != 0.)

            log_likelihood[seq_idx] = logsumexp(np.log(self.mixCoef[y[seq_idx],
                                                                    nonzero_idx
                                                                    ])
                                                + ll_per_comp[nonzero_idx])

        return log_likelihood

    def score(self, X, y, lengths=None):
        '''
        Computes the mean log-likelihood for sequences in X coming from
        nodes y.
        Inputs:
            X - np.array of size (n_samples, n_features).
            y - np.int of size n_sequences, whose entries are in the
                range [0, n_nodes-1].
            lengths - list containing the lengths of each individual sequence
                      in X, with size n_sequences.
        Outputs:
            log_likelihood - scalar.
        '''
        if type(X) == list:
            lengths = [x.shape[0] for x in X]
            X = np.concatenate(X)
            y = np.array(y)

        self._check()

        Nsamples = X.shape[0]
        log_likelihood = np.sum(self.scores_per_seq(X, y, lengths))

        return log_likelihood /Nsamples


    def _check(self):
        '''
        Validates mixCoef parameter. The remaining parameters are validated
        by the hmm.check() routine.
        Raises
        ------
        ValueError
                If mixCoef have an invalid shape or do not sum to 1.
        '''
        if self.mixCoef.shape != (self.n_nodes, self.mix_dim):
            raise ValueError('mixCoef must have length n_components')
        if not np.allclose(self.mixCoef.sum(axis=1), 1.0):
            raise ValueError('mixCoef must sum to 1.0 (got {0:.4f})'
                             .format(self.mixCoef.sum(axis=1)))

    def _compute_mixture_posteriors(self, X, y, lengths):
        '''
        Computes the posterior log-probability of each mixture component given
        the observations X, y.
        Inputs:
            X - np.array of size (n_samples, n_features).
            y - np.int of size n_sequences, whose entries are in the
                range [0, n_nodes-1].
            lengths - list containing the lengths of each individual sequence
                      in X, with size n_sequences.
        Outputs:
            logmixpost - np.array of size (n_sequences, mix_dim).
        '''
        N = len(lengths)

        transitions = []
        #means = []
        logmixpost = np.zeros((N, self.mix_dim))
        for m in range(self.mix_dim):
            ll_m = np.zeros(N)
            for seq_idx, (i, j) in enumerate(iter_from_X_lengths(X, lengths)):
                ll_m[seq_idx] = self.mixModels[m].score(X[i:j, :])
            transitions = np.append(transitions, self.mixModels[m].transmat_)
            #means = np.append(means, self.mixModels[m].means_)

            logmixpost[:, m] = ll_m + np.log(self.mixCoef[y, m] + .000000001)

        log_normalize(logmixpost, axis=1)

        return logmixpost, transitions#, means

    def _compute_sufficient_statistics_in_mix_comp(self, X, y, lengths,
                                                   logmixpost, stats):
        '''
        Accumulates sufficient statistics for the parameters of each HMM in the
        mixture.
        Inputs:
            X - np.array of size (n_samples, n_features).
            y - np.int of size n_sequences, whose entries are in the
                range [0, n_nodes-1].
            lengths - list containing the lengths of each individual sequence
                      in X, with size n_sequences.
            logmixpost - np.array of size (n_sequences, mix_dim).
            stats - dictionary containing sufficient statistics (changed
                    inplace).
        '''
        for m in range(self.mix_dim):
            for seq_idx, (i, j) in enumerate(iter_from_X_lengths(X, lengths)):
                if self.mixCoef[y[seq_idx], m] == 0.:
                    continue

                framelogprob = self.mixModels[m]._compute_log_likelihood(X[i:j,
                                                                           :])
                _, fwdlattice = (self.mixModels[m]
                                 ._do_forward_pass(framelogprob))
                bwdlattice = self.mixModels[m]._do_backward_pass(framelogprob)
                posteriors = self.mixModels[m]._compute_posteriors(fwdlattice,
                                                                   bwdlattice)
                fwdlattice += logmixpost[seq_idx, m]
                bwdlattice += logmixpost[seq_idx, m]
                posteriors *= np.exp(logmixpost[seq_idx, m])

                self.mixModels[m]._accumulate_sufficient_statistics(
                    stats['mix_idx' + str(m)], X[i:j, :], framelogprob,
                    posteriors,
                    fwdlattice, bwdlattice)

    def _compute_sufficient_statistics(self, X, y, lengths):
        '''
        Computes sufficient statistics to be used in the M-step.
        Inputs:
            X - np.array of size (n_samples, n_features).
            y - np.int of size n_sequences, whose entries are in the
                range [0, n_nodes-1].
            lengths - list containing the lengths of each individual sequence
                      in X, with size n_sequences.
        Outputs:
            stats - dictionary containing sufficient statistics.
        '''

        def check_large_diff(rates):
            return abs(rates[0] - rates[1]) == np.max(abs(rates[0] - rates[1])) #>= .9 #10

        def get_index_of_large_diff(rates):
            which_cluster = 0
            if (rates[0] - rates[1]).any() < 0:
                which_cluster = 1
            return which_cluster, np.where(abs(rates[0] - rates[1]) == np.max(abs(rates[0] - rates[1])))[0][0]# np.where(abs(rates[0] - rates[1]) >= .9)[0][0]

        means = np.zeros((self.mix_dim, self.mixModels[0].n_components))
        for m in range(self.mix_dim):
            means[m] = self.mixModels[m].means_.flatten()

        i = self.n_components  # some init value so that it never collides with possible indexes of the arrays
        greater_cluster = 0
        if check_large_diff(means).any():
            greater_cluster, i = get_index_of_large_diff(means)


        stats = {'mix_post': np.zeros((self.n_nodes, self.mix_dim))}
        for m in range(self.mix_dim):
            stats['mix_idx' + str(m)] = (self.mixModels[m]._initialize_sufficient_statistics())

            stats['trans_prior' + str(m)] = np.ones((self.n_components, self.n_components))

            print(means[m])

        logmixpost, trans = self._compute_mixture_posteriors(X, y, lengths)

        for k in range(self.n_nodes):
            stats['mix_post'][k, :] = np.sum(np.exp(logmixpost[y == k, :]),
                                             axis=0)

        logmixpost -= np.amax(logmixpost, axis=0).reshape(1, self.mix_dim)

        self._compute_sufficient_statistics_in_mix_comp(X, y, lengths,
                                                        logmixpost, stats)

        if self.reg_:
            stats['n_seqs_per_node'] = np.zeros(self.n_nodes)
            for k in range(self.n_nodes):
                stats['n_seqs_per_node'][k] = np.sum(y == k)

        return stats

    def _fit_coef(self, stats):
        '''
        Performs the M step of the EM algorithm for the mixture coefficients,
        via gradient ascent. This function is used only when a graph is given.
        Inputs:
            stats - dictionary containing sufficient statistics.
            n_iter - number of update iterations.
        '''
        Nseqs = np.sum(stats['n_seqs_per_node'])
        for it in range(self.n_iter_mstep):
            grad = np.zeros_like(self.mixCoefUnNorm)
            post_coef_dif = (stats['mix_post']
                             - self.mixCoef * (stats['n_seqs_per_node']
                                               .reshape(-1, 1)))
            G_mixCoef = self.graph @ self.mixCoef
            reg_dif = (self.mixCoef *
                       (G_mixCoef - (np.sum(self.mixCoef * G_mixCoef, axis=1)
                                     .reshape(-1, 1))))
            mask = (self.mixCoefUnNorm > 0.)
            grad[mask] = (drelu(self.mixCoefUnNorm[mask])
                          / relu(self.mixCoefUnNorm[mask]))
            grad *= post_coef_dif/Nseqs + reg_dif

            self.mixCoefUnNorm = self._adam(self.mixCoefUnNorm, grad)

            self.mixCoef = relu_normalization(self.mixCoefUnNorm, axis=1)

    def _adam(self, w, dw, delta=1e-8):
        '''
        Performs an ascending step using the Adam algorithm.
        Inputs:
            w - np.array, the current value of the parameter.
            dw - np.array with the same shape as w, the gradient of the
                 objective with respect to w.
            delta - small constant to avoid division by zero (default: 1e-8)
        Outputs:
            next_w - np.array with the same shape as w, the updated value of
                     the parameter.
        '''
        next_first_moment = self.rho1 * self.first_moment_ + (1-self.rho1) * dw
        next_second_moment = (self.rho2 * self.second_moment_
                              + (1 - self.rho2) * dw**2)

        correct_first_moment = next_first_moment / (1 - self.rho1**self.time_)
        correct_second_moment = (next_second_moment
                                 / (1 - self.rho2**self.time_))

        upd_w = (self.lr_mstep * correct_first_moment
                 / (np.sqrt(correct_second_moment) + delta))
        next_w = w + upd_w

        self.time_ += 1
        self.first_moment_ = next_first_moment
        self.second_moment_ = next_second_moment

        return next_w

    def _do_mstep(self, stats):
        '''
        Performs the M step of the EM algorithm, updating all model parameters.
        Inputs:
            stats - dictionary containing sufficient statistics.
        '''
        if self.reg_:
            self._fit_coef(stats)
        else:
            self.mixCoef = stats['mix_post']
            normalize(self.mixCoef, axis=1)

        for m in range(self.mix_dim):
            self.mixModels[m].transmat_prior = stats['trans_prior' + str(m)]
            self.mixModels[m]._do_mstep(stats['mix_idx' + str(m)])

    def fit(self, X, y, lengths=None, valid_data=None):
        '''
        Trains SpaMHMM on data X, y, using the EM algorithm.
        Inputs:
            X - np.array of size (n_samples, n_features).
            y - np.int of size n_sequences, whose entries are in the
                range [0, n_nodes-1].
            lengths - list containing the lengths of each individual sequence
                      in X, with size n_sequences.
            valid_data - tuple (X_valid, y_valid, lengths_valid) containing the
                         validation data; if validation data is given, the
                         model with the lowest validation loss is saved in a
                         pickle file (optional, default:None).
        '''
        if type(X) == list:
            lengths = [x.shape[0] for x in X]
            X = np.concatenate(X)
            y = np.array(y)

        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, False)

        if valid_data is not None:
            X_valid, y_valid, lengths_valid = valid_data

            if type(X_valid) == list:
                lengths_valid = [x.shape[0] for x in X_valid]
                X_valid = np.concatenate(X_valid)
                y_valid = np.array(y_valid)

            max_validscore = float('-inf')
            validloss_hist = []

        if self.graph is not None:
            self.reg_ = True
        else:
            self.reg_ = False

        self.init_params(X)
        self._check()

        for m in range(self.mix_dim):
            #self.mixModels[m].means_ = np.array([np.min(X), np.max(X) / 4, np.max(X) / 2, np.max(X)])[:, np.newaxis]
            self.mixModels[m].means_ = np.sort(self.mixModels[m].means_.flatten())[:, np.newaxis] #
            #self.mixModels[m].means_ = np.array([4, 9, 17, 25])[:, np.newaxis]

        prevscore = float('-inf')
        trainloss_hist = []
        for it in range(self.n_iter):
            t0 = time.time()
            stats = self._compute_sufficient_statistics(X, y, lengths)

            #similarity = diversified_hmm.get_kernel(self.mixModels[0].transmat_, self.mixModels[1].transmat_) * 1000000000000000
            norm0 = math.sqrt(np.trace(self.mixModels[0].transmat_.dot(self.mixModels[0].transmat_)))
            norm1 = math.sqrt(np.trace(self.mixModels[1].transmat_.dot(self.mixModels[1].transmat_)))
            similarity = np.trace(self.mixModels[0].transmat_.dot(self.mixModels[1].transmat_)) / (norm0 * norm1)
            from scipy.linalg import det
            #similarity = diversified_hmm.get_prior(np.dot(self.mixModels[0].transmat_.T, self.mixModels[1].transmat_))
            self.mixModels[0].similarity = similarity #diversified_hmm.calculate_entropy(self.mixModels[0].transmat_)
            self.mixModels[1].similarity = similarity #diversified_hmm.calculate_entropy(self.mixModels[1].transmat_)

            #print('similarity {}'.format(similarity))

            self.mixModels[0].iter = it

            # if it == 0:
            #     self.mixModels[0].other_trans = np.array([[.1, 0, .3, .6], [.1, 0, .5, .4], [0, .4, .4, .2], [.2, .2, .2, .4]])
            #     #self.mixModels[1].other_trans = np.array([[.7, .1, .2, 0], [.1, 0, .2, .7], [.5, .1, .4, 0], [0, .2, .1, .7]])
            # else:
            self.mixModels[0].other_trans = self.mixModels[1].transmat_
            self.mixModels[1].other_trans = self.mixModels[0].transmat_

            #self._do_mstep(stats)
            if self.reg_:
                self._fit_coef(stats)
            else:
                self.mixCoef = stats['mix_post']
                normalize(self.mixCoef, axis=1)
            self.mixModels[0].transmat_prior = stats['trans_prior' + str(0)]
            self.mixModels[0]._do_mstep(stats['mix_idx' + str(0)])
            self.mixModels[1].other_trans = self.mixModels[0].transmat_ # take the update from this iter
            self.mixModels[1].transmat_prior = stats['trans_prior' + str(1)]
            self.mixModels[1]._do_mstep(stats['mix_idx' + str(1)])
            ####

            print('trans0 {}'.format(self.mixModels[0].transmat_  ))
            print('trans1 {}'.format(self.mixModels[1].transmat_  ))

            t1 = time.time()

            currscore = self.score(X, y, lengths)
            trainloss_hist.append(-currscore)
            if valid_data is not None:
                validscore = self.score(X_valid, y_valid, lengths_valid)
                validloss_hist.append(-validscore)

                if validscore > max_validscore:
                    max_validscore = validscore
                    f = open(self.name + '.pkl', 'wb')
                    pickle.dump(self, f)

            if self.verbose:
                if (not self.reg_) and (prevscore > currscore):
                    print('WARNING: loss has increased at iteration {}!'
                          .format(it))
                    print('prev loss = {:.5f}, curr loss = {:.5f}'
                          .format(-prevscore, -currscore))
                elif valid_data is not None:
                    print('it {}: train loss = {:.5f}, valid loss = {:.5f}, '
                          '{:.3f} sec/it'.format(it+1, -currscore, -validscore,
                                                 t1-t0))
                else:
                    print('it {}: loss = {:.5f}, {:.3f} sec/it'
                          .format(it+1, -currscore, t1-t0))


            ll = np.sum(self.scores_per_seq(X, y, lengths))
            print("ll: {}".format(ll))

            # ll = 0
            # for m in range(self.mix_dim):
            #     ll += np.max(self.mixModels[m]._compute_log_likelihood(X))


            self.monitor_.report(currscore)
            # self.monitor_.report(ll)
            # print("ll: {}".format(ll))

            if self.monitor_.converged:
                if self.verbose:
                    print('Loss improved less than {}. Training stopped.'
                          .format(self.tol))
                break

            prevscore = currscore

        if valid_data:
            return trainloss_hist, validloss_hist
        else:
            return trainloss_hist

    def predict_next_observ(self, X, y, x_candidates):
        '''
        Finds the most likely next observation, given the sequence X at node y,
        by trying every candidate point in x_candidates.
        Inputs:
            X - observed sequence, np.array of size (length, n_features).
            y - the node index, integer.
            x_candidates - candidate points, np.array of size (n_candidates,
                           n_features).
        Outputs:
            next_observ - predicted observation, np.array of size n_features.
        '''
        length = X.shape[0]
        Ncand = x_candidates.shape[0]
        ll_per_comp_X = np.zeros(self.mix_dim)
        ll_per_comp_nxt_obs = np.zeros((Ncand, self.mix_dim))
        for m in range(self.mix_dim):
            if self.mixCoef[y, m] == 0.:
                continue

            ll_per_comp_X[m], state_post = self.mixModels[m].score_samples(X)
            final_state_post = state_post[length-1, :]
            next_state_logpost = logsumexp((np.log(self.mixModels[m]
                                                   .transmat_.T)
                                            + np.log(final_state_post)),
                                           axis=1)
            emiss_ll = self.mixModels[m]._compute_log_likelihood(x_candidates)
            ll_per_comp_nxt_obs[:, m] = logsumexp(emiss_ll
                                                  + (next_state_logpost
                                                     .reshape(1, -1)), axis=1)

        nonzero_idx = (self.mixCoef[y, :] != 0.)
        ll_next_observ = logsumexp((ll_per_comp_nxt_obs[:, nonzero_idx]
                                    + ll_per_comp_X[nonzero_idx]
                                    + np.log(self.mixCoef[y, nonzero_idx])),
                                   axis=1)

        max_idx = np.argmax(ll_next_observ)
        next_observ = x_candidates[max_idx, :]

        return next_observ

    def greedy_predict_seq(self, X, y, x_candidates, n_samples):
        '''
        Finds a greedy approximation of the most likely next n_samples, given
        the  sequence X at node y, trying every candidate point in x_candidates
        for each sample.
        Inputs:
            X - observed sequence, np.array of size (length, n_features).
            y - the node index, integer.
            x_candidates - candidate points, np.array of size (n_candidates,
                           n_features).
        Outputs:
            Xpred - predicted sequence, np.array of size (n_samples,
                    n_features).
        '''
        length = X.shape[0]
        Xpred = X
        for i in range(n_samples):
            next_observ = (self.predict_next_observ(Xpred, y, x_candidates)
                           .reshape(1, -1))
            Xpred = np.concatenate([Xpred, next_observ], axis=0)

        return Xpred[length::, :]

    def sample(self, y, n_samples, Xpref=None):
        '''
        Samples a sequence of observations from the MHMM observation
        distribution. If a prefix sequence is given, the new sequence is
        sampled from the posterior distribution given that prefix sequence.
        Inputs:
            y - the node index, integer.
            n_samples - the number of samples, integer.
            Xpref - prefix sequence, np.array of size (pref_len, n_features)
                    (optional, default: None).
        Outputs:
            X - sampled sequence, np.array of size (n_samples, n_features)
            mix_idx - the component which the sequence was sampled from,
                      integer.
            state_seq - the produced state sequence, np.int of size n_samples.
        '''
        if Xpref is not None:
            pref_len = Xpref.shape[0]
            mix, trans = self._compute_mixture_posteriors(Xpref, y, [pref_len])
            mix_post = np.exp(mix)
        else:
            mix_post = self.mixCoef[y, :]

        mix_idx = np.random.choice(self.mix_dim, p=mix_post.reshape(-1))

        if Xpref is not None:
            state_prior = self.mixModels[mix_idx].startprob_
            state_post = self.mixModels[mix_idx].predict_proba(Xpref)[-1, :]
            self.mixModels[mix_idx].startprob_ = state_post

        X, state_seq = self.mixModels[mix_idx].sample(n_samples=n_samples)

        if Xpref is not None:
            self.mixModels[mix_idx].startprob_ = state_prior

        return X, mix_idx, state_seq