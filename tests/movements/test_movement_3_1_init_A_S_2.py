from oMHMM.tests.digits import spamhmm_orthogonal_init_A as oMHMM
from oMHMM.tests.digits import spamhmm_init_A as MHMM
import numpy as np
import csv
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import accuracy_score


input = np.empty((0, 91), dtype=object)
with open('./movement_libras.data', encoding='US-ASCII') as f:
    reader = csv.reader(f)
    for row in reader:
        input = np.vstack((input, row))

input = input.astype(float)

data1 = np.empty((0, 90), dtype=float)
data2 = np.empty((0, 90), dtype=float)
i = 0
for label in input[:, -1]:
    if label == 3:
        data1 = np.vstack((data1, input[i, :-1]))
    elif label == 1:
        data2 = np.vstack((data2, input[i, :-1]))
    i += 1

data = np.concatenate((data1[:, :], data2[:, :]), axis=0)

np.random.seed(1)

num_of_cells = len(data[:, 0])
l = 90
y = np.zeros(num_of_cells, dtype=int)
M = 2
S = 2 # 2: 3 MHMM, 3 oMHMM; 4: 5 MHMM, 6 oMHMM
lengths = (np.ones(num_of_cells, dtype=int) * l).tolist()
###################################################################################################
mhmm = oMHMM.SpaMHMM(epsilon=.0000001, n_nodes=1,
                       mix_dim=M,
                       n_components=S,
                       n_features=l,
                       graph=None,
                       n_iter=100,
                       verbose=True,
                       name='mhmm')
mhmm.fit(data.flatten()[:, np.newaxis], y, lengths)
pi_nk, transitions = mhmm._compute_mixture_posteriors(data.flatten()[:, np.newaxis], y, lengths)

print(transitions.reshape(M, S, S))

print(np.exp(pi_nk))

predicted_cluster = []
label_0 = [0 for i in range(len(data1[:, 0]))]
label_1 = [1 for i in range(len(data2[:, 0]))]
for n in range(num_of_cells):
    cell = np.float64(pi_nk[n])
    predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell))[0][0])

label = label_0 + label_1

print(label)
print(predicted_cluster)

v = v_measure_score(label, predicted_cluster)
print('oMHMM v-measure: {}'.format(v))

acc = accuracy_score(label, predicted_cluster)
print('oMHMM acc: {}'.format(acc))

###################################################################################################

mhmm = MHMM.SpaMHMM(n_nodes=1,
                       mix_dim=M,
                       n_components=S,
                       n_features=l,
                       graph=None,
                       n_iter=100,
                       verbose=True,
                       name='mhmm')
mhmm.fit(data.flatten()[:, np.newaxis], y, lengths)
pi_nk, transitions = mhmm._compute_mixture_posteriors(data.flatten()[:, np.newaxis], y, lengths)

print(transitions.reshape(M, S, S))

print(np.exp(pi_nk))

predicted_cluster = []
label_0 = [0 for i in range(len(data1[:, 0]))]
label_1 = [1 for i in range(len(data2[:, 0]))]
for n in range(num_of_cells):
    cell = np.float64(pi_nk[n])
    predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell))[0][0])

label = label_0 + label_1

print(label)
print(predicted_cluster)

v = v_measure_score(label, predicted_cluster)
print('MHMM v-measure: {}'.format(v))

acc = accuracy_score(label, predicted_cluster)
print('MHMM acc: {}'.format(acc))


