import numpy as np
import csv
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
from oMHMM.tests.digits import spamhmm_init_A, spamhmm_init_B, spamhmm_orthogonal_init_A, spamhmm_orthogonal_init_B


def plot(c, name, color):
    fig, ax = plt.subplots(1)
    #plt.plot(np.arange(808), np.mean(c, axis=0), color=color)
    plt.scatter(np.arange(808), c, s=4, color=color)
    plt.xlabel('sequence position')
    plt.ylabel('average count')
    #ax.set_ylim([50, 210])
    ax.set_ylim([0, 400])
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title(name)
    plt.savefig('./cluster' + name + '.eps')


input = np.empty((0,37), dtype=object)
with open('./chrom4.csv', encoding='US-ASCII') as f:
    reader = csv.reader(f)
    for row in reader:
        input = np.vstack((input, row))


Xtrain = np.swapaxes(input[1:, 1:], 0, 1).astype(int)

data = Xtrain[:, :]

# plot(data[:18, :], 'metastatic cells', 'black')
# plot(data[18:, :], 'primary cells', 'blue')
#plot(data[15, :], 'cell M-67', 'black')
#plot(data[35:, :], 'cell P-8', 'blue')

np.random.seed(1)

l = 808
y = np.zeros(36, dtype=int)
M = 2
S = 3
lengths = (np.ones(36, dtype=int) * l).tolist()
mhmm = spamhmm_orthogonal_init_A.SpaMHMM(epsilon=0, n_nodes=1,
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
label = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for n in range(36):
    cell = np.float64(pi_nk[n])
    predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell))[0][0])
    if label[n] == 0:
        label[n] = 1
    else:
        label[n] = 0



print(label)
print(predicted_cluster)

v = v_measure_score(label, predicted_cluster)
print('v-measure: {}'.format(v))

acc = accuracy_score(label, predicted_cluster)
print('accuracy: {}'.format(acc))



