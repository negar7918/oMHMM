import numpy as np
from scipy.stats import dirichlet


def generate_categorical_prob(num_of_categories, alpha=None):
    if alpha is None:
        alpha = np.random.mtrand.dirichlet([10] * num_of_categories)
    else:
        alpha = alpha * np.ones(num_of_categories)
    var = dirichlet.rvs(alpha=alpha, size=1, random_state=None)
    return var[0]


def generate_transitions_random(i, num_of_clusters, num_of_states, alpha):
    np.random.seed(i)
    trans = np.zeros((num_of_clusters, num_of_states, num_of_states))
    counter = 0
    for c in range(num_of_clusters):
        for s in range(num_of_states):
            trans[c, s] = generate_categorical_prob(num_of_states, alpha)
            counter += 1
    print(trans)
    return trans