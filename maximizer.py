import numpy as np
from cvxpy import *


def maximize_new(gamma, other_trans, eps):
    J = gamma.shape[0]
    # Describe the concave problem
    A = Variable((J, J))
    log_A = cvxpy.log(A)
    term_1 = cvxpy.multiply(gamma, log_A)
    mult = cvxpy.multiply(A, other_trans)
    penalty = cvxpy.sum(cvxpy.sum(mult, axis=1))
    obj = Maximize(cvxpy.sum(cvxpy.sum(term_1, axis=1)) - penalty)
    constr = [A * np.ones(J) == np.ones(J), A >= 0]
    prob = Problem(obj, constr)

    # solve the problem
    prob.solve()

    print("Problem Status: {}".format(prob.status))
    print("Optimal value = : {}".format(prob.value))
    print("Optimal solution = : {}".format(A.value))

    return A.value + eps # eps is to cope with the bug in cvxpy producing a negative A_ij sometimes


def maximize_multi_clusters(gamma, other_trans, eps):
    J = gamma.shape[0]
    # Describe the concave problem
    A = Variable((J, J))
    log_A = cvxpy.log(A)
    term_1 = cvxpy.multiply(gamma, log_A)
    mult1 = cvxpy.multiply(A, other_trans[0])
    mult2 = cvxpy.multiply(A, other_trans[1])
    mult = mult1 + mult2
    penalty = cvxpy.sum(cvxpy.sum(mult, axis=1))
    obj = Maximize(cvxpy.sum(cvxpy.sum(term_1, axis=1)) - penalty)
    constr = [A * np.ones(J) == np.ones(J), A >= 0]
    prob = Problem(obj, constr)

    # solve the problem
    prob.solve()

    print("Problem Status: {}".format(prob.status))
    print("Optimal value = : {}".format(prob.value))
    print("Optimal solution = : {}".format(A.value))

    return A.value + eps # to cope with the bug in cvxpy producing a negative A_ij sometimes


