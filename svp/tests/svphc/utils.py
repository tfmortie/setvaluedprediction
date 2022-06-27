"""
Some additional functions needed for the experiments for paper "Set-valued prediction in hierarchical classification with constrained representation complexity"

Author: Thomas Mortier
Date: January 2022
"""
import cvxpy
import time
import torch
import numpy as np

from svp.utils import HLabelTransformer
from itertools import combinations

""" Get Torch tensor (S x K, with S the search space and K the number of classes) which represents the hierarchy """


def get_hstruct_tensor(classes, params):
    # first extract hstruct
    hlt = HLabelTransformer(sep=";")
    hlt.fit(classes)
    hstruct = hlt.hstruct_
    hstruct_t = []
    rc = params["c"]
    if params["svptype"] == "sizectrl":
        rc = min(params["size"], rc)
    for ci in range(1, rc + 1):
        combs = list(combinations(hstruct, ci))
        for c in combs:
            if ci == 1:
                c = c[0]
                if params["svptype"] == "sizectrl":
                    if params["size"] >= len(c):
                        crow = np.zeros((1, len(hstruct[0])), dtype=np.uint8)
                        crow[0, c] = 1
                        hstruct_t.append(crow)
                else:
                    crow = np.zeros((1, len(hstruct[0])), dtype=np.uint8)
                    crow[0, c] = 1
                    hstruct_t.append(crow)
            else:
                if len(set(sum(list(c), []))) == len(sum(list(c), [])) and (
                    set(sum(list(c), [])) not in [set(n) for n in hstruct]
                ):
                    if params["svptype"] == "sizectrl":
                        if params["size"] >= len(sum(c, [])):
                            crow = np.zeros((1, len(hstruct[0])), dtype=np.uint8)
                            crow[0, sum(c, [])] = 1
                            hstruct_t.append(crow)
                    else:
                        crow = np.zeros((1, len(hstruct[0])), dtype=np.uint8)
                        crow[0, sum(c, [])] = 1
                        hstruct_t.append(crow)

    return torch.tensor(np.concatenate(hstruct_t, axis=0))


""" Function which returns A and b matrix for the KCG problem """


def pwk_ilp_get_ab(hstruct, params):
    A = []
    A.append(np.array([len(s) for s in hstruct]))
    # add 1
    A.append(np.ones(len(hstruct)))
    # add E
    # run over adjecency matric
    for i in range(len(hstruct)):
        for j in range(i + 1, len(hstruct)):
            if len(set(hstruct[i]) & set(hstruct[j])) > 0:
                # we have found an edge
                e = np.zeros(len(hstruct))
                e[i] = 1
                e[j] = 1
                A.append(e)
    A = np.vstack(A)
    # construct b
    b = np.ones(A.shape[0])
    b[0] = params["size"]
    b[1] = params["c"]

    return A, b


""" Solve KCG problem """


def pwk_ilp(P, A, b, hstruct, solver, transformer):
    o_t = []
    t = 0.0
    for pi in P:
        # get p
        p = []
        for s in hstruct:
            p_s = 0
            for si in s:
                p_s += pi[si]
            p.append(p_s)
        p = np.array(p)
        # solve our ILP
        start_time = time.time()
        selection = cvxpy.Variable(len(hstruct), boolean=True)
        constraint = A @ selection <= b
        utility = p @ selection
        knapsack_problem = cvxpy.Problem(cvxpy.Maximize(utility), [constraint])
        if solver == "GLPK_MI":
            knapsack_problem.solve(solver=cvxpy.GLPK_MI)
        elif solver == "CBC":
            knapsack_problem.solve(solver=cvxpy.CBC)
        else:
            knapsack_problem.solve(solver=cvxpy.SCIP)
        stop_time = time.time()
        t += stop_time - start_time
        sel_ind = list(np.where(selection.value)[0])
        pred = []
        for i in sel_ind:
            pred.extend(hstruct[i])
        o_t.append(pred)
    t /= P.shape[0]

    # inverse transform sets
    o = []
    for o_t_i in o_t:
        o_t_i = transformer.inverse_transform(o_t_i)
        o.append(o_t_i)

    return o, t
