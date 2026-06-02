# Reverse Engineering Strut-and-Tie Models
# ================================
# Karin Yu, 2026

# Part 1: Layout optimisation
# Solver

__author__ = 'Karin Yu'

import numpy as np
from scipy import sparse
import cvxpy as cvx
from src.problem import adapt_layout, reduce_members
from src.plotter import plotTruss
from shapely.geometry import LineString
from shapely.ops import split
import matplotlib.pyplot as plt

# calculate equilibrium matrix B
# Code from He, L., M. Gilbert and X. Song: "A Python script for adaptive layout optimization of truss structures", Struct. Multidisc. Optim. 2019
def calcB(Nd, Cn, dof):
    m, n1, n2 = len(Cn), Cn[:,0].astype(int), Cn[:,1].astype(int)
    l, X, Y = Cn[:,2], Nd[n2,0]-Nd[n1,0], Nd[n2,1]-Nd[n1,1]
    l[l <= 1e-6] = 1e-5
    d0, d1, d2, d3 = dof[n1*2], dof[n1*2+1], dof[n2*2], dof[n2*2+1]
    s = np.concatenate((-X/l*d0, -Y/l*d1, X/l*d2, Y/l*d3))
    r = np.concatenate((n1*2, n1*2+1, n2*2, n2*2+1))
    c = np.concatenate((np.arange(m), np.arange(m), np.arange(m), np.arange(m)))
    return sparse.coo_matrix((s, (r, c)), shape=(len(Nd)*2, m))

def checkEquilibrium(Nd, CnC, CnS, f, dof, qC, qS, tol = 1e-2):
    # function checks static equilibrium in all nodes
    print('### Checking equilibrium')
    BC = calcB(Nd, CnC, dof) # len(f) x len(CnC) = number of struts
    BS = calcB(Nd, CnS, dof) # len(f) x len(CnS) = number of ties
    fC = BC@qC
    fS = BS@qS
    is_satisfied = True
    for i in range(len(f)):
        if abs(fC[i]+fS[i]-f[i]) > tol:
            print('# Equilibrium not satisfied at node %d, Error: %f, force: %f, Sum of C: %f, Sum of S: %f' %(i, fC[i]+fS[i]-f[i], f[i], fC[i], fS[i]))
            k = i // 2
            j = i % 2
            print('# Node: ', Nd[k, :], 'in direction of', 'x' if j == 0 else 'y')
            is_satisfied = False
        if abs(f[i]) > 0:
            print('## External force at node %d, force: %f' % (i, f[i]))
    if is_satisfied:
        print('## Equilibrium is satisfied for all nodes.')
    else:
        print('## Equilibrium is not satisfied for all nodes!!!')

def solve(Nd, CnC, CnS, f, dof, sC, sS, W, member_penalty, weight_W = 1e-2, weight_JP = 1e-4): #, weight_W = 1e-2, weight_JP = 1e-4
    # CnC structure: [Node index i, Node index j, length, concrete area for compression]
    # CnS structure: [Node index i, Node index j, length, concrete area for compression, steel area for tension]
    # solves linear programming problem
    # get equilibrium matrices
    BC = calcB(Nd, CnC, dof) # len(f) x len(CnC) = number of struts
    BS = calcB(Nd, CnS, dof) # len(f) x len(CnS) = number of ties
    # initialize variables
    f_int_S = cvx.Variable(len(CnS))
    f_int_C = cvx.Variable(len(CnC))
    load_factor = cvx.Variable(1) # maxmimize load factor
    # define areas for constraint
    a_S_T = np.array([col[4] for col in CnS]) # positive, steel
    a_S_C = np.array([col[3] for col in CnS])  # negative, steel
    a_C = np.array([col[3] for col in CnC]) # negative, concrete
    l_C = np.array([col[2]+member_penalty for col in CnC]) # length of concrete struts
    # objective using standard L1-norm heuristic with W
    obj = cvx.Minimize(-load_factor + weight_W*W@cvx.abs(f_int_C)+weight_JP*cvx.sum(cvx.abs(f_int_C)@l_C/sC))
    f_array = np.array(f)
    # add boundary conditions
    cons = []
    # equilibrium boundary
    cons.append(BC@f_int_C + BS@f_int_S == f_array*dof*load_factor)
    # lower and upper limit boundary
    cons.extend([f_int_S >= -sC*a_S_C, f_int_S <= sS*a_S_T, f_int_C >= -sC*a_C, f_int_C <= 0, load_factor >= 1])
    prob = cvx.Problem(obj, cons)
    # solve with MOSEK optimizer
    obj_value = prob.solve(solver='MOSEK', accept_unknown=True)
    # assign final results
    qS = np.array([np.array(qi.value).flatten() for qi in f_int_S])
    qC = np.array([np.array(qi.value).flatten() for qi in f_int_C])
    return obj_value, qS, qC, load_factor.value

def solveProblem(Nd, CnC, CnS, f, dof, fsd, fcd, bc_coords, reinforcement, max_incl_prev, polygon_in, polygon_out, ac_min, max_num_iter = 10, max_threshold_S = 0.0001, max_threshold_C = 0.01, inclination_increase = 0.25, **kwargs):
    # solves Part 1: Layout optimisation
    # initialize weights for standard L1-norm heuristic
    W = np.ones(len(CnC))
    delta = 1e-9 # for weight update
    # initialize objective value with large value
    obj_value_prev = 1e9
    for itr in range(0, max_num_iter):
        print('### Start iteration %d' % int(itr+1))
        obj_value, qS, qC, lF = solve(Nd, CnC, CnS, f, dof, fcd, fsd, W, **kwargs)
        print('1. load factor: %f, #struts: %d, #ties: %d' % (lF, len(CnC), len(CnS)))
        if itr == max_num_iter-1:
            break
        else:
            # thresholding
            threshold_S = max_threshold_S * max(qS)
            threshold_C = max_threshold_C * min(qC)
            print('Thresholds S: %f, C: %f' % (threshold_S, threshold_C))
            CnC, CnS, qC, qS = apply_thresholding(CnC, CnS, qC, qS, threshold_C, threshold_S)
            # adapt layout based on larger inclinations
            Nd_active = get_active_nodes(Nd, CnC, CnS, qC, qS, threshold_C, threshold_S)
            print('Active nodes:', sum(Nd_active), 'of', len(Nd))
            print('Allowed inclination:', max_incl_prev*(1+inclination_increase))
            # Adapt layout based on increased inclination and strut angle requirement.
            CnC_new = adapt_layout(Nd, Nd_active, CnS, CnC, qS, bc_coords, reinforcement, max_incl_prev, max_incl_prev*(1+inclination_increase), polygon_in, polygon_out, ac_min, threshold_S = threshold_S)
            max_incl_prev += inclination_increase*max_incl_prev
            if CnC_new.shape[0] == 0:
                if (abs(obj_value_prev - obj_value) < 1e-6 or np.isnan(obj_value)):
                    print('No new struts found, stopping iteration')
                    break
            else:
                CnC = np.concatenate((CnC, CnC_new), axis=0)
                qC = np.concatenate((np.array(qC), np.ones((CnC_new.shape[0], 1))), axis=0)
            # recalculate weights
            W = np.ones(len(CnC)) / (delta*np.ones(len(CnC))+ np.abs(qC.flatten())) # to promote sparsity of f_int_C
            delta *= 0.9 # decrease delta to increase sparsity
            obj_value_prev = obj_value
        print('Maximum forces: ', max(qS), min(qC))
        print('Filtered mems C: %d, mems S: %d' % (len(CnC), len(CnS)))
    return CnC, CnS, qC, qS, lF, obj_value_prev

def get_active_nodes(Nd, CnC, CnS, qC, qS, threshold_C, threshold_S):
    # gets a reduced set of nodes of Nd, which are active (have members with "larger" forces)
    active_nodes = [False for _ in range(len(Nd))]
    for i in range(len(qC)):
        if qC[i] <= threshold_C:
            active_nodes[int(CnC[i,0])] = True
            active_nodes[int(CnC[i,1])] = True
    for i in range(len(qS)):
        if qS[i] >= threshold_S:
            active_nodes[int(CnS[i, 0])] = True
            active_nodes[int(CnS[i, 1])] = True
    return active_nodes

def apply_thresholding(CnC, CnS, qC, qS, threshold_C, threshold_S):
    # applied thresholding, first to struts
    CnC = CnC[np.array(qC).flatten() <= threshold_C]
    qC = qC[np.array(qC).flatten() <= threshold_C]
    # add negative ties to struts
    if CnS[np.array(qS).flatten() <= threshold_C].shape[0] > 0:
        CnC = np.concatenate((CnC, CnS[np.array(qS).flatten() <= threshold_C][:, :4]), axis=0)
        qC = np.concatenate((qC, qS[np.array(qS).flatten() <= threshold_C]))
    CnS = CnS[np.array(qS).flatten() >= threshold_S]
    qS = qS[np.array(qS).flatten() >= threshold_S]
    return CnC, CnS, qC, qS