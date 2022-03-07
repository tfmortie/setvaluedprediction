import random
import cvxpy
import gurobipy as gp
from gurobipy import GRB
import copy
import networkx as nx
import argparse
import numpy as np
import pandas as pd
import heapq
import ast
import os
import itertools
import time
import torch
import matplotlib.pyplot as plt
from itertools import combinations
from IPython.display import Image, IFrame
from wand.image import Image as WImage

counter_proposal = 0

def numl_to_str(l):
    ret_str = "None"
    if l is not None:
        l = sorted(l)
        ret_str = '['
        for i,n in enumerate(l):
            if i==0:
                ret_str+=str(n)
                ret_str+=','
            else:
                if l[i-1]+1==n:
                    if ret_str[-1]!=':':
                        ret_str = ret_str[:-1]+':'
                    if i==len(l)-1:
                        ret_str+=str(n)+','
                else:
                    if ret_str[-1]==':':
                        ret_str+=str(l[i-1])+','
                    ret_str+=str(n)+','
        ret_str = ret_str[:-1]+"]"
        
    return ret_str  

"""
Priority queue needed for the RBOP inference algorithm
"""
class PriorityList():
    def __init__(self):
        self.list = []
        
    def push(self,prob,node):
        heapq.heappush(self.list,[1-prob,node])

    def pop(self):
        return heapq.heappop(self.list)

    def remove_all(self):
        self.list = []
        
    def size(self):
        return len(self.list)

    def is_empty(self):
        return len(self.list) == 0

    def __repr__(self):
        ret_str = ""
        for l in self.list:
            ret_str+="({0:.2f},{1}), ".format(1-l[0],l[1])
        return ret_str
    
def get_hstruct_tensor(K, hstruct, params):
    hstruct_t = []
    rc = params["c"]
    if params["framework"] == "size":
        rc = min(params["k"],rc)
    for ci in range(1,rc+1):
        combs = list(combinations(hstruct, ci))
        for c in combs:
            if ci==1:
                c=c[0]
                if params["framework"] == "size":
                    if params["k"] >= len(c):
                        crow = np.zeros((1,K),dtype=np.uint8)
                        crow[0,c] = 1
                        hstruct_t.append(crow)
                else:
                    crow = np.zeros((1,len(hstruct[0])),dtype=np.uint8)
                    crow[0,c] = 1
                    hstruct_t.append(crow)
            else:
                if len(set(sum(list(c),[])))==len(sum(list(c),[])) and (set(sum(list(c),[])) not in [set(n) for n in hstruct]):
                    if params["framework"] == "size":
                        if params["k"] >= len(sum(c,[])):
                            crow = np.zeros((1,K),dtype=np.uint8)
                            crow[0,sum(c,[])] = 1
                            hstruct_t.append(crow)
                    else:
                        crow = np.zeros((1,K),dtype=np.uint8)
                        crow[0,sum(c,[])] = 1
                        hstruct_t.append(crow)
                        
    return torch.tensor(np.concatenate(hstruct_t,axis=0))
    
class GRBOP:
    def __init__(self, K, kmin=2, kmax=3, rsprob=2021, rshier=2021):
        self.K = K
        self.kmin = kmin
        self.kmax = kmax
        self.rsprob = rsprob
        self.rshier = rshier
        self.tree = {}
        # first create random hierarchy with K classes and probabilities
        np.random.seed(self.rsprob)
        lbls_to_process = list(zip([[c] for c in list(range(self.K))],list(np.random.dirichlet([1]*self.K,1)[0])))
        si = 0
        while len(lbls_to_process) > 1:
            random.Random(self.rshier).shuffle(lbls_to_process)
            nodes = []
            for i in range(min(random.Random(self.rshier+si).randint(self.kmin,self.kmax),len(lbls_to_process))):
                ch, chprob = lbls_to_process.pop(0)
                ch.sort()
                nodes.append((ch, chprob))
                # add child to tree if not yet in tree
                if str(ch) not in self.tree:
                    self.tree[str(ch)] = {"p": chprob, "ch": []}
            # add parent to tree
            parent_lbl = [item for sublist in [n[0] for n in nodes] for item in sublist]
            parent_lbl.sort()
            self.tree[str(parent_lbl)] = {"p": np.sum([n[1] for n in nodes]), "ch": [str(n[0]) for n in nodes]}
            # register parent to child nodes 
            for (n, nprob) in nodes:
                self.tree[str(n)]["parent"] = str(parent_lbl)
            lbls_to_process.append(([item for sublist in [n[0] for n in nodes] for item in sublist], np.sum([n[1] for n in nodes])))
            si+=1
            
    def visitnodes(self, filename="treenodes"):
        visited = []
        comb_list, p_list, c_list, k_list = [], [], [], []
        for ci in range(1,self.K):
            # first get nodes
            nodes = [ast.literal_eval(n) for n in self.tree.keys()]
            # get all possible combinations
            combs = list(itertools.combinations(nodes, ci))
            validcombs = []
            probs_validcombs = []
            for c in combs:
                # check if we have a valid combination
                prob = 0
                if ci==1:
                    for s in list(c):
                        prob += self.tree[str(s)]["p"]
                    validcombs.append(list(c))
                    probs_validcombs.append(prob)
                else:
                    if len(set(sum(list(c), []))) == len(sum(list(c), [])) and (set(sum(list(c), [])) not in [set(n) for n in nodes]):
                        for s in list(c):
                            prob += self.tree[str(s)]["p"]
                        validcombs.append(list(c))
                        probs_validcombs.append(prob)
            # now sort in terms of probabilities
            ret = sorted(zip(validcombs, probs_validcombs), key=lambda x: x[1])[::-1]
            for t in ret:
                ti = set(sum(t[0], []))
                if ti not in visited:
                    visited.append(ti)
                    comb_list.append(t[0])
                    p_list.append(t[1])
                    c_list.append(ci)
                    k_list.append(len(set(sum(list(t[0]), []))))
        df = pd.DataFrame({"yhat": comb_list, "p": p_list, "c": c_list, "k": k_list})
        df.to_csv("./"+filename+".csv", index=False)  
        
    def parsetreepdf(self, filename="tree.tex", scale=1.0):
        tree_structure = "\\Tree "+self.parse_tree(str(list(range(self.K))),1)
        doc = r'''\documentclass[landscape]{article}
        \usepackage[utf8]{inputenc}
        \usepackage[T1]{fontenc}
        \usepackage[margin=1in]{geometry}
        \usepackage{tikz-qtree}
        \usetikzlibrary{shadows,trees}
        \begin{document}
        \tikzset{font=\small,
        edge from parent fork down,
        level distance=1.75cm,
        every node/.style=
            {top color=white,
            rectangle,rounded corners,
            minimum height=8mm,
            draw=black!75,
            very thick,
            align=center,
            text depth = 0pt
            },
        edge from parent/.style=
            {draw=black!50,
            thick
            }}
        \centering
        \scalebox{'''+str(scale)+r'''}{
        \begin{tikzpicture}'''+tree_structure+r'''
        \end{tikzpicture}
        }
        \end{document}'''
        f = open(filename, "w")
        f.write(doc)
        f.close()
        
    def parsedebugpdf(self, filename="debug.tex", scale=1.0):
        tree_structure = "\\Tree "+self.debug
        doc = r'''\documentclass[landscape]{article}
        \usepackage[utf8]{inputenc}
        \usepackage[T1]{fontenc}
        \usepackage[margin=1in]{geometry}
        \usepackage{tikz-qtree}
        \usetikzlibrary{shadows,trees}
        \begin{document}
        \tikzset{font=\small,
        edge from parent fork down,
        level distance=1.75cm,
        every node/.style=
            {top color=white,
            rectangle,rounded corners,
            minimum height=8mm,
            draw=black!75,
            very thick,
            align=center,
            text depth = 0pt
            },
        edge from parent/.style=
            {draw=black!50,
            thick
            }}
        \centering
        \scalebox{'''+str(scale)+r'''}{
        \begin{tikzpicture}'''+tree_structure+r'''
        \end{tikzpicture}
        }
        \end{document}'''
        f = open(filename, "w")
        f.write(doc)
        f.close()  

    def parse_tree(self, node, prob):
        if len(ast.literal_eval(node))==1:
            return r'''[.{{$\{{{0}\}},{1}$}} ]'''.format(node.replace("[","").replace("]",""),str(round(prob,2)))
        else:
            ret_str = r'''[.{{$\{{{0}\}},{1}$}} '''.format(node.replace("[","").replace("]",""),str(round(prob,2)))
            for ch in self.tree[node]["ch"]:
                ret_str += self.parse_tree(ch,self.tree[ch]["p"])
            ret_str += r''' ] '''
            return ret_str
        
    def pw_bop(self, K, params, hstruct, verbose=False):
        # get probabilities 
        probs = []
        for l in range(K):
            probs.append(self.tree[str([l])]["p"])
        probs = torch.tensor(probs)
        # to gpu
        probs = probs.cuda()
        hstruct = hstruct.cuda()
        hstruct = hstruct.type(torch.float64)
        p = torch.matmul(hstruct,probs)
        si_optimal, si_optimal_u = 0, 0
        for i, pi in enumerate(p):
            si_curr_p = pi
            if params["framework"]=="size":
                if si_curr_p >= si_optimal_u:
                    si_optimal = i
                    si_optimal_u = si_curr_p    
            elif params["framework"]=="error":
                if si_curr_p >= 1-params["e"]:
                    si_curr_u = 1/hstruct[i,:].sum()
                    if si_curr_u >= si_optimal_u:
                        si_optimal = i
                        si_optimal_u = si_curr_p
            else:
                si_curr_size = hstruct[i,:].sum()
                si_curr_u = si_curr_p*((1+(params["b"]**2))/(si_curr_size+(params["b"]**2)))
                if si_curr_u >= si_optimal_u:
                    si_optimal = i
                    si_optimal_u = si_curr_u
        ystar = []
        for j, hj in enumerate(hstruct[si_optimal,:]):
            if hj==1:
                ystar.append(j)
                
        return ystar, si_optimal_u
                
    def pw_proposal(self, params, verbose=False):
        global counter_proposal
        visit_list = PriorityList()
        visit_list.push(1,str(list(range(self.K))))
        counter_proposal = 0
        pred = None
        if params["framework"] == "size":
            self.debug = r'''[.{{$\{{{0}\}},{1}$}} '''.format("","\\neg")
            pred = self._pw_proposal_size((None, 0), ([], 0), visit_list, params["k"], params["c"], verbose)
            self.debug += r''']'''
        elif params["framework"] == "error":
            pred = self._pw_proposal_error((None, 0), ([], 0), visit_list, params["e"], params["c"], verbose)
        else:
            pred = self._pw_proposal_utility((None, 0), ([], 0), visit_list, params["u"], params["c"], verbose)
        #print("[info] number of solutions = {0}".format(counter_proposal))
        
        return pred
            
    def _pw_proposal_size(self, ys, yc, Q, k, c, verbose=False):
        """
            ys : tuple (list ystar, float ystar_prob) which represent best solution so far
            yc : tuple (list ycur, float ycur_prob) which represent current solution/node in extended search tree
            Q : priority queue which consists of nodes to visit
            k : restriction set size
            c : current level in extended search tree 
            verbose : >0 prints info to output
        """
        global counter_proposal
        if verbose:
            print("ys: " + str(ys))
            print("yc: " + str(yc))
            print("Q: " + str(Q))
            print("k: " + str(k))
            print("c: " + str(c))
            print("")
        ystar, ystar_prob = ys
        # run over candidates in Q to add to current combination yc
        while Q.size() != 0:
            counter_proposal+=1
            node_prob, node = Q.pop()
            node_prob = 1-node_prob
            node_l = ast.literal_eval(node) 
            if len(node_l)+len(yc[0]) <= k:
                if yc[1]+node_prob >= ystar_prob:
                    # we have found a new optimal solution, hence update
                    ystar, ystar_prob = yc[0]+node_l, yc[1]+node_prob
            if len(node_l)+len(yc[0]) <= k:
                if c>1:
                    if len(node_l)+len(yc[0]) != k:
                        # no, hence, calculate best solution for the new combination yc[0]+node_l
                        ystar, ystar_prob = self._pw_proposal_size((ystar, ystar_prob), (yc[0]+node_l, yc[1]+node_prob), copy.deepcopy(Q), k, c-1, verbose)
                else: 
                    # yes, hence, we don't need to visit other possible combinations for current solution yc[0] (as prob. mass of consecutive solutions are going to be equal or smaller)
                    break
            # if current node is not a leaf node -> visit children
            if len(node_l) > 1:
                for ch in self.tree[node]["ch"]:
                    ch_prob = self.tree[ch]["p"]
                    Q.push(ch_prob, ch)
            else:
                # current nodes is a leaf node -> stopping criterion! 
                break
        
        return ystar, ystar_prob
    
    def _pw_proposal_error(self, ys, yc, Q, e, c, verbose=False):
        """
            ys : tuple (list ystar, float ystar_prob) which represent best solution so far
            yc : tuple (list ycur, float ycur_prob) which represent current solution/node in extended search tree
            Q : priority queue which consists of nodes to visit
            e : restriction error
            c : current level in extended search tree 
            verbose : >0 prints info to output
        """
        if verbose:
            print("ys: " + str(ys))
            print("yc: " + str(yc))
            print("Q: " + str(Q))
            print("e: " + str(e))
            print("c: " + str(c))
            print("")
        ystar, ystar_u = ys
        # run over candidates in Q to add to current combination yc
        while Q.size() != 0:
            node_prob, node = Q.pop()
            node_prob = 1-node_prob
            node_l = ast.literal_eval(node)
            if node_prob+yc[1] >= 1-e:
                if 1/(len(node_l)+len(yc[0])) > ystar_u:
                    # we have found a new optimal solution, hence update
                    ystar, ystar_u = yc[0]+node_l, 1/(len(node_l)+len(yc[0]))   
            if node_prob+yc[1] < 1-e:
                if c>1:
                    ystar, ystar_u = self._pw_proposal_error((ystar, ystar_u), (yc[0]+node_l, yc[1]+node_prob), copy.deepcopy(Q), e, c-1, verbose)
                else:
                    break 
            # if current node is not a leaf node -> visit children
            if len(node_l) > 1:
                for ch in self.tree[node]["ch"]:
                    ch_prob = self.tree[ch]["p"]
                    Q.push(ch_prob, ch)
            else:
                # current nodes is a leaf node -> stopping criterion! 
                break       
        return ystar, ystar_u

    def _pw_proposal_utility(self, ys, yc, Q, u, c, verbose=False):
        """
            ys : tuple (list ystar, float ystar_prob) which represent best solution so far
            yc : tuple (list ycur, float ycur_prob) which represent current solution/node in extended search tree
            Q : priority queue which consists of nodes to visit
            u : utility function 
            c : current level in extended search tree 
            verbose : >0 prints info to output
        """
        if verbose:
            print("ys: " + str(ys))
            print("yc: " + str(yc))
            print("Q: " + str(Q))
            print("u: " + str(u))
            print("c: " + str(c))
            print("")
        ystar, ystar_u = ys
        # run over candidates in Q to add to current combination yc
        while Q.size() != 0:
            node_prob, node = Q.pop()
            node_prob = 1-node_prob
            node_l = ast.literal_eval(node)   
            # do we have a new optimal solution?
            if u(len(node_l+yc[0]),node_prob+yc[1]) >= ystar_u:
                # we have found a new optimal solution, hence update
                ystar, ystar_u = yc[0]+node_l, u(len(node_l+yc[0]),node_prob+yc[1])
            if c>1:
                ystar, ystar_u = self._pw_proposal_utility((ystar, ystar_u), (yc[0]+node_l, yc[1]+node_prob), copy.deepcopy(Q), u, c-1, verbose) 
            # if current node is not a leaf node -> visit children
            if len(node_l) > 1:
                for ch in self.tree[node]["ch"]:
                    ch_prob = self.tree[ch]["p"]
                    Q.push(ch_prob, ch)
            else:
                # current nodes is a leaf node -> stopping criterion! 
                break   
    
        return ystar, ystar_u
    
    def enumerator(self, c, k):
        visit_list = PriorityList()
        visit_list.push(1,str(list(range(self.K))))
        self.debug = r'''[.{{$\{{{0}\}},{1}$}} '''.format("","\\neg")
        self._enumerator(([], 0), visit_list, c, k)
        self.debug += r''']'''
    
    def _enumerator(self, yc, Q, c, k):
        # run over candidates in Q to add to current combination yc
        while Q.size() != 0:
            node_prob, node = Q.pop()
            node_prob = 1-node_prob
            node_l = ast.literal_eval(node)
            #print("{0} u {1} with representation complexity {2} and probability {3}".format(yc[0],node_l, c, yc[1]+node_prob))
            self.debug += r'''[.{{$\{{{0}\}},{1}$}} '''.format(numl_to_str(node_l).replace("[","").replace("]",""),str(round(yc[1]+node_prob,2)))
            if len(node_l)+len(yc[0]) < k:
                if c>1:
                    #self.debug += r'''[.{{$\{{{0}\}},{1}$}} '''.format(numl_to_str(node_l).replace("[","").replace("]",""),str(round(yc[1]+node_prob,2)))
                    self._enumerator((yc[0]+node_l, yc[1]+node_prob), copy.deepcopy(Q), c-1, k)
            # if current node is not a leaf node -> visit children
            self.debug += r'''] '''
            if len(node_l) > 1:
                for ch in self.tree[node]["ch"]:
                    ch_prob = self.tree[ch]["p"]
                    Q.push(ch_prob, ch)
                
        #self.debug += r''' ] '''
        
    def topk(self, k):
        classes, classes_prob = [], []
        for c in range(self.K):
            classes.append(c)
            classes_prob.append(self.tree["["+str(c)+"]"]["p"])

        return [classes[i] for i in np.argsort(classes_prob)[::-1]][:k]
    
    def pw_ilp_cvxpy(self, c, k):
        A = []
        struct = [ast.literal_eval(n) for n in self.tree]
        A.append(np.array([len(s) for s in struct]))
        # add 1
        A.append(np.ones(len(struct)))
        # add E
        # run over adjecency matric
        for i in range(len(struct)):
            for j in range(i+1,len(struct)):
                if len(set(struct[i])&set(struct[j]))>0:
                    # we have found an edge
                    e = np.zeros(len(struct))
                    e[i] = 1
                    e[j] = 1
                    A.append(e)
        A = np.vstack(A)
        # construct b
        b = np.ones(A.shape[0])
        b[0] = k
        b[1] = c
        start_time = time.time()
        # get p
        p = np.array([self.tree[str(s)]["p"] for s in struct])
        # solve our ILP
        selection = cvxpy.Variable(len(struct), boolean=True)
        constraint = A @ selection <= b
        utility = p @ selection
        knapsack_problem = cvxpy.Problem(cvxpy.Maximize(utility), [constraint])
        knapsack_problem.solve(solver=cvxpy.GLPK_MI, verbose=False)
        sel_ind = list(np.where(selection.value)[0])
        pred = []
        for i in sel_ind:
            pred.extend(struct[i])
        stop_time = time.time()
        print("Executed in {0} s".format(stop_time-start_time))
        
        return set(pred)
    
    def pw_ilp_gurobi(self, c, k):
        A = []
        struct = [ast.literal_eval(n) for n in self.tree]
        A.append(np.array([len(s) for s in struct]))
        # add 1
        A.append(np.ones(len(struct)))
        # add E
        # run over adjecency matric
        for i in range(len(struct)):
            for j in range(i+1,len(struct)):
                if len(set(struct[i])&set(struct[j]))>0:
                    # we have found an edge
                    e = np.zeros(len(struct))
                    e[i] = 1
                    e[j] = 1
                    A.append(e)
        A = np.vstack(A)
        # construct b
        b = np.ones(A.shape[0])
        b[0] = k
        b[1] = c
        start_time = time.time()
        # get p
        p = np.array([self.tree[str(s)]["p"] for s in struct])
        # solve our ILP
        m = gp.Model()
        # decision variables
        x = m.addVars(len(struct), vtype=GRB.BINARY, name='x')
        # set objective 
        m.setObjective(gp.quicksum(p[i] * x[i] for i in range(len(struct))), GRB.MAXIMIZE)
        # add constraints
        for i in range(A.shape[0]):
            m.addConstr((gp.quicksum(A[i,j] * x[j] for j in range(len(struct))) <= b[i]), name="knapsack"+str(i))
        m.update()
        m.optimize()
        stop_time = time.time()
        print(x)
        print("Executed in {0} s".format(stop_time-start_time))
        
        return None

if __name__=="__main__":
    k, c = 5, 1
    n = 10000
    for i in range(n):
        model = GRBOP(256, 10, 100, i, i)
        params = {"framework": "size", "k": k, "c":c}
        pred_prop, _ = model.pw_proposal(params)
        pred_prop = set(pred_prop)
        pred_ilp = model.pw_ilp_cvxpy(c,k)
        if pred_prop != pred_ilp:
            print(i)
            print(pred_prop)
            print(pred_ilp)
            break
