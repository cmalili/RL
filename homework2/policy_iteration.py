#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 20:42:27 2025

@author: cmalili
"""

import numpy as np

lam = 0.95
epsilon = 0.01

print(" States: 0 for s1, 1 for s2.")
print(" We represent the policy for s1 as an index: 0 for a_{1,1} and 1 for a_{1,2}.")
print(" For s2, there is only one action.")
print("")
# States: 0 for s1, 1 for s2.
# We represent the policy for s1 as an index: 0 for a_{1,1} and 1 for a_{1,2}.
# For s2, there is only one action.
policy = {0: 0, 1: 0}  # Start with choosing a_{1,1} for s1

def q_s1_a11(V):
    return 5 + lam * (0.5 * V[0] + 0.5 * V[1])

def q_s1_a12(V):
    return 10 + lam * V[1]

def q_s2(V):
    return -1 + lam * V[1]

def policy_evaluation(policy, V_init, tol=epsilon):
    V = V_init.copy()
    while True:
        V_new = np.zeros_like(V)
        # For s2 (only one action)
        V_new[1] = q_s2(V)
        # For s1, follow the current policy
        if policy[0] == 0:
            V_new[0] = q_s1_a11(V)
        else:
            V_new[0] = q_s1_a12(V)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V

# Initialize value function arbitrarily
V = np.array([0.0, 0.0])

stable = False
iteration = 0
while not stable:
    iteration += 1
    V = policy_evaluation(policy, V)
    
    # Policy improvement for s1
    Q1 = q_s1_a11(V)
    Q2 = q_s1_a12(V)
    new_action = 0 if Q1 >= Q2 else 1
    policy_stable = (new_action == policy[0])
    policy[0] = new_action  # update policy for s1
    
    print(f"Policy Iteration {iteration}: V = {V}, policy = {policy}")
    if policy_stable:
        stable = True

print("Optimal policy found:", policy)
print("Optimal Value Function from Policy Iteration:", V)
