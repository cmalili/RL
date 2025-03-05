#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 20:14:09 2025

@author: cmalili
"""

import numpy as np

# Parameters
lam = 0.95
epsilon = 0.01

# State indices: 0 corresponds to s1, 1 corresponds to s2.
V = np.array([0.0, 0.0])  # Initial value estimates

def Q_s1_a11(V):
    # Action a_{1,1} in s1: reward=5, transitions: 0.5 to s1, 0.5 to s2.
    return 5 + lam * (0.5 * V[0] + 0.5 * V[1])

def Q_s1_a12(V):
    # Action a_{1,2} in s1: reward=10, transitions: 1 to s2.
    return 10 + lam * V[1]

def Q_s2(V):
    # In s2, only one action: reward=-1, remains in s2.
    return -1 + lam * V[1]

iteration = 0
while True:
    V_new = np.zeros_like(V)
    # Update for s1:
    Q1 = Q_s1_a11(V)
    Q2 = Q_s1_a12(V)
    V_new[0] = max(Q1, Q2)
    # Update for s2:
    V_new[1] = Q_s2(V)
    
    diff = np.max(np.abs(V_new - V))
    V = V_new
    iteration += 1
    print(f"Iteration {iteration}: V = {V}, diff = {diff:.4f}")
    
    if diff < epsilon:
        break

print("Optimal Value Function:", V)
