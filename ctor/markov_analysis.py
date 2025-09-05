# markov_ergodicity.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import math

@dataclass
class MarkovAnalysis:
    n_states: int
    irreducible: bool
    period: Optional[int]
    aperiodic: bool
    ergodic: bool
    stationary_distribution: np.ndarray
    sccs: List[List[int]]
    primitive_k: Optional[int]

def _validate_row_stochastic(P: np.ndarray, tol: float) -> None:
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix.")
    if (P < -tol).any():
        raise ValueError("P must have nonnegative entries (within tolerance).")
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        raise ValueError("Each row of P must sum to 1 (within tolerance).")

def _adjacency(P: np.ndarray, tol: float) -> Tuple[List[List[int]], List[List[int]]]:
    n = P.shape[0]
    adj = [[] for _ in range(n)]
    radj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if P[i, j] > tol:
                adj[i].append(j)
                radj[j].append(i)
    return adj, radj

def _kosaraju_scc(adj: List[List[int]], radj: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
    n = len(adj)
    visited = [False] * n
    order: List[int] = []

    def dfs1(u: int):
        stack = [u]
        while stack:
            v = stack.pop()
            if v < 0:
                order.append(~v)
                continue
            if visited[v]:
                continue
            visited[v] = True
            stack.append(~v)
            for w in adj[v]:
                if not visited[w]:
                    stack.append(w)

    for v in range(n):
        if not visited[v]:
            dfs1(v)

    comp = [-1] * n
    cid = 0
    for v in reversed(order):
        if comp[v] != -1:
            continue
        stack = [v]
        comp[v] = cid
        while stack:
            x = stack.pop()
            for w in radj[x]:
                if comp[w] == -1:
                    comp[w] = cid
                    stack.append(w)
        cid += 1

    sccs: List[List[int]] = [[] for _ in range(cid)]
    for v in range(n):
        sccs[comp[v]].append(v)
    return sccs, comp

def _is_irreducible(P: np.ndarray, tol: float) -> Tuple[bool, List[List[int]], List[int], List[List[int]]]:
    adj, radj = _adjacency(P, tol)
    sccs, comp = _kosaraju_scc(adj, radj)
    return (len(sccs) == 1), sccs, comp, adj

def _period_of_irreducible(adj: List[List[int]]) -> int:
    n = len(adj)
    dist = [-1] * n
    g = 0
    dist[0] = 0
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                stack.append(v)
            else:
                g = math.gcd(g, abs(dist[u] + 1 - dist[v]))
    return max(1, g)

def _stationary_distribution(P: np.ndarray, tol: float) -> np.ndarray:
    n = P.shape[0]
    A = np.eye(n) - P.T
    A[-1, :] = 1.0
    b = np.zeros(n); b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    if s <= tol:
        w, v = np.linalg.eig(P.T)
        i = int(np.argmin(np.abs(w - 1.0)))
        pi = np.real(v[:, i])
        pi = np.maximum(pi, 0.0)
        s = pi.sum()
        if s <= tol:
            raise ValueError("Could not compute a valid stationary distribution.")
    return pi / pi.sum()

def _primitive_k(P: np.ndarray, tol: float, max_k: int) -> Optional[int]:
    n = P.shape[0]
    M = P.copy()
    for k in range(1, max_k + 1):
        if (M > tol).all():
            return k
        M = M @ P
    return None

def analyze_markov_chain(
    P: np.ndarray,
    tol: float = 1e-12,
    compute_primitive: bool = False,
    max_k: int = 256,
) -> MarkovAnalysis:
    P = np.asarray(P, dtype=float)
    _validate_row_stochastic(P, tol)
    irreducible, sccs, comp, adj = _is_irreducible(P, tol)
    if irreducible:
        period = _period_of_irreducible(adj)
        aperiodic = (period == 1)
        ergodic = aperiodic
    else:
        period = None
        aperiodic = False
        ergodic = False
    pi = _stationary_distribution(P, tol)
    prim_k = _primitive_k(P, tol, max_k) if compute_primitive else None
    return MarkovAnalysis(
        n_states=P.shape[0],
        irreducible=irreducible,
        period=period,
        aperiodic=aperiodic,
        ergodic=ergodic,
        stationary_distribution=pi,
        sccs=sccs,
        primitive_k=prim_k,
    )
