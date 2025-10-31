"""
Week 6-1: Australia Map Coloring - Constraint Satisfaction Problem
Backtracking with MRV, LCV, Forward Checking, and AC-3
"""

from typing import Dict, List, Callable, Set, Tuple, Optional
from copy import deepcopy

Assignment = Dict[str, str]
Domain = Dict[str, List[str]]
Constraint = Callable[[str, str, str, str], bool]

# ============================================================================
# CSP CLASS
# ============================================================================

class CSP:
    def __init__(self, variables: List[str], domains: Domain):
        self.variables = variables
        self.domains = {v: list(domains[v]) for v in variables}
        self.neigh: Dict[str, Set[str]] = {v: set() for v in variables}
        self.binary_constraints: List[Tuple[str, str, Constraint]] = []

    def add_binary_constraint(self, xi: str, xj: str, pred: Constraint):
        self.neigh[xi].add(xj)
        self.neigh[xj].add(xi)
        self.binary_constraints.append((xi, xj, pred))
        self.binary_constraints.append((xj, xi, 
            lambda a, va, b, vb, pred=pred: pred(b, vb, a, va)))

    def neighbors(self, x: str) -> Set[str]:
        return self.neigh[x]

    def consistent_pair(self, xi: str, vi: str, xj: str, vj: str) -> bool:
        for a, b, p in self.binary_constraints:
            if a == xi and b == xj:
                if not p(xi, vi, xj, vj):
                    return False
        return True

# ============================================================================
# HEURISTICS
# ============================================================================

def mrv(assignment: Assignment, csp: CSP) -> str:
    """Minimum Remaining Values with degree tie-breaking."""
    unassigned = [x for x in csp.variables if x not in assignment]
    
    # Count legal values for each unassigned variable
    legal_counts = {}
    for x in unassigned:
        count = 0
        for v in csp.domains[x]:
            if all(csp.consistent_pair(x, v, y, assignment[y]) 
                   for y in csp.neighbors(x) if y in assignment):
                count += 1
        legal_counts[x] = count
    
    # MRV: minimum remaining values
    min_count = min(legal_counts.values())
    candidates = [x for x in unassigned if legal_counts[x] == min_count]
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Degree tie-break: most constraints on unassigned variables
    def degree(x):
        return sum(1 for y in csp.neighbors(x) if y not in assignment)
    
    candidates.sort(key=lambda x: -degree(x))
    return candidates[0]

def lcv(x: str, assignment: Assignment, csp: CSP) -> List[str]:
    """Least Constraining Value ordering."""
    def score(v):
        # Count how many values are ruled out in neighboring variables
        ruled_out = 0
        for y in csp.neighbors(x):
            if y in assignment:
                continue
            for w in csp.domains[y]:
                if not csp.consistent_pair(x, v, y, w):
                    ruled_out += 1
        return ruled_out
    
    return sorted(csp.domains[x], key=score)

# ============================================================================
# FORWARD CHECKING
# ============================================================================

def forward_check(x: str, v: str, assignment: Assignment, csp: CSP):
    """Prune domains of unassigned neighbors. Returns (ok, removed_list)."""
    removed = []
    
    for y in csp.neighbors(x):
        if y in assignment:
            continue
        
        to_remove = [w for w in csp.domains[y] 
                     if not csp.consistent_pair(x, v, y, w)]
        
        if to_remove:
            csp.domains[y] = [w for w in csp.domains[y] if w not in to_remove]
            removed.append((y, to_remove))
            
            if not csp.domains[y]:  # Domain wipeout
                return False, removed
    
    return True, removed

def undo_forward_check(removed, csp: CSP):
    """Restore pruned values."""
    for y, vals in removed:
        cur = set(csp.domains[y])
        for w in vals:
            if w not in cur:
                csp.domains[y].append(w)

# ============================================================================
# AC-3 (Arc Consistency)
# ============================================================================

def ac3(csp: CSP) -> bool:
    """AC-3 algorithm. Returns False if inconsistency detected."""
    from collections import deque
    
    queue = deque()
    for xi in csp.variables:
        for xj in csp.neighbors(xi):
            queue.append((xi, xj))
    
    while queue:
        xi, xj = queue.popleft()
        
        if revise(csp, xi, xj):
            if not csp.domains[xi]:
                return False
            
            for xk in csp.neighbors(xi):
                if xk != xj:
                    queue.append((xk, xi))
    
    return True

def revise(csp: CSP, xi: str, xj: str) -> bool:
    """Remove values from xi's domain that have no support in xj's domain."""
    revised = False
    
    for vi in list(csp.domains[xi]):
        # Check if vi has any supporting value in xj's domain
        has_support = any(csp.consistent_pair(xi, vi, xj, vj) 
                         for vj in csp.domains[xj])
        
        if not has_support:
            csp.domains[xi].remove(vi)
            revised = True
    
    return revised

# ============================================================================
# BACKTRACKING SEARCH
# ============================================================================

def backtracking_search(csp: CSP, use_mrv=True, use_lcv=True, 
                       use_fc=True, use_ac3=False):
    """
    Backtracking search with configurable heuristics.
    Returns (success, assignment, nodes, backtracks, order_log).
    """
    assignment: Assignment = {}
    nodes = 0
    backtracks = 0
    order_log = []
    
    # Optional: Run AC-3 preprocessing
    if use_ac3:
        if not ac3(csp):
            return False, {}, 0, 0, []
    
    def backtrack():
        nonlocal nodes, backtracks
        
        if len(assignment) == len(csp.variables):
            return True
        
        # Variable selection
        if use_mrv:
            x = mrv(assignment, csp)
        else:
            x = next(v for v in csp.variables if v not in assignment)
        
        # Value ordering
        if use_lcv:
            values = lcv(x, assignment, csp)
        else:
            values = list(csp.domains[x])
        
        for v in values:
            nodes += 1
            
            # Check consistency with assigned neighbors
            if not all(csp.consistent_pair(x, v, y, assignment[y]) 
                      for y in csp.neighbors(x) if y in assignment):
                continue
            
            # Make assignment
            assignment[x] = v
            order_log.append((x, v))
            
            # Forward checking
            removed = []
            ok = True
            if use_fc:
                ok, removed = forward_check(x, v, assignment, csp)
            
            if ok:
                if backtrack():
                    return True
            
            # Undo
            if use_fc:
                undo_forward_check(removed, csp)
            order_log.pop()
            assignment.pop(x, None)
        
        backtracks += 1
        return False
    
    success = backtrack()
    return success, assignment, nodes, backtracks, order_log

# ============================================================================
# PROBLEM INSTANCES
# ============================================================================

def australia_csp():
    """Australia map coloring problem."""
    vars = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]
    dom = {v: ["R", "G", "B"] for v in vars}
    csp = CSP(vars, dom)
    
    edges = [
        ("WA", "NT"), ("WA", "SA"), ("NT", "SA"), ("NT", "Q"),
        ("SA", "Q"), ("SA", "NSW"), ("SA", "V"), ("Q", "NSW"), ("NSW", "V")
    ]
    
    ne = lambda xi, vi, xj, vj: vi != vj
    for a, b in edges:
        csp.add_binary_constraint(a, b, ne)
    
    return csp

def n_queens(n: int = 8):
    """N-Queens problem as CSP."""
    vars = [f"Q{i}" for i in range(n)]
    dom = {v: list(range(n)) for v in vars}
    csp = CSP(vars, dom)
    
    def no_attack(xi, vi, xj, vj):
        # vi, vj are row positions; column positions are i, j
        i = int(xi[1:])
        j = int(xj[1:])
        # Different rows
        if vi == vj:
            return False
        # Not on same diagonal
        if abs(vi - vj) == abs(i - j):
            return False
        return True
    
    for i in range(n):
        for j in range(i + 1, n):
            csp.add_binary_constraint(f"Q{i}", f"Q{j}", no_attack)
    
    return csp

# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_compare_techniques():
    """Compare different backtracking configurations."""
    print("=" * 80)
    print("EXPERIMENT 1: Algorithm Comparison on Australia Map")
    print("=" * 80)
    
    configs = [
        ("Plain Backtracking", False, False, False, False),
        ("+ MRV", True, False, False, False),
        ("+ MRV + LCV", True, True, False, False),
        ("+ MRV + LCV + FC", True, True, True, False),
        ("+ MRV + LCV + FC + AC-3", True, True, True, True),
    ]
    
    print(f"\n{'Configuration':<30s} {'Nodes':>8s} {'Backtracks':>12s} {'Success':>8s}")
    print("-" * 80)
    
    for name, mrv, lcv, fc, ac3 in configs:
        csp = australia_csp()
        success, sol, nodes, backs, order = backtracking_search(
            csp, use_mrv=mrv, use_lcv=lcv, use_fc=fc, use_ac3=ac3
        )
        print(f"{name:<30s} {nodes:8d} {backs:12d} {str(success):>8s}")
    
    # Show detailed solution for best configuration
    print("\n" + "=" * 80)
    print("Best Configuration Solution (MRV + LCV + FC):")
    print("=" * 80)
    
    csp = australia_csp()
    success, sol, nodes, backs, order = backtracking_search(
        csp, use_mrv=True, use_lcv=True, use_fc=True, use_ac3=False
    )
    
    print(f"\nSolution: {sol}")
    print(f"\nAssignment Order:")
    for i, (var, val) in enumerate(order, 1):
        print(f"  {i}. {var} = {val}")
    
    print(f"\nStatistics:")
    print(f"  Nodes expanded: {nodes}")
    print(f"  Backtracks: {backs}")
    print(f"  Solution valid: {verify_solution(australia_csp(), sol)}")

def experiment_n_queens():
    """Compare techniques on N-Queens problem."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Algorithm Comparison on 8-Queens")
    print("=" * 80)
    
    configs = [
        ("Plain Backtracking", False, False, False),
        ("+ MRV + LCV", True, True, False),
        ("+ MRV + LCV + FC", True, True, True),
    ]
    
    print(f"\n{'Configuration':<30s} {'Nodes':>8s} {'Backtracks':>12s} {'Time(ms)':>10s}")
    print("-" * 80)
    
    import time
    
    for name, mrv, lcv, fc in configs:
        csp = n_queens(8)
        start = time.time()
        success, sol, nodes, backs, order = backtracking_search(
            csp, use_mrv=mrv, use_lcv=lcv, use_fc=fc
        )
        elapsed = (time.time() - start) * 1000
        print(f"{name:<30s} {nodes:8d} {backs:12d} {elapsed:10.2f}")
    
    # Show solution
    print("\n8-Queens Solution (MRV + LCV + FC):")
    csp = n_queens(8)
    success, sol, nodes, backs, order = backtracking_search(csp, True, True, True)
    
    # Visualize board
    board = [['.' for _ in range(8)] for _ in range(8)]
    for var, row in sol.items():
        col = int(var[1:])
        board[row][col] = 'Q'
    
    print()
    for row in board:
        print("  " + " ".join(row))

def verify_solution(csp: CSP, assignment: Assignment) -> bool:
    """Verify that solution satisfies all constraints."""
    for xi in csp.variables:
        if xi not in assignment:
            return False
        for xj in csp.neighbors(xi):
            if xj in assignment:
                if not csp.consistent_pair(xi, assignment[xi], xj, assignment[xj]):
                    return False
    return True

def experiment_scaling():
    """Test scaling with different N values for N-Queens."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Scaling Analysis (N-Queens)")
    print("=" * 80)
    
    print(f"\n{'N':>3s} {'Nodes':>10s} {'Backtracks':>12s} {'Time(ms)':>10s}")
    print("-" * 80)
    
    import time
    
    for n in [4, 6, 8, 10]:
        csp = n_queens(n)
        start = time.time()
        success, sol, nodes, backs, order = backtracking_search(
            csp, use_mrv=True, use_lcv=True, use_fc=True
        )
        elapsed = (time.time() - start) * 1000
        print(f"{n:3d} {nodes:10d} {backs:12d} {elapsed:10.2f}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("WEEK 6-1: CONSTRAINT SATISFACTION PROBLEMS")
    print("=" * 80)
    
    experiment_compare_techniques()
    experiment_n_queens()
    experiment_scaling()
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED")
    print("=" * 80)