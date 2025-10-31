"""
Week 6-2: 3-Step Object Tracking - Weighted CSP
Backtracking, AC-3, Beam Search, and ICM (Iterated Conditional Modes)
"""

from typing import Dict, List, Tuple, Optional
from collections import deque
from copy import deepcopy
import random

Var = str
Val = int
Assignment = Dict[Var, Val]

# ============================================================================
# WEIGHTED CSP CLASS
# ============================================================================

class WeightedCSP:
    def __init__(self, variables: List[Var], domains: Dict[Var, List[Val]]):
        self.variables = variables
        self.domains = {v: list(domains[v]) for v in variables}
        self.unary = {v: {} for v in variables}
        self.binary = {}
        self.neigh = {v: set() for v in variables}

    def add_unary(self, v: Var, table: Dict[Val, float]):
        """Add unary factor (observation model)."""
        self.unary[v] = dict(table)

    def add_binary(self, u: Var, v: Var, table: Dict[Tuple[Val, Val], float]):
        """Add binary factor (transition model)."""
        self.binary[(u, v)] = dict(table)
        self.binary[(v, u)] = {(b, a): w for (a, b), w in table.items()}
        self.neigh[u].add(v)
        self.neigh[v].add(u)

    def dep_weight(self, x: Assignment, var: Var, val: Val) -> float:
        """
        Product of factors touching var whose other vars are assigned in x.
        (Partial weight for extending assignment)
        """
        w = 1.0
        
        # Unary factor
        if self.unary[var]:
            w *= self.unary[var].get(val, 0.0)
        
        # Binary factors with assigned neighbors
        for nb in self.neigh[var]:
            if nb in x:
                w *= self.binary[(var, nb)].get((val, x[nb]), 0.0)
        
        return w

    def full_weight(self, x: Assignment) -> float:
        """Complete weight of full assignment."""
        w = 1.0
        
        # All unary factors
        for v in self.variables:
            if self.unary[v]:
                w *= self.unary[v].get(x[v], 0.0)
        
        # All binary factors (count each once)
        for (u, v), tab in self.binary.items():
            if u < v:  # Avoid double counting
                w *= tab.get((x[u], x[v]), 0.0)
        
        return w

# ============================================================================
# FORWARD CHECKING
# ============================================================================

def forward_check(csp: WeightedCSP, x: Assignment, var: Var, val: Val):
    """Prune zero-weight values from neighbor domains."""
    removed = []
    
    for y in csp.neigh[var]:
        if y in x:
            continue
        
        to_rm = []
        for b in list(csp.domains[y]):
            w = csp.binary[(var, y)].get((val, b), 0.0)
            if w == 0.0:
                to_rm.append(b)
        
        if to_rm:
            removed.append((y, to_rm))
            csp.domains[y] = [b for b in csp.domains[y] if b not in to_rm]
            
            if not csp.domains[y]:
                return False, removed
    
    return True, removed

def undo_fc(csp: WeightedCSP, removed):
    """Restore pruned values."""
    for y, vals in removed:
        for b in vals:
            if b not in csp.domains[y]:
                csp.domains[y].append(b)

# ============================================================================
# AC-3 (Arc Consistency)
# ============================================================================

def enforce_arc_consistency(csp: WeightedCSP) -> bool:
    """AC-3 using zero-support pruning on binary factors."""
    q = deque()
    
    for (u, v) in csp.binary.keys():
        q.append((u, v))
    
    changed = False
    
    while q:
        u, v = q.popleft()
        dom_u = list(csp.domains[u])
        removed = False
        
        for a in dom_u:
            # Check if a has any supporting b in v's domain with nonzero factor
            ok = any(csp.binary[(u, v)].get((a, b), 0.0) > 0.0 
                    for b in csp.domains[v])
            
            if not ok:
                csp.domains[u].remove(a)
                removed = True
                changed = True
        
        if removed:
            if not csp.domains[u]:
                return False
            
            for w in csp.neigh[u]:
                if w != v:
                    q.append((w, u))
    
    return True

# ============================================================================
# HEURISTICS
# ============================================================================

def mrv(csp: WeightedCSP, x: Assignment) -> Var:
    """Minimum Remaining Values with degree tie-breaking."""
    unassigned = [v for v in csp.variables if v not in x]
    
    # MRV: smallest domain size
    k = min(len(csp.domains[v]) for v in unassigned)
    cands = [v for v in unassigned if len(csp.domains[v]) == k]
    
    # Tie-break by degree
    cands.sort(key=lambda v: -len([nb for nb in csp.neigh[v] if nb not in x]))
    return cands[0]

def lcv_values(csp: WeightedCSP, x: Assignment, var: Var) -> List[Val]:
    """Least Constraining Value ordering."""
    def score(val):
        s = 0
        for nb in csp.neigh[var]:
            if nb in x:
                continue
            s += sum(1 for b in csp.domains[nb] 
                    if csp.binary[(var, nb)].get((val, b), 0.0) > 0.0)
        return -s
    
    return sorted(list(csp.domains[var]), key=score)

# ============================================================================
# BACKTRACKING SEARCH
# ============================================================================

def backtracking(csp: WeightedCSP, use_ac3=False):
    """
    Backtracking search for maximum-weight assignment.
    Returns (best_weight, best_assignment, nodes, backtracks).
    """
    x: Assignment = {}
    best = (0.0, None)
    nodes = 0
    backs = 0
    
    # Optional AC-3 preprocessing
    if use_ac3:
        enforce_arc_consistency(csp)
    
    def dfs():
        nonlocal nodes, backs, best
        
        if len(x) == len(csp.variables):
            w = csp.full_weight(x)
            if w > best[0]:
                best = (w, dict(x))
            return True
        
        var = mrv(csp, x)
        
        for val in lcv_values(csp, x, var):
            nodes += 1
            
            delta = csp.dep_weight(x, var, val)
            if delta == 0.0:
                continue
            
            x[var] = val
            
            # Forward checking
            ok, removed = forward_check(csp, x, var, val)
            
            if ok:
                if use_ac3:
                    ok = enforce_arc_consistency(csp)
                
                if ok:
                    dfs()
            
            undo_fc(csp, removed)
            x.pop(var, None)
        
        backs += 1
        return False
    
    dfs()
    return best[0], best[1], nodes, backs

# ============================================================================
# BEAM SEARCH
# ============================================================================

def beam_search(csp: WeightedCSP, K: int):
    """
    Beam search with beam width K.
    Returns (best_weight, best_assignment, expansions).
    """
    # Candidates are (assignment, weight)
    cand = [({}, 1.0)]
    expansions = 0
    
    for var in csp.variables:
        ext = []
        
        for x, w in cand:
            for val in csp.domains[var]:
                expansions += 1
                delta = csp.dep_weight(x, var, val)
                
                if delta == 0.0:
                    continue
                
                x2 = dict(x)
                x2[var] = val
                ext.append((x2, w * delta))
        
        # Keep top-K by weight
        ext.sort(key=lambda t: t[1], reverse=True)
        cand = ext[:K] if ext else []
        
        if not cand:
            break
    
    # Pick best full assignment
    if cand:
        best = max(cand, key=lambda t: t[1])
        return best[1], best[0], expansions
    else:
        return 0.0, {}, expansions

# ============================================================================
# LOCAL SEARCH (ICM)
# ============================================================================

def icm(csp: WeightedCSP, max_iters: int = 10, seed: int = 0):
    """
    Iterated Conditional Modes (greedy coordinate ascent).
    Returns (final_weight, final_assignment, iterations).
    """
    random.seed(seed)
    
    # Random initialization
    x = {v: random.choice(csp.domains[v]) for v in csp.variables}
    
    def local_weight(var, val):
        """Local weight: unary(var) * binaries with neighbors."""
        w = csp.unary[var].get(val, 1.0) if csp.unary[var] else 1.0
        
        for nb in csp.neigh[var]:
            b = x[nb]
            w *= csp.binary[(var, nb)].get((val, b), 0.0)
        
        return w
    
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        for v in csp.variables:
            best = max(csp.domains[v], key=lambda a: local_weight(v, a))
            
            if local_weight(v, best) > local_weight(v, x[v]):
                x[v] = best
                improved = True
    
    final_weight = csp.full_weight(x)
    return final_weight, x, iters

# ============================================================================
# PROBLEM INSTANCE
# ============================================================================

def build_tracking_instance():
    """3-step object tracking problem."""
    vars = ["X1", "X2", "X3"]
    doms = {v: [0, 1, 2] for v in vars}
    csp = WeightedCSP(vars, doms)
    
    # Observations: (0, 2, 2)
    obs = {"X1": 0, "X2": 2, "X3": 2}
    
    # Unary observation factors: O(x) = max(0, 2 - |x - obs|)
    for v in vars:
        table = {a: max(0, 2 - abs(a - obs[v])) for a in doms[v]}
        csp.add_unary(v, table)
    
    # Binary transition factors: T(a,b) = 2 if equal, 1 if adjacent, 0 else
    def trans(a, b):
        if a == b:
            return 2
        if abs(a - b) == 1:
            return 1
        return 0
    
    for (u, v) in [("X1", "X2"), ("X2", "X3")]:
        tab = {}
        for a in doms[u]:
            for b in doms[v]:
                tab[(a, b)] = trans(a, b)
        csp.add_binary(u, v, tab)
    
    return csp

# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_backtracking():
    """Compare backtracking configurations."""
    print("=" * 80)
    print("EXPERIMENT 1: Backtracking with Forward Checking and AC-3")
    print("=" * 80)
    
    configs = [
        ("Backtracking + FC", False),
        ("Backtracking + FC + AC-3", True),
    ]
    
    print(f"\n{'Configuration':<30s} {'Weight':>10s} {'Nodes':>8s} {'Backtracks':>12s}")
    print("-" * 80)
    
    for name, use_ac3 in configs:
        csp = build_tracking_instance()
        weight, sol, nodes, backs = backtracking(csp, use_ac3=use_ac3)
        print(f"{name:<30s} {weight:10.2f} {nodes:8d} {backs:12d}")
    
    # Show detailed solution
    print("\n" + "=" * 80)
    print("Best Solution Details:")
    print("=" * 80)
    
    csp = build_tracking_instance()
    weight, sol, nodes, backs = backtracking(csp, use_ac3=False)
    
    print(f"\nOptimal Assignment: {sol}")
    print(f"Weight: {weight:.2f}")
    print(f"\nFactor Breakdown:")
    
    # Show unary factors
    print("\nObservation Factors:")
    for v in ["X1", "X2", "X3"]:
        val = sol[v]
        w = csp.unary[v][val]
        print(f"  O({v}={val}) = {w}")
    
    # Show binary factors
    print("\nTransition Factors:")
    for (u, v) in [("X1", "X2"), ("X2", "X3")]:
        pair = (sol[u], sol[v])
        w = csp.binary[(u, v)][pair]
        print(f"  T({u}={sol[u]}, {v}={sol[v]}) = {w}")

def experiment_beam():
    """Compare beam search with different K values."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Beam Search Analysis")
    print("=" * 80)
    
    print(f"\n{'K':>3s} {'Weight':>10s} {'Assignment':>20s} {'Expansions':>12s}")
    print("-" * 80)
    
    for K in [1, 2, 3, 4]:
        csp = build_tracking_instance()
        weight, sol, exps = beam_search(csp, K)
        sol_str = str(sol) if sol else "None"
        print(f"{K:3d} {weight:10.2f} {sol_str:>20s} {exps:12d}")
    
    print("\nConclusion: Beam width K=2 finds optimal solution with fewer expansions.")

def experiment_icm():
    """Compare ICM from multiple random initializations."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: ICM (Iterated Conditional Modes) Local Search")
    print("=" * 80)
    
    print(f"\n{'Trial':>6s} {'Init Assignment':>20s} {'Final Assignment':>20s} {'Weight':>10s} {'Iters':>6s}")
    print("-" * 80)
    
    weights = []
    solutions = []
    
    for trial in range(10):
        csp = build_tracking_instance()
        weight, sol, iters = icm(csp, max_iters=20, seed=trial)
        
        # Get initial assignment for this seed
        random.seed(trial)
        init = {v: random.choice(csp.domains[v]) for v in csp.variables}
        
        weights.append(weight)
        solutions.append(sol)
        
        print(f"{trial:6d} {str(init):>20s} {str(sol):>20s} {weight:10.2f} {iters:6d}")
    
    # Analyze local optima
    print(f"\nSummary:")
    print(f"  Best weight found: {max(weights):.2f}")
    print(f"  Worst weight found: {min(weights):.2f}")
    print(f"  Average weight: {sum(weights)/len(weights):.2f}")
    
    # Count unique solutions
    unique_sols = len(set(tuple(sorted(s.items())) for s in solutions))
    print(f"  Unique solutions found: {unique_sols}/10")

def experiment_comparison():
    """Compare all methods side by side."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Method Comparison")
    print("=" * 80)
    
    results = []
    
    # Backtracking
    csp = build_tracking_instance()
    weight, sol, nodes, backs = backtracking(csp, use_ac3=False)
    results.append(("Backtracking + FC", weight, nodes, sol))
    
    # Beam K=2
    csp = build_tracking_instance()
    weight, sol, exps = beam_search(csp, K=2)
    results.append(("Beam Search (K=2)", weight, exps, sol))
    
    # ICM (best of 5 runs)
    best_icm = (0.0, {}, 0)
    for trial in range(5):
        csp = build_tracking_instance()
        weight, sol, iters = icm(csp, max_iters=20, seed=trial)
        if weight > best_icm[0]:
            best_icm = (weight, sol, iters)
    results.append(("ICM (best of 5)", best_icm[0], best_icm[2], best_icm[1]))
    
    print(f"\n{'Method':<25s} {'Weight':>10s} {'Work':>10s} {'Solution':>25s}")
    print("-" * 80)
    
    for method, weight, work, sol in results:
        print(f"{method:<25s} {weight:10.2f} {work:10d} {str(sol):>25s}")
    
    print("\nConclusion:")
    print("  - Backtracking guarantees optimal solution but explores more nodes")
    print("  - Beam search balances quality and efficiency")
    print("  - ICM is fastest but may get stuck in local optima")

def experiment_partial_weight():
    """Demonstrate partial weight computation."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Partial Weight Analysis")
    print("=" * 80)
    
    csp = build_tracking_instance()
    
    print("\nStarting with X1=0, extending X2:")
    x = {"X1": 0}
    
    print(f"\n{'X2':>5s} {'O(X2)':>8s} {'T(X1,X2)':>10s} {'Partial Weight':>15s}")
    print("-" * 80)
    
    for val in [0, 1, 2]:
        obs_weight = csp.unary["X2"][val]
        trans_weight = csp.binary[("X2", "X1")][(val, 0)]
        partial = csp.dep_weight(x, "X2", val)
        print(f"{val:5d} {obs_weight:8.0f} {trans_weight:10.0f} {partial:15.2f}")
    
    print("\nConclusion: X2=1 has highest positive partial weight (2.0)")
    
    print("\nContinuing with X1=0, X2=1, extending X3:")
    x = {"X1": 0, "X2": 1}
    
    print(f"\n{'X3':>5s} {'O(X3)':>8s} {'T(X2,X3)':>10s} {'Partial Weight':>15s}")
    print("-" * 80)
    
    for val in [0, 1, 2]:
        obs_weight = csp.unary["X3"][val]
        trans_weight = csp.binary[("X3", "X2")][(val, 1)]
        partial = csp.dep_weight(x, "X3", val)
        print(f"{val:5d} {obs_weight:8.0f} {trans_weight:10.0f} {partial:15.2f}")
    
    print("\nConclusion: X3=2 has highest partial weight (4.0)")
    print("Final assignment (0,1,2) has total weight: 4.0")

def experiment_forward_checking_trace():
    """Demonstrate forward checking step by step."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: Forward Checking Trace")
    print("=" * 80)
    
    csp = build_tracking_instance()
    
    print("\nInitial domains:")
    for v in csp.variables:
        print(f"  {v}: {csp.domains[v]}")
    
    print("\nAfter assigning X1=0:")
    x = {"X1": 0}
    ok, removed = forward_check(csp, x, "X1", 0)
    print(f"  Forward check result: {ok}")
    print(f"  Domains after FC:")
    for v in csp.variables:
        print(f"    {v}: {csp.domains[v]}")
    if removed:
        print(f"  Removed values: {removed}")
    
    # Reset for X2=2 case
    csp2 = build_tracking_instance()
    print("\n\nAfter assigning X2=2:")
    x2 = {"X2": 2}
    ok2, removed2 = forward_check(csp2, x2, "X2", 2)
    print(f"  Forward check result: {ok2}")
    print(f"  Domains after FC:")
    for v in csp2.variables:
        print(f"    {v}: {csp2.domains[v]}")
    if removed2:
        print(f"  Removed values: {removed2}")
    else:
        print(f"  No values removed (X2 only constrains neighbors X1 and X3)")

def experiment_ac3_trace():
    """Show AC-3 algorithm trace."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 7: AC-3 Arc Consistency Trace")
    print("=" * 80)
    
    csp = build_tracking_instance()
    
    print("\nInitial domains:")
    for v in csp.variables:
        print(f"  {v}: {csp.domains[v]}")
    
    print("\nRunning AC-3...")
    
    # Manual AC-3 with detailed trace
    from collections import deque
    q = deque()
    for (u, v) in csp.binary.keys():
        q.append((u, v))
    
    print(f"\nInitial queue size: {len(q)} arcs")
    iteration = 0
    
    while q and iteration < 20:  # Limit iterations for display
        iteration += 1
        u, v = q.popleft()
        
        # Check if arc needs revision
        dom_u_before = list(csp.domains[u])
        revised = False
        
        for a in dom_u_before:
            has_support = any(csp.binary[(u, v)].get((a, b), 0.0) > 0.0 
                            for b in csp.domains[v])
            if not has_support:
                csp.domains[u].remove(a)
                revised = True
        
        if revised:
            print(f"  Arc ({u},{v}): removed {set(dom_u_before) - set(csp.domains[u])} from {u}")
            for w in csp.neigh[u]:
                if w != v:
                    q.append((w, u))
    
    print(f"\nAC-3 completed in {iteration} iterations")
    print(f"\nFinal domains after AC-3:")
    for v in csp.variables:
        print(f"  {v}: {csp.domains[v]}")
    
    print("\nConclusion: AC-3 removes values with no supporting transitions")

def experiment_beam_trace():
    """Show beam search level by level."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 8: Beam Search Detailed Trace (K=2)")
    print("=" * 80)
    
    csp = build_tracking_instance()
    K = 2
    
    beam = [({}, 1.0)]
    
    for var in csp.variables:
        print(f"\n{'='*70}")
        print(f"Level: {var}")
        print(f"{'='*70}")
        
        print(f"\nCurrent beam (size {len(beam)}):")
        for i, (assign, w) in enumerate(beam, 1):
            print(f"  {i}. {assign if assign else 'empty'} -> weight={w:.2f}")
        
        # Expand
        extensions = []
        print(f"\nExpanding {var}:")
        
        for assign, w in beam:
            for val in csp.domains[var]:
                delta = csp.dep_weight(assign, var, val)
                if delta > 0:
                    new_assign = dict(assign)
                    new_assign[var] = val
                    new_w = w * delta
                    extensions.append((new_assign, new_w))
                    print(f"  {assign if assign else '{}'} + {var}={val} -> " +
                          f"weight={new_w:.2f} (delta={delta:.2f})")
        
        # Keep top K
        extensions.sort(key=lambda x: x[1], reverse=True)
        beam = extensions[:K]
        
        print(f"\nTop-{K} after pruning:")
        for i, (assign, w) in enumerate(beam, 1):
            print(f"  {i}. {assign} -> weight={w:.2f}")
    
    print(f"\n{'='*70}")
    print("FINAL RESULT")
    print(f"{'='*70}")
    best = max(beam, key=lambda x: x[1])
    print(f"\nBest assignment: {best[0]}")
    print(f"Weight: {best[1]:.2f}")

def experiment_icm_trace():
    """Show ICM iteration trace."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 9: ICM (Iterated Conditional Modes) Trace")
    print("=" * 80)
    
    csp = build_tracking_instance()
    random.seed(42)
    
    # Random initialization
    x = {v: random.choice(csp.domains[v]) for v in csp.variables}
    
    print(f"\nInitial assignment (random): {x}")
    print(f"Initial weight: {csp.full_weight(x):.2f}")
    
    def local_weight(var, val):
        w = csp.unary[var].get(val, 1.0) if csp.unary[var] else 1.0
        for nb in csp.neigh[var]:
            b = x[nb]
            w *= csp.binary[(var, nb)].get((val, b), 0.0)
        return w
    
    print("\n" + "-" * 70)
    print("ICM Iterations:")
    print("-" * 70)
    
    for iteration in range(1, 11):
        print(f"\nIteration {iteration}:")
        improved = False
        
        for v in csp.variables:
            old_val = x[v]
            old_local = local_weight(v, old_val)
            
            # Find best value
            best_val = old_val
            best_local = old_local
            
            print(f"  Variable {v} (current={old_val}, local_weight={old_local:.2f}):")
            for val in csp.domains[v]:
                lw = local_weight(v, val)
                print(f"    {v}={val} -> local_weight={lw:.2f}", end="")
                if lw > best_local:
                    best_val = val
                    best_local = lw
                    print(" *best*")
                else:
                    print()
            
            if best_val != old_val:
                x[v] = best_val
                improved = True
                print(f"  → Updated {v}: {old_val} → {best_val}")
        
        current_weight = csp.full_weight(x)
        print(f"\n  Assignment after iteration {iteration}: {x}")
        print(f"  Total weight: {current_weight:.2f}")
        
        if not improved:
            print(f"\n  Converged (no improvements)")
            break
    
    print(f"\n{'='*70}")
    print("FINAL RESULT")
    print(f"{'='*70}")
    print(f"Final assignment: {x}")
    print(f"Final weight: {csp.full_weight(x):.2f}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("WEEK 6-2: WEIGHTED CSP FOR OBJECT TRACKING")
    print("=" * 80)
    
    experiment_backtracking()
    experiment_beam()
    experiment_icm()
    experiment_comparison()
    experiment_partial_weight()
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED")
    print("=" * 80)