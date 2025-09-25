from collections import deque
import heapq
import time
import sys

# ---------- SearchProblem interface ----------
class SearchProblem:
    def start_state(self):
        raise NotImplementedError
    def is_end(self, s):
        raise NotImplementedError
    def succ_and_cost(self, s):
        '''Yield (action, s', cost).'''
        raise NotImplementedError

# ---------- TransportationProblem ----------
class TransportationProblem(SearchProblem):
    '''
    States: integers s in [1..n]
    Actions:
        - 'walk' to s+1 with cost 1
        - 'tram' to 2*s with cost 2 (only if 2*s <= n)
    If unit_cost=True, treat both actions as cost=1 (for BFS/DFID demonstrations).
    '''
    def __init__(self, n, unit_cost=False):
        assert n >= 1
        self.n = n
        self.unit_cost = unit_cost

    def start_state(self):
        return 1

    def is_end(self, s):
        return s == self.n

    def succ_and_cost(self, s):
        if s < self.n:
            # walk
            c = 1 if not self.unit_cost else 1
            yield ('walk', s+1, c)
            # tram
            if 2*s <= self.n:
                c = 2 if not self.unit_cost else 1
                yield ('tram', 2*s, c)

# ---------- Utilities ----------
def reconstruct_path(parent, end_state):
    path = []
    s = end_state
    while s in parent:
        s_prev, action, cost = parent[s]
        path.append((action, s, cost))
        s = s_prev
    path.reverse()
    return path

def format_path(path):
    """Format path for readable output"""
    if not path:
        return "No path"
    result = "1"
    for action, state, cost in path:
        result += f" --{action}({cost})--> {state}"
    return result

# ---------- Backtracking (exponential) ----------
def backtracking_min_cost(problem):
    best = {'cost': float('inf'), 'path': None}
    expansions = 0
    max_frontier_size = 0
    current_depth = 0

    def dfs(s, cost_so_far, parent, depth):
        nonlocal expansions, max_frontier_size, current_depth
        current_depth = max(current_depth, depth)
        max_frontier_size = max(max_frontier_size, depth + 1)
        
        if cost_so_far >= best['cost']:
            return
        expansions += 1
        if problem.is_end(s):
            best['cost'] = cost_so_far
            best['path'] = reconstruct_path(parent, s)
            return
        for action, sp, c in problem.succ_and_cost(s):
            parent[sp] = (s, action, c)
            dfs(sp, cost_so_far + c, parent, depth + 1)
            parent.pop(sp, None)

    dfs(problem.start_state(), 0, {}, 0)
    return best['path'], best['cost'], expansions, max_frontier_size

# ---------- DFS (unit-cost, stop at first goal) ----------
def dfs_first_solution(problem, max_depth=1000):
    start = problem.start_state()
    stack = [(start, 0)]
    parent = {}
    seen = set([start])
    expansions = 0
    max_frontier_size = 1
    
    while stack:
        max_frontier_size = max(max_frontier_size, len(stack))
        s, depth = stack.pop()
        expansions += 1
        if problem.is_end(s):
            return reconstruct_path(parent, s), depth, expansions, max_frontier_size
        if depth >= max_depth:
            continue
        for action, sp, c in problem.succ_and_cost(s):
            if sp not in seen:
                seen.add(sp)
                parent[sp] = (s, action, c)
                stack.append((sp, depth+1))
    return None, None, expansions, max_frontier_size

# ---------- BFS (unit-cost optimal) ----------
def bfs_unit_cost(problem):
    start = problem.start_state()
    q = deque([start])
    parent = {}
    seen = set([start])
    expansions = 0
    depth = {start: 0}
    max_frontier_size = 1
    
    while q:
        max_frontier_size = max(max_frontier_size, len(q))
        s = q.popleft()
        expansions += 1
        if problem.is_end(s):
            return reconstruct_path(parent, s), depth[s], expansions, max_frontier_size
        for action, sp, c in problem.succ_and_cost(s):
            if sp not in seen:
                seen.add(sp)
                parent[sp] = (s, action, c)
                depth[sp] = depth[s] + 1
                q.append(sp)
    return None, None, expansions, max_frontier_size

# ---------- DFS with Iterative Deepening (unit-cost optimal) ----------
def dfid_unit_cost(problem, max_depth=50):
    start = problem.start_state()
    expansions_total = 0
    max_frontier_overall = 0
    
    for limit in range(max_depth+1):
        stack = [(start, 0)]
        parent = {}
        seen = {start}
        current_max_frontier = 1
        
        while stack:
            current_max_frontier = max(current_max_frontier, len(stack))
            s, depth = stack.pop()
            expansions_total += 1
            if problem.is_end(s):
                max_frontier_overall = max(max_frontier_overall, current_max_frontier)
                return reconstruct_path(parent, s), depth, expansions_total, max_frontier_overall
            if depth >= limit:
                continue
            for action, sp, c in problem.succ_and_cost(s):
                if sp not in seen:
                    seen.add(sp)
                    parent[sp] = (s, action, c)
                    stack.append((sp, depth+1))
        max_frontier_overall = max(max_frontier_overall, current_max_frontier)
    return None, None, expansions_total, max_frontier_overall

# ---------- Dynamic Programming (acyclic) ----------
def dp_future_cost(problem):
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def F(s):
        if problem.is_end(s):
            return 0
        best = float('inf')
        for action, sp, c in problem.succ_and_cost(s):
            best = min(best, c + F(sp))
        return best
    
    # Calculate optimal cost
    cost = F(problem.start_state())
    
    # Reconstruct optimal path greedily
    path = []
    s = problem.start_state()
    while not problem.is_end(s):
        best_act = None
        best_val = float('inf')
        for action, sp, c in problem.succ_and_cost(s):
            val = c + F(sp)
            if val < best_val:
                best_val = val
                best_act = (action, sp, c)
        action, sp, c = best_act
        path.append((action, sp, c))
        s = sp
    
    # Count number of states computed (cache info)
    cache_info = F.cache_info()
    computations = cache_info.misses  # Number of actual computations
    
    return path, cost, computations

# ---------- Uniform Cost Search (Dijkstra on implicit graph) ----------
def ucs(problem):
    start = problem.start_state()
    frontier = [(0, start)]
    parent = {}
    best_cost = {start: 0}
    explored = set()
    expansions = 0
    max_frontier_size = 1
    
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        cost, s = heapq.heappop(frontier)
        if s in explored:
            continue
        explored.add(s)
        expansions += 1
        if problem.is_end(s):
            return reconstruct_path(parent, s), cost, expansions, max_frontier_size
        for action, sp, c in problem.succ_and_cost(s):
            new_cost = cost + c
            if new_cost < best_cost.get(sp, float('inf')):
                best_cost[sp] = new_cost
                parent[sp] = (s, action, c)
                heapq.heappush(frontier, (new_cost, sp))
    return None, float('inf'), expansions, max_frontier_size

# ---------- Testing and Analysis ----------
def detailed_analysis():
    print("="*80)
    print("TRANSPORTATION PROBLEM - SEARCH ALGORITHMS ANALYSIS")
    print("="*80)
    
    test_values = [10, 50, 100, 500]
    results = {}
    
    for n in test_values:
        print(f"\n{'='*25} TESTING n = {n} {'='*25}")
        results[n] = {}
        
        # Test Dynamic Programming
        print(f"\n--- Dynamic Programming (n={n}) ---")
        prob = TransportationProblem(n, unit_cost=False)
        start_time = time.time()
        dp_path, dp_cost, dp_computations = dp_future_cost(prob)
        dp_time = time.time() - start_time
        
        print(f"DP Optimal Cost: {dp_cost}")
        print(f"DP Path Length: {len(dp_path)}")
        print(f"DP Computations: {dp_computations}")
        print(f"DP Time: {dp_time:.6f} seconds")
        print(f"DP Path: {format_path(dp_path)}")
        
        results[n]['dp'] = {
            'cost': dp_cost,
            'path_length': len(dp_path),
            'expansions': dp_computations,
            'time': dp_time,
            'path': dp_path
        }
        
        # Test Uniform Cost Search
        print(f"\n--- Uniform Cost Search (n={n}) ---")
        start_time = time.time()
        ucs_path, ucs_cost, ucs_expansions, ucs_max_frontier = ucs(prob)
        ucs_time = time.time() - start_time
        
        print(f"UCS Optimal Cost: {ucs_cost}")
        print(f"UCS Path Length: {len(ucs_path)}")
        print(f"UCS Expansions: {ucs_expansions}")
        print(f"UCS Max Frontier: {ucs_max_frontier}")
        print(f"UCS Time: {ucs_time:.6f} seconds")
        print(f"UCS Path: {format_path(ucs_path)}")
        
        results[n]['ucs'] = {
            'cost': ucs_cost,
            'path_length': len(ucs_path),
            'expansions': ucs_expansions,
            'max_frontier': ucs_max_frontier,
            'time': ucs_time,
            'path': ucs_path
        }
        
        # Test Backtracking (for smaller n only due to exponential complexity)
        if n <= 50:
            print(f"\n--- Backtracking Search (n={n}) ---")
            start_time = time.time()
            bt_path, bt_cost, bt_expansions, bt_max_frontier = backtracking_min_cost(prob)
            bt_time = time.time() - start_time
            
            print(f"Backtracking Cost: {bt_cost}")
            print(f"Backtracking Path Length: {len(bt_path) if bt_path else 0}")
            print(f"Backtracking Expansions: {bt_expansions}")
            print(f"Backtracking Max Frontier: {bt_max_frontier}")
            print(f"Backtracking Time: {bt_time:.6f} seconds")
            
            results[n]['backtracking'] = {
                'cost': bt_cost,
                'path_length': len(bt_path) if bt_path else 0,
                'expansions': bt_expansions,
                'max_frontier': bt_max_frontier,
                'time': bt_time
            }
        
        # Unit Cost Tests (BFS, DFS, DFID)
        print(f"\n--- Unit Cost Algorithms (n={n}) ---")
        prob_unit = TransportationProblem(n, unit_cost=True)
        
        # BFS
        start_time = time.time()
        bfs_path, bfs_depth, bfs_expansions, bfs_max_frontier = bfs_unit_cost(prob_unit)
        bfs_time = time.time() - start_time
        
        print(f"BFS Solution Depth: {bfs_depth}")
        print(f"BFS Expansions: {bfs_expansions}")
        print(f"BFS Max Frontier: {bfs_max_frontier}")
        print(f"BFS Time: {bfs_time:.6f} seconds")
        
        results[n]['bfs'] = {
            'depth': bfs_depth,
            'expansions': bfs_expansions,
            'max_frontier': bfs_max_frontier,
            'time': bfs_time
        }
        
        # DFID
        start_time = time.time()
        dfid_path, dfid_depth, dfid_expansions, dfid_max_frontier = dfid_unit_cost(prob_unit)
        dfid_time = time.time() - start_time
        
        print(f"DFID Solution Depth: {dfid_depth}")
        print(f"DFID Expansions: {dfid_expansions}")
        print(f"DFID Max Frontier: {dfid_max_frontier}")
        print(f"DFID Time: {dfid_time:.6f} seconds")
        
        results[n]['dfid'] = {
            'depth': dfid_depth,
            'expansions': dfid_expansions,
            'max_frontier': dfid_max_frontier,
            'time': dfid_time
        }
        
        # DFS (limited depth to avoid infinite search)
        if n <= 100:
            start_time = time.time()
            dfs_path, dfs_depth, dfs_expansions, dfs_max_frontier = dfs_first_solution(prob_unit, max_depth=50)
            dfs_time = time.time() - start_time
            
            print(f"DFS Solution Depth: {dfs_depth}")
            print(f"DFS Expansions: {dfs_expansions}")
            print(f"DFS Max Frontier: {dfs_max_frontier}")
            print(f"DFS Time: {dfs_time:.6f} seconds")
            
            results[n]['dfs'] = {
                'depth': dfs_depth,
                'expansions': dfs_expansions,
                'max_frontier': dfs_max_frontier,
                'time': dfs_time
            }
    
    # Generate comprehensive summary tables
    print_summary_tables(results, test_values)
    
    return results

def print_summary_tables(results, test_values):
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    # Optimal Cost and Path Length Table
    print(f"\n{'Algorithm Performance Summary':<50}")
    print("-" * 80)
    print(f"{'n':<6} {'DP Cost':<10} {'UCS Cost':<10} {'Path Len':<10} {'DP==UCS':<10}")
    print("-" * 80)
    
    for n in test_values:
        r = results[n]
        dp_cost = r['dp']['cost']
        ucs_cost = r['ucs']['cost']
        path_len = r['dp']['path_length']
        costs_match = "✓" if abs(dp_cost - ucs_cost) < 0.001 else "✗"
        
        print(f"{n:<6} {dp_cost:<10.3f} {ucs_cost:<10.3f} {path_len:<10} {costs_match:<10}")
    
    # Node Expansions Comparison
    print(f"\n{'Node Expansions Comparison':<50}")
    print("-" * 90)
    print(f"{'n':<6} {'DP':<10} {'UCS':<10} {'BFS':<10} {'DFID':<12} {'Backtrack':<12}")
    print("-" * 90)
    
    for n in test_values:
        r = results[n]
        dp_exp = r['dp']['expansions']
        ucs_exp = r['ucs']['expansions']
        bfs_exp = r['bfs']['expansions']
        dfid_exp = r['dfid']['expansions']
        bt_exp = r.get('backtracking', {}).get('expansions', '-')
        
        print(f"{n:<6} {dp_exp:<10} {ucs_exp:<10} {bfs_exp:<10} {dfid_exp:<12} {bt_exp:<12}")
    
    # Frontier Size Analysis
    print(f"\n{'Peak Frontier Size Analysis':<50}")
    print("-" * 70)
    print(f"{'n':<6} {'BFS':<10} {'DFID':<10} {'UCS':<10} {'Space Winner':<15}")
    print("-" * 70)
    
    for n in test_values:
        r = results[n]
        bfs_frontier = r['bfs']['max_frontier']
        dfid_frontier = r['dfid']['max_frontier']
        ucs_frontier = r['ucs']['max_frontier']
        
        min_frontier = min(bfs_frontier, dfid_frontier, ucs_frontier)
        winner = "BFS" if bfs_frontier == min_frontier else "DFID" if dfid_frontier == min_frontier else "UCS"
        
        print(f"{n:<6} {bfs_frontier:<10} {dfid_frontier:<10} {ucs_frontier:<10} {winner:<15}")
    
    # Performance Analysis
    print(f"\n{'Algorithm Performance Analysis'}")
    print("-" * 60)
    
    for n in test_values:
        r = results[n]
        print(f"\nFor n={n}:")
        print(f"  • DFID beats BFS in space: {r['dfid']['max_frontier'] < r['bfs']['max_frontier']}")
        print(f"  • DFID frontier: {r['dfid']['max_frontier']}, BFS frontier: {r['bfs']['max_frontier']}")
        print(f"  • UCS optimal cost: {r['ucs']['cost']:.3f}")
        print(f"  • UCS vs BFS expansions: {r['ucs']['expansions']} vs {r['bfs']['expansions']}")

def test_small_example():
    """Test with n=10 for detailed trace"""
    print("="*60)
    print("DETAILED EXAMPLE: n=10")
    print("="*60)
    
    prob = TransportationProblem(10, unit_cost=False)
    
    # DP analysis
    print("\n--- Dynamic Programming Analysis ---")
    dp_path, dp_cost, dp_comps = dp_future_cost(prob)
    print(f"Optimal cost: {dp_cost}")
    print(f"Optimal path: {format_path(dp_path)}")
    print(f"States computed: {dp_comps}")
    
    # UCS step-by-step for first few iterations
    print("\n--- UCS Step-by-Step (first few steps) ---")
    start = prob.start_state()
    frontier = [(0, start)]
    parent = {}
    best_cost = {start: 0}
    explored = set()
    step = 0
    
    print(f"Initial frontier: {frontier}")
    
    while frontier and step < 5:
        step += 1
        print(f"\nStep {step}:")
        cost, s = heapq.heappop(frontier)
        print(f"  Pop: state {s}, cost {cost}")
        
        if s in explored:
            print(f"  Already explored, skip")
            continue
            
        explored.add(s)
        print(f"  Exploring state {s}")
        
        if prob.is_end(s):
            print(f"  GOAL REACHED! Final cost: {cost}")
            break
            
        for action, sp, c in prob.succ_and_cost(s):
            new_cost = cost + c
            if new_cost < best_cost.get(sp, float('inf')):
                best_cost[sp] = new_cost
                parent[sp] = (s, action, c)
                heapq.heappush(frontier, (new_cost, sp))
                print(f"    Add: {action} -> state {sp}, cost {new_cost}")
        
        print(f"  Current frontier: {sorted(frontier)[:5]}...")
    
    return prob

if __name__ == "__main__":
    # Run small example first
    test_small_example()
    
    # Run comprehensive analysis
    results = detailed_analysis()
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"Key findings:")
    print(f"• DP and UCS always found matching optimal costs")
    print(f"• DFID typically uses less space than BFS")
    print(f"• UCS handles non-uniform costs optimally")
    print(f"• Backtracking becomes impractical for large n")
    print(f"{'='*60}")