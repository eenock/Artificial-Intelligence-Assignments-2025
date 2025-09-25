from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import time

State = Tuple[int, int]  # (row, col)
Action = str            # 'U','D','L','R'

class GridWorldMDP:
    def __init__(self, rows=3, cols=4, walls={(2,2)}, goals={(1,4):1.0}, lava={(2,4):-1.0},
                    step_reward=-0.04, slip=0.2):
        self.R = rows
        self.C = cols
        self.walls = set(walls)
        self.terminal = dict(goals)
        self.terminal.update(lava)
        self.step_reward = step_reward
        self.slip = slip
        self.actions_list = ['U', 'D', 'L', 'R']
    
    def states(self):
        """Generate all valid states"""
        for r in range(1, self.R + 1):
            for c in range(1, self.C + 1):
                if (r, c) not in self.walls:
                    yield (r, c)
    
    def is_end(self, s):
        """Check if state is terminal"""
        return s in self.terminal
    
    def actions(self, s):
        """Available actions from state s"""
        return [] if self.is_end(s) else self.actions_list
    
    def _move(self, s, a):
        """Execute movement action from state s"""
        r, c = s
        drc = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}[a]
        rr, cc = r + drc[0], c + drc[1]
        
        # Check bounds and walls
        if not (1 <= rr <= self.R and 1 <= cc <= self.C) or (rr, cc) in self.walls:
            return s  # Stay in place if invalid move
        return (rr, cc)
    
    def transitions(self, s, a):
        """Get transition probabilities and rewards"""
        if self.is_end(s):
            return
        
        perp = {'U': ['L', 'R'], 'D': ['L', 'R'], 'L': ['U', 'D'], 'R': ['U', 'D']}[a]
        
        # Possible outcomes: intended action + perpendicular slip
        outcomes = [
            (self._move(s, a), 1 - self.slip),           # Intended direction
            (self._move(s, perp[0]), self.slip / 2.0),   # Slip left/up
            (self._move(s, perp[1]), self.slip / 2.0)    # Slip right/down
        ]
        
        # Aggregate probabilities for duplicate states
        probs = {}
        for sp, p in outcomes:
            probs[sp] = probs.get(sp, 0.0) + p
        
        # Yield transitions with rewards
        for sp, p in probs.items():
            r = self.terminal.get(sp, self.step_reward)
            yield (sp, p, r)

def value_iteration(mdp, gamma=0.99, eps=1e-6, asynchronous=False, max_iters=10000):
    """
    Value iteration with detailed logging
    """
    V = {s: 0.0 for s in mdp.states()}
    residuals = []
    value_history = []
    iters = 0
    
    while iters < max_iters:
        iters += 1
        delta = 0.0
        old_V = V.copy()
        
        for s in list(mdp.states()):
            if mdp.is_end(s):
                V[s] = mdp.terminal[s]
                continue
            
            # Compute Bellman backup
            best = float('-inf')
            for a in mdp.actions(s):
                q = 0.0
                for sp, p, r in mdp.transitions(s, a):
                    q += p * (r + gamma * V[sp])
                if q > best:
                    best = q
            
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        
        residuals.append(delta)
        value_history.append(V.copy())
        
        if delta <= eps:
            break
    
    # Extract greedy policy
    pi = {}
    for s in mdp.states():
        if mdp.is_end(s):
            continue
        
        best_a, best_q = None, float('-inf')
        for a in mdp.actions(s):
            q = sum(p * (r + gamma * V[sp]) for sp, p, r in mdp.transitions(s, a))
            if q > best_q:
                best_q, best_a = q, a
        pi[s] = best_a
    
    return V, pi, iters, residuals, value_history

def policy_evaluation(mdp, pi, gamma=0.99, eps=1e-8, max_iters=10000):
    """
    Policy evaluation with convergence tracking
    """
    V = {s: 0.0 for s in mdp.states()}
    residuals = []
    
    for iter_count in range(max_iters):
        delta = 0.0
        Vprev = V.copy()
        
        for s in mdp.states():
            if mdp.is_end(s):
                V[s] = mdp.terminal[s]
                continue
            
            a = pi[s]
            val = sum(p * (r + gamma * Vprev[sp]) for sp, p, r in mdp.transitions(s, a))
            delta = max(delta, abs(val - Vprev[s]))
            V[s] = val
        
        residuals.append(delta)
        if delta <= eps:
            break
    
    return V, residuals

def policy_improvement(mdp, V, gamma=0.99):
    """
    Policy improvement step
    """
    pi = {}
    policy_changed = False
    
    for s in mdp.states():
        if mdp.is_end(s):
            continue
        
        # Find best action
        best_a, best_q = None, float('-inf')
        for a in mdp.actions(s):
            q = sum(p * (r + gamma * V[sp]) for sp, p, r in mdp.transitions(s, a))
            if q > best_q:
                best_q, best_a = q, a
        pi[s] = best_a
    
    return pi

def policy_iteration(mdp, gamma=0.99, eval_eps=1e-8, max_pe_iters=1000):
    """
    Policy iteration with detailed logging
    """
    # Initialize with uniform random policy (or all 'R')
    pi = {s: 'R' for s in mdp.states() if not mdp.is_end(s)}
    
    pi_iters = 0
    all_residuals = []
    policy_history = []
    value_history = []
    
    while True:
        pi_iters += 1
        policy_history.append(pi.copy())
        
        # Policy evaluation
        V, pe_residuals = policy_evaluation(mdp, pi, gamma=gamma, eps=eval_eps, max_iters=max_pe_iters)
        all_residuals.extend(pe_residuals)
        value_history.append(V.copy())
        
        # Policy improvement
        new_pi = policy_improvement(mdp, V, gamma=gamma)
        
        # Check convergence
        if new_pi == pi:
            break
        pi = new_pi
    
    return V, pi, pi_iters, all_residuals, policy_history, value_history

def q_value_iteration(mdp, gamma=0.99, eps=1e-6, max_iters=10000):
    """
    Q-value iteration with logging
    """
    Q = {(s, a): 0.0 for s in mdp.states() for a in mdp.actions(s)}
    residuals = []
    q_history = []
    
    def best_next(sp):
        if mdp.is_end(sp):
            return 0.0
        return max(Q[(sp, a)] for a in mdp.actions(sp))
    
    iters = 0
    while iters < max_iters:
        iters += 1
        delta = 0.0
        
        for s in mdp.states():
            if mdp.is_end(s):
                # Terminal states
                for a in ['U', 'D', 'L', 'R']:
                    if (s, a) in Q:
                        Q[(s, a)] = mdp.terminal[s]
                continue
            
            for a in mdp.actions(s):
                old = Q[(s, a)]
                new = sum(p * (r + gamma * best_next(sp)) for sp, p, r in mdp.transitions(s, a))
                Q[(s, a)] = new
                delta = max(delta, abs(new - old))
        
        residuals.append(delta)
        q_history.append(Q.copy())
        
        if delta <= eps:
            break
    
    # Extract greedy policy
    pi = {}
    for s in mdp.states():
        if not mdp.is_end(s):
            pi[s] = max(mdp.actions(s), key=lambda a: Q[(s, a)])
    
    return Q, pi, iters, residuals, q_history

def print_policy(mdp, pi, title="Policy"):
    """
    Print policy as a grid with arrows
    """
    print(f"\n{title}:")
    arrow_map = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
    
    for r in range(1, mdp.R + 1):
        row_str = ""
        for c in range(1, mdp.C + 1):
            if (r, c) in mdp.walls:
                row_str += "█ "
            elif (r, c) in mdp.terminal:
                if mdp.terminal[(r, c)] > 0:
                    row_str += "G "
                else:
                    row_str += "L "
            else:
                row_str += arrow_map.get(pi.get((r, c), '?'), '?') + " "
        print(row_str)

def print_values(mdp, V, title="Values"):
    """
    Print value function as a grid
    """
    print(f"\n{title}:")
    for r in range(1, mdp.R + 1):
        row_str = ""
        for c in range(1, mdp.C + 1):
            if (r, c) in mdp.walls:
                row_str += "   █   "
            else:
                val = V.get((r, c), 0.0)
                row_str += f"{val:6.3f} "
        print(row_str)

def run_comprehensive_experiment():
    """
    Run comprehensive experiments across different parameter settings
    """
    print("="*70)
    print("MDP GRIDWORLD COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    # Parameter combinations to test
    slip_values = [0.0, 0.1, 0.2]
    step_reward_values = [-0.04, -0.02, 0.0]
    
    results = {}
    
    for slip in slip_values:
        for r_step in step_reward_values:
            print(f"\n{'='*50}")
            print(f"EXPERIMENT: slip={slip}, step_reward={r_step}")
            print(f"{'='*50}")
            
            # Create MDP with current parameters
            mdp = GridWorldMDP(slip=slip, step_reward=r_step)
            
            # Run all algorithms
            algorithms = {}
            
            # Value Iteration
            print("\n--- Value Iteration ---")
            start_time = time.time()
            V_vi, pi_vi, iters_vi, res_vi, vh_vi = value_iteration(mdp, eps=1e-5)
            time_vi = time.time() - start_time
            print(f"Converged in {iters_vi} iterations, {time_vi:.4f}s")
            print(f"Final residual: {res_vi[-1]:.2e}")
            
            algorithms['VI'] = {
                'V': V_vi, 'pi': pi_vi, 'iters': iters_vi, 
                'time': time_vi, 'residuals': res_vi
            }
            
            # Policy Iteration
            print("\n--- Policy Iteration ---")
            start_time = time.time()
            V_pi, pi_pi, iters_pi, res_pi, pih_pi, vh_pi = policy_iteration(mdp, eval_eps=1e-6)
            time_pi = time.time() - start_time
            print(f"Converged in {iters_pi} policy iterations, {time_pi:.4f}s")
            print(f"Total PE steps: {len(res_pi)}")
            
            algorithms['PI'] = {
                'V': V_pi, 'pi': pi_pi, 'iters': iters_pi,
                'time': time_pi, 'residuals': res_pi
            }
            
            # Q-Value Iteration
            print("\n--- Q-Value Iteration ---")
            start_time = time.time()
            Q_qi, pi_qi, iters_qi, res_qi, qh_qi = q_value_iteration(mdp, eps=1e-5)
            time_qi = time.time() - start_time
            print(f"Converged in {iters_qi} iterations, {time_qi:.4f}s")
            print(f"Final residual: {res_qi[-1]:.2e}")
            
            algorithms['Q-Iter'] = {
                'Q': Q_qi, 'pi': pi_qi, 'iters': iters_qi,
                'time': time_qi, 'residuals': res_qi
            }
            
            # Print policies and values for this configuration
            print_values(mdp, V_vi, "Value Function (VI)")
            print_policy(mdp, pi_vi, "Optimal Policy")
            
            # Store results
            results[(slip, r_step)] = algorithms
    
    return results

def analyze_convergence(results):
    """
    Analyze and plot convergence behavior
    """
    print(f"\n{'='*70}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*70}")
    
    # Create summary table
    print(f"\n{'Parameter Set':<20} {'VI Iters':<10} {'PI Iters':<10} {'Q Iters':<10} {'VI Time':<10} {'PI Time':<10}")
    print("-" * 80)
    
    for (slip, r_step), algs in results.items():
        param_str = f"s={slip}, r={r_step}"
        vi_iters = algs['VI']['iters']
        pi_iters = algs['PI']['iters']
        qi_iters = algs['Q-Iter']['iters']
        vi_time = algs['VI']['time']
        pi_time = algs['PI']['time']
        
        print(f"{param_str:<20} {vi_iters:<10} {pi_iters:<10} {qi_iters:<10} {vi_time:<10.4f} {pi_time:<10.4f}")
    
    # Analyze policy differences
    print(f"\n{'Policy Consistency Analysis'}")
    print("-" * 40)
    
    for (slip, r_step), algs in results.items():
        vi_pi = algs['VI']['pi']
        pi_pi = algs['PI']['pi']
        qi_pi = algs['Q-Iter']['pi']
        
        # Check if all policies are identical
        vi_pi_match = vi_pi == pi_pi
        qi_pi_match = vi_pi == qi_pi
        
        print(f"slip={slip}, r_step={r_step}: VI==PI: {vi_pi_match}, VI==Q: {qi_pi_match}")

def create_convergence_plots(results):
    """
    Create convergence plots for different algorithms
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MDP Algorithm Convergence Analysis', fontsize=16)
    
    plot_idx = 0
    for (slip, r_step), algs in list(results.items())[:6]:  # Plot first 6 configurations
        row = plot_idx // 3
        col = plot_idx % 3
        ax = axes[row, col]
        
        # Plot VI convergence
        vi_residuals = algs['VI']['residuals']
        ax.semilogy(range(len(vi_residuals)), vi_residuals, 'b-', label='Value Iteration', linewidth=2)
        
        # Plot PI convergence (approximate by PE residuals)
        pi_residuals = algs['PI']['residuals']
        ax.semilogy(range(len(pi_residuals)), pi_residuals, 'r-', label='Policy Iteration', linewidth=2)
        
        # Plot Q-Iter convergence
        qi_residuals = algs['Q-Iter']['residuals']
        ax.semilogy(range(len(qi_residuals)), qi_residuals, 'g-', label='Q-Value Iteration', linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual')
        ax.set_title(f'slip={slip}, step_reward={r_step}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main experimental driver
    """
    print("Starting MDP GridWorld experiments...")
    
    # Run comprehensive experiments
    results = run_comprehensive_experiment()
    
    # Analyze results
    analyze_convergence(results)
    
    # Create plots
    try:
        create_convergence_plots(results)
    except Exception as e:
        print(f"Plotting failed: {e}")
        print("Continuing with text-based analysis...")
    
    print(f"\n{'='*70}")
    print("EXPERIMENTAL ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    return results

if __name__ == '__main__':
    results = main()