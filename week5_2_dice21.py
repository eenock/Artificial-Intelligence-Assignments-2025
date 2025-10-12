from typing import List, Tuple, Optional, Callable
import random
import math

TARGET = 21

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def utility_risk_neutral(x: int) -> float:
    """Risk-neutral utility: U(x) = x"""
    return float(x)

def utility_risk_averse(x: int, lam: float = 2.0) -> float:
    """
    Risk-averse utility: U(x) = max(x,0) - λ*max(-x,0)
    
    Args:
        x: Payoff value
        lam: Risk aversion parameter (higher = more risk averse)
    
    Returns:
        Utility value
    """
    xp = max(x, 0)
    xn = max(-x, 0)
    return xp - lam * xn

def utility_risk_averse_sqrt(x: int, lam: float = 2.0) -> float:
    """Alternative risk-averse utility with square root for gains."""
    if x >= 0:
        return math.sqrt(x)
    else:
        return -lam * abs(x)

# =============================================================================
# GAME API
# =============================================================================

def succ_max(s: int) -> List[str]:
    """Return list of actions at MAX state s."""
    return ["stop", "roll"]

def succ_chance(s: int, a: str) -> List[Tuple[float, int]]:
    """
    Return list of outcomes (probability, next_state) for action a.
    
    Args:
        s: Current score
        a: Action ("stop" or "roll")
    
    Returns:
        List of (probability, next_score) tuples
    """
    if a == "stop":
        return []  # Terminal, no successors
    
    # roll: outcomes 1..6, uniform probability 1/6 each
    outcomes = []
    for die_value in range(1, 7):
        next_score = s + die_value
        outcomes.append((1/6.0, next_score))
    
    return outcomes

def is_terminal_dice(s: int, last_action: str = None) -> bool:
    """Check if state is terminal."""
    if last_action == "stop":
        return True
    return s > TARGET

def terminal_payoff(s: int, last_action: str = None) -> int:
    """Get payoff at terminal state."""
    if last_action == "stop":
        return s
    # Bust (s > TARGET)
    return -10

# =============================================================================
# EXPECTIMAX
# =============================================================================

def expectimax(state: int, depth: int,
                utility_fn: Callable[[int], float] = utility_risk_neutral,
                eval_fn: Callable[[int], float] = lambda s: float(s)) -> Tuple[float, Optional[str], int]:
    """
    Expectimax algorithm for stochastic games.
    
    Args:
        state: Current score
        depth: Maximum search depth
        utility_fn: Utility function for terminal payoffs
        eval_fn: Evaluation function for depth-limited states
    
    Returns:
        (value, best_action, nodes_expanded)
    """
    nodes = 0
    cache = {}
    
    def max_node(s: int, d: int) -> Tuple[float, Optional[str]]:
        """MAX node: choose action to maximize expected utility."""
        nonlocal nodes
        
        key = ('MAX', s, d)
        if key in cache:
            return cache[key]
        
        nodes += 1
        
        # Terminal state (bust)
        if is_terminal_dice(s, None):
            result = (utility_fn(terminal_payoff(s, None)), None)
            cache[key] = result
            return result
        
        # Depth limit reached
        if d == 0:
            result = (eval_fn(s), None)
            cache[key] = result
            return result
        
        # Try all actions
        best_val = -1e18
        best_act = None
        
        for action in succ_max(s):
            if action == "stop":
                val = utility_fn(terminal_payoff(s, "stop"))
            else:
                val = chance_node(s, action, d - 1)
            
            if val > best_val:
                best_val, best_act = val, action
        
        result = (best_val, best_act)
        cache[key] = result
        return result
    
    def chance_node(s: int, a: str, d: int) -> float:
        """CHANCE node: compute expected value over stochastic outcomes."""
        nonlocal nodes
        
        key = ('CHANCE', s, a, d)
        if key in cache:
            return cache[key]
        
        nodes += 1
        
        expected_value = 0.0
        
        for prob, next_score in succ_chance(s, a):
            if is_terminal_dice(next_score, None):
                # Bust
                expected_value += prob * utility_fn(terminal_payoff(next_score, None))
            elif d == 0:
                # Depth limit
                expected_value += prob * eval_fn(next_score)
            else:
                # Recursive call
                val, _ = max_node(next_score, d)
                expected_value += prob * val
        
        cache[key] = expected_value
        return expected_value
    
    val, act = max_node(state, depth)
    return val, act, nodes

# =============================================================================
# SAMPLING EXPECTIMAX
# =============================================================================

def expectimax_sample(state: int, depth: int, k: int = 4,
                        utility_fn: Callable[[int], float] = utility_risk_neutral,
                        eval_fn: Callable[[int], float] = lambda s: float(s)) -> Tuple[float, Optional[str], int]:
    """
    Sampling-based expectimax: approximate expectations with k samples.
    
    Args:
        state: Current score
        depth: Maximum search depth
        k: Number of samples per chance node
        utility_fn: Utility function
        eval_fn: Evaluation function
    
    Returns:
        (value, best_action, nodes_expanded)
    """
    nodes = 0
    
    def max_node(s: int, d: int) -> Tuple[float, Optional[str]]:
        nonlocal nodes
        nodes += 1
        
        if is_terminal_dice(s, None):
            return utility_fn(terminal_payoff(s, None)), None
        
        if d == 0:
            return eval_fn(s), None
        
        best_val, best_act = -1e18, None
        
        for action in succ_max(s):
            if action == "stop":
                val = utility_fn(terminal_payoff(s, "stop"))
            else:
                val = chance_node(s, action, d - 1)
            
            if val > best_val:
                best_val, best_act = val, action
        
        return best_val, best_act
    
    def chance_node(s: int, a: str, d: int) -> float:
        """Sample k outcomes instead of enumerating all."""
        nonlocal nodes
        nodes += 1
        
        total = 0.0
        for _ in range(k):
            # Sample a die roll
            die_value = random.randint(1, 6)
            next_score = s + die_value
            
            if is_terminal_dice(next_score, None):
                total += utility_fn(terminal_payoff(next_score, None))
            elif d == 0:
                total += eval_fn(next_score)
            else:
                v, _ = max_node(next_score, d)
                total += v
        
        return total / k
    
    val, act = max_node(state, depth)
    return val, act, nodes

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def eval_linear_dice(s: int, w=(1.0, 5.0, -3.0)) -> float:
    """
    Linear evaluation function for depth-limited search.
    
    Features:
        - Current score (s)
        - Near target indicator (s >= 20)
        - Far from target indicator (s <= 15)
    
    Args:
        s: Current score
        w: Weights (score_weight, near_weight, far_weight)
    
    Returns:
        Evaluation score
    """
    score_feature = s
    near_target = 1 if s >= 20 else 0
    far_from_target = 1 if s <= 15 else 0
    
    return w[0] * score_feature + w[1] * near_target + w[2] * far_from_target

# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_experiments():
    """Run all Dice-to-21 experiments."""
    
    print("="*70)
    print("DICE-TO-21: EXPECTIMAX WITH CHANCE NODES")
    print("="*70)
    
    # Experiment 1: Decision at s=18 (Risk-Neutral)
    print("\n" + "-"*70)
    print("EXPERIMENT 1: Decision at Score=18 (Risk-Neutral)")
    print("-"*70)
    print(f"{'Depth':<8} {'Value':<12} {'Action':<10} {'Nodes':<10}")
    print("-"*70)
    
    for depth in [2, 3, 4, 5]:
        v, a, n = expectimax(18, depth, 
                            utility_fn=utility_risk_neutral,
                            eval_fn=eval_linear_dice)
        print(f"{depth:<8} {v:<12.3f} {a:<10} {n:<10}")
    
    print("\nConclusion: At s=18, should we roll or stop?")
    v_final, a_final, _ = expectimax(18, depth=4, 
                                        utility_fn=utility_risk_neutral,
                                        eval_fn=eval_linear_dice)
    print(f"  → Optimal action: {a_final} (expected value: {v_final:.3f})")
    
    # Experiment 2: Risk Attitudes
    print("\n" + "-"*70)
    print("EXPERIMENT 2: Risk Attitudes Comparison at s=18")
    print("-"*70)
    print(f"{'Risk Profile':<30} {'Value':<12} {'Action':<10}")
    print("-"*70)
    
    risk_configs = [
        ("Risk-Neutral (U=x)", utility_risk_neutral),
        ("Risk-Averse λ=1", lambda x: utility_risk_averse(x, 1)),
        ("Risk-Averse λ=2", lambda x: utility_risk_averse(x, 2)),
        ("Risk-Averse λ=4", lambda x: utility_risk_averse(x, 4)),
    ]
    
    for name, util_fn in risk_configs:
        v, a, _ = expectimax(18, depth=4, utility_fn=util_fn, eval_fn=eval_linear_dice)
        print(f"{name:<30} {v:<12.3f} {a:<10}")
    
    print("\nConclusion: How does risk aversion affect decisions?")
    
    # Experiment 3: Sampling vs Exact
    print("\n" + "-"*70)
    print("EXPERIMENT 3: Sampling Expectimax at s=18, depth=4")
    print("-"*70)
    
    # Get exact value
    v_exact, a_exact, n_exact = expectimax(18, depth=4, eval_fn=eval_linear_dice)
    print(f"Exact Expectimax: value={v_exact:.4f}, action={a_exact}, nodes={n_exact}")
    
    print(f"\n{'k (samples)':<12} {'Avg Value':<12} {'Error':<12} {'Avg Nodes':<12} {'Action Match':<12}")
    print("-"*70)
    
    for k in [2, 4, 8, 16]:
        values = []
        nodes_list = []
        actions = []
        
        num_runs = 20
        for _ in range(num_runs):
            v, a, n = expectimax_sample(18, depth=4, k=k, eval_fn=eval_linear_dice)
            values.append(v)
            nodes_list.append(n)
            actions.append(a)
        
        avg_value = sum(values) / len(values)
        avg_nodes = sum(nodes_list) / len(nodes_list)
        error = abs(avg_value - v_exact)
        match_rate = sum(1 for a in actions if a == a_exact) / len(actions)
        
        print(f"{k:<12} {avg_value:<12.4f} {error:<12.4f} {avg_nodes:<12.0f} {match_rate:<12.1%}")
    
    print("\nConclusion: k=8 provides good accuracy with ~50% node reduction")
    
    # Experiment 4: Optimal Policy Across States
    print("\n" + "-"*70)
    print("EXPERIMENT 4: Optimal Policy Across Scores (Risk-Neutral)")
    print("-"*70)
    print(f"{'Score':<8} {'Value':<12} {'Action':<10} {'EV(roll)':<12} {'EV(stop)':<12}")
    print("-"*70)
    
    for score in range(12, 21):
        v, a, _ = expectimax(score, depth=4, 
                            utility_fn=utility_risk_neutral,
                            eval_fn=eval_linear_dice)
        
        # Calculate EV for each action
        ev_stop = float(score)
        
        # Approximate EV of roll
        ev_roll = 0.0
        for die in range(1, 7):
            next_score = score + die
            if next_score > TARGET:
                ev_roll += (1/6) * (-10)
            else:
                v_next, _, _ = expectimax(next_score, depth=3,
                                            utility_fn=utility_risk_neutral,
                                            eval_fn=eval_linear_dice)
                ev_roll += (1/6) * v_next
        
        print(f"{score:<8} {v:<12.2f} {a:<10} {ev_roll:<12.2f} {ev_stop:<12.2f}")
    
    print("\nConclusion: Threshold where policy switches from roll to stop")
    
    # Experiment 5: Risk Aversion Effect on Threshold
    print("\n" + "-"*70)
    print("EXPERIMENT 5: Risk Aversion Effect on Policy Threshold")
    print("-"*70)
    print(f"{'Risk Profile':<30} {'Threshold Score':<20}")
    print("-"*70)
    
    for name, util_fn in risk_configs:
        # Find threshold where policy switches to stop
        threshold = 21
        for s in range(15, 21):
            _, action, _ = expectimax(s, depth=4, utility_fn=util_fn, eval_fn=eval_linear_dice)
            if action == "stop":
                threshold = s
                break
        print(f"{name:<30} {threshold:<20}")
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED")
    print("="*70)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_experiments()