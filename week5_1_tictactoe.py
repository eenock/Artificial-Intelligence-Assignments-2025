from typing import List, Tuple, Optional, Dict
import random

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not found. Install with 'pip install matplotlib' for plots.")

MAX, MIN = 'X', 'O'

# =============================================================================
# GAME API
# =============================================================================

def pretty(s: str) -> str:
    """Pretty print Tic-Tac-Toe board."""
    g = [s[i:i+3] for i in range(0, 9, 3)]
    return "\n".join(" ".join(c if c != '.' else '_' for c in row) for row in g)

def player_to_move(s: str) -> str:
    return MAX if s.count(MAX) == s.count(MIN) else MIN

def legal_moves(s: str) -> List[int]:
    return [i for i, c in enumerate(s) if c == '.']

def next_state(s: str, a: int) -> str:
    p = player_to_move(s)
    return s[:a] + p + s[a+1:]

def lines() -> List[Tuple[int, int, int]]:
    return [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]

def winner(s: str) -> Optional[str]:
    for a, b, c in lines():
        if s[a] != '.' and s[a] == s[b] == s[c]:
            return s[a]
    return None

def is_terminal(s: str) -> bool:
    return winner(s) is not None or '.' not in s

def utility(s: str) -> int:
    w = winner(s)
    if w == MAX:
        return +1
    if w == MIN:
        return -1
    return 0

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def eval_features(s: str) -> Tuple[int, int, int, int]:
    """
    Return (open-X-2s, open-O-2s, centerX, cornerX)
    - open-X-2s: Lines with 2 X's and no O's
    - open-O-2s: Lines with 2 O's and no X's
    - centerX: 1 if X controls center
    - cornerX: Number of corners controlled by X
    """
    openX = openO = 0
    for a, b, c in lines():
        line = s[a] + s[b] + s[c]
        if line.count(MIN) == 0 and line.count(MAX) == 2:
            openX += 1
        if line.count(MAX) == 0 and line.count(MIN) == 2:
            openO += 1
    
    centerX = 1 if s[4] == MAX else 0
    corners = [0, 2, 6, 8]
    cornerX = sum(1 for i in corners if s[i] == MAX)
    
    return (openX, openO, centerX, cornerX)

def eval_linear(s: str, w=(3, -3, 1, 1)) -> float:
    """
    Linear evaluation function: w^T * features
    Default weights: (openX=3, openO=-3, centerX=1, cornerX=1)
    """
    f = eval_features(s)
    return sum(wi * fi for wi, fi in zip(w, f))

# =============================================================================
# MINIMAX
# =============================================================================

def minimax(s: str, depth: int, w=(3, -3, 1, 1)) -> Tuple[float, Optional[int], int]:
    """
    Minimax algorithm with depth limiting.
    
    Args:
        s: Current state (9-char string)
        depth: Maximum search depth
        w: Evaluation weights (openX, openO, centerX, cornerX)
    
    Returns:
        (value, best_move, nodes_expanded)
    """
    nodes = 0
    
    def mm(state, d) -> float:
        nonlocal nodes
        nodes += 1
        
        # Terminal or depth limit
        if is_terminal(state) or d == 0:
            return utility(state) if is_terminal(state) else eval_linear(state, w)
        
        p = player_to_move(state)
        moves = legal_moves(state)
        
        if p == MAX:
            return max(mm(next_state(state, a), d - 1) for a in moves)
        else:
            return min(mm(next_state(state, a), d - 1) for a in moves)
    
    # Root level: find best move
    p = player_to_move(s)
    best_move = None
    best_val = -1e9 if p == MAX else 1e9
    
    for a in legal_moves(s):
        v = mm(next_state(s, a), depth - 1)
        if (p == MAX and v > best_val) or (p == MIN and v < best_val):
            best_val, best_move = v, a
    
    return best_val, best_move, nodes

# =============================================================================
# ALPHA-BETA PRUNING
# =============================================================================

def move_order_heuristic(s: str, moves: List[int]) -> List[int]:
    """
    Order moves by heuristic quality: center > corners > edges
    This improves alpha-beta pruning effectiveness.
    """
    center = [4]
    corners = [0, 2, 6, 8]
    edges = [1, 3, 5, 7]
    order = center + corners + edges
    return sorted(moves, key=lambda a: order.index(a) if a in order else 99)

def alphabeta(s: str, depth: int, w=(3, -3, 1, 1), use_tt=True) -> Tuple[float, Optional[int], int, int]:
    """
    Alpha-beta pruning with move ordering and transposition table.
    
    Args:
        s: Current state
        depth: Maximum search depth
        w: Evaluation weights
        use_tt: Whether to use transposition table
    
    Returns:
        (value, best_move, nodes_expanded, prunes_count)
    """
    nodes = 0
    prunes = 0
    TT: Dict[Tuple[str, int], float] = {} if use_tt else None
    
    def ab(state, d, alpha, beta) -> float:
        nonlocal nodes, prunes
        nodes += 1
        
        # Transposition table lookup
        if use_tt:
            key = (state, d)
            if key in TT:
                return TT[key]
        
        # Terminal or depth limit
        if is_terminal(state) or d == 0:
            val = utility(state) if is_terminal(state) else eval_linear(state, w)
            if use_tt:
                TT[(state, d)] = val
            return val
        
        p = player_to_move(state)
        moves = move_order_heuristic(state, legal_moves(state))
        
        if p == MAX:
            val = -1e9
            for a in moves:
                val = max(val, ab(next_state(state, a), d - 1, alpha, beta))
                alpha = max(alpha, val)
                if alpha >= beta:
                    prunes += 1
                    break
            if use_tt:
                TT[(state, d)] = val
            return val
        else:
            val = 1e9
            for a in moves:
                val = min(val, ab(next_state(state, a), d - 1, alpha, beta))
                beta = min(beta, val)
                if alpha >= beta:
                    prunes += 1
                    break
            if use_tt:
                TT[(state, d)] = val
            return val
    
    # Root level: find best move
    p = player_to_move(s)
    best_move = None
    best_val = -1e9 if p == MAX else 1e9
    
    for a in legal_moves(s):
        v = ab(next_state(s, a), depth - 1, -1e9, 1e9)
        if (p == MAX and v > best_val) or (p == MIN and v < best_val):
            best_val, best_move = v, a
    
    return best_val, best_move, nodes, prunes

# =============================================================================
# EXPERIMENTS
# =============================================================================

def generate_random_midgame():
    """Generate random mid-game position."""
    state = ['.'] * 9
    num_moves = random.randint(3, 6)
    for i in range(num_moves):
        empty = [j for j, c in enumerate(state) if c == '.']
        if not empty:
            break
        pos = random.choice(empty)
        state[pos] = MAX if i % 2 == 0 else MIN
    return ''.join(state)

def run_experiments():
    """Run all Tic-Tac-Toe experiments."""
    
    print("="*70)
    print("TIC-TAC-TOE: MINIMAX AND ALPHA-BETA EXPERIMENTS")
    print("="*70)
    
    # Test position
    test_state = "X.O..O..."
    
    print("\nTest Position:")
    print(pretty(test_state))
    print(f"\nPlayer to move: {player_to_move(test_state)}")
    print(f"Features (openX, openO, centerX, cornerX): {eval_features(test_state)}")
    
    # Experiment 1: Algorithm comparison on single position
    print("\n" + "-"*70)
    print("EXPERIMENT 1: Minimax vs Alpha-Beta (Single Position)")
    print("-"*70)
    print(f"{'Depth':<8} {'MM Nodes':<12} {'AB Nodes':<12} {'Prunes':<10} {'Speedup':<10}")
    print("-"*70)
    
    exp1_depths = []
    exp1_mm_nodes = []
    exp1_ab_nodes = []
    exp1_prunes = []
    
    for depth in [2, 3, 4, 5, 6]:
        v1, a1, n1 = minimax(test_state, depth)
        v2, a2, n2, p2 = alphabeta(test_state, depth)
        speedup = n1 / n2 if n2 > 0 else 1.0
        print(f"{depth:<8} {n1:<12} {n2:<12} {p2:<10} {speedup:<10.2f}x")
        
        exp1_depths.append(depth)
        exp1_mm_nodes.append(n1)
        exp1_ab_nodes.append(n2)
        exp1_prunes.append(p2)
    
    # Experiment 2: Average performance on random positions
    print("\n" + "-"*70)
    print("EXPERIMENT 2: Average Performance (50 Random Mid-game States)")
    print("-"*70)
    
    states = []
    while len(states) < 50:
        state = generate_random_midgame()
        if not is_terminal(state):
            states.append(state)
    
    print(f"{'Depth':<8} {'Avg MM':<12} {'Avg AB':<12} {'Avg Prunes':<12} {'Speedup':<10}")
    print("-"*70)
    
    exp2_depths = []
    exp2_mm_avg = []
    exp2_ab_avg = []
    exp2_prunes_avg = []
    exp2_speedups = []
    
    for depth in [3, 4, 5]:
        mm_nodes = []
        ab_nodes = []
        ab_prunes = []
        
        for state in states:
            _, _, n_mm = minimax(state, depth)
            _, _, n_ab, p_ab = alphabeta(state, depth)
            mm_nodes.append(n_mm)
            ab_nodes.append(n_ab)
            ab_prunes.append(p_ab)
        
        avg_mm = sum(mm_nodes) / len(mm_nodes)
        avg_ab = sum(ab_nodes) / len(ab_nodes)
        avg_prunes = sum(ab_prunes) / len(ab_prunes)
        speedup = avg_mm / avg_ab
        
        print(f"{depth:<8} {avg_mm:<12.0f} {avg_ab:<12.0f} {avg_prunes:<12.0f} {speedup:<10.2f}x")
        
        exp2_depths.append(depth)
        exp2_mm_avg.append(avg_mm)
        exp2_ab_avg.append(avg_ab)
        exp2_prunes_avg.append(avg_prunes)
        exp2_speedups.append(speedup)
    
    # Experiment 3: Weight optimization
    print("\n" + "-"*70)
    print("EXPERIMENT 3: Weight Optimization (Grid Search)")
    print("-"*70)
    print("Testing weight combinations to maximize win rate vs depth-2 opponent...")
    
    best_weights = None
    best_win_rate = -1
    
    # Grid search (simplified for speed)
    weight_candidates = [
        (3, -3, 1, 1),
        (4, -4, 1, 1),
        (3, -3, 2, 1),
        (2, -3, 1, 1),
        (3, -4, 1, 1),
    ]
    
    print(f"\n{'Weights':<20} {'Win Rate':<12} {'Draws':<10} {'Losses':<10}")
    print("-"*70)
    
    weight_results = []
    
    for weights in weight_candidates:
        wins = draws = losses = 0
        num_games = 20  # Test games per weight config
        
        for _ in range(num_games):
            # Generate random mid-game position
            state = generate_random_midgame()
            if is_terminal(state):
                continue
            
            # Play out game: our agent (MAX) vs opponent (MIN)
            current = state
            moves = 0
            max_moves = 20  # Prevent infinite loops
            
            while not is_terminal(current) and moves < max_moves:
                if player_to_move(current) == MAX:
                    # Our agent with test weights
                    _, move, _, _ = alphabeta(current, depth=3, w=weights)
                else:
                    # Opponent with default weights
                    _, move, _, _ = alphabeta(current, depth=2, w=(3, -3, 1, 1))
                
                if move is None:
                    break
                
                current = next_state(current, move)
                moves += 1
            
            # Check outcome
            w = winner(current)
            if w == MAX:
                wins += 1
            elif w == MIN:
                losses += 1
            else:
                draws += 1
        
        total = wins + draws + losses
        win_rate = wins / total if total > 0 else 0
        
        print(f"{str(weights):<20} {win_rate:<12.2%} {draws:<10} {losses:<10}")
        weight_results.append((weights, win_rate))
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_weights = weights
    
    print(f"\nBest weights: {best_weights} with win rate: {best_win_rate:.2%}")
    
    # Experiment 4: Evaluation weight analysis
    print("\n" + "-"*70)
    print("EXPERIMENT 4: Evaluation Function Analysis")
    print("-"*70)
    print("Testing best weights on sample positions...")
    
    test_positions = [
        "X.O..O...",
        "X..X.O...",
        "XX.O.O...",
    ]
    
    print(f"\n{'Position':<15} {'Eval Score':<12} {'Best Move':<10}")
    print("-"*70)
    
    for pos in test_positions:
        score = eval_linear(pos, best_weights)
        _, move, _, _ = alphabeta(pos, depth=4, w=best_weights)
        print(f"{pos:<15} {score:<12.2f} {move:<10}")
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED")
    print("="*70)
    
    # Generate plots if matplotlib is available
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        generate_plots(exp1_depths, exp1_mm_nodes, exp1_ab_nodes, exp1_prunes,
                        exp2_depths, exp2_mm_avg, exp2_ab_avg, exp2_speedups,
                        weight_results)
        print("✓ Plots saved!")
    else:
        print("\nTo generate plots, install matplotlib:")
        print("  pip install matplotlib")
        print("\nThen run the script again.")

def generate_plots(exp1_depths, exp1_mm_nodes, exp1_ab_nodes, exp1_prunes,
                    exp2_depths, exp2_mm_avg, exp2_ab_avg, exp2_speedups,
                    weight_results):
    """Generate all plots for the report."""
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Tic-Tac-Toe: Minimax vs Alpha-Beta Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Nodes expanded (single position)
    ax = axes[0, 0]
    ax.plot(exp1_depths, exp1_mm_nodes, 'o-', linewidth=2, markersize=8, label='Minimax')
    ax.plot(exp1_depths, exp1_ab_nodes, 's-', linewidth=2, markersize=8, label='Alpha-Beta')
    ax.set_xlabel('Depth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Nodes Expanded', fontsize=11, fontweight='bold')
    ax.set_title('Nodes Expanded (Single Position)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average nodes (50 positions)
    ax = axes[0, 1]
    ax.plot(exp2_depths, exp2_mm_avg, 'o-', linewidth=2, markersize=8, label='Minimax')
    ax.plot(exp2_depths, exp2_ab_avg, 's-', linewidth=2, markersize=8, label='Alpha-Beta')
    ax.set_xlabel('Depth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Nodes Expanded', fontsize=11, fontweight='bold')
    ax.set_title('Average Performance (50 Positions)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Speedup factor
    ax = axes[1, 0]
    ax.plot(exp2_depths, exp2_speedups, 'o-', linewidth=2, markersize=10, color='green')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
    ax.set_xlabel('Depth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Speedup Factor (MM/AB)', fontsize=11, fontweight='bold')
    ax.set_title('Alpha-Beta Speedup', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Weight optimization results
    ax = axes[1, 1]
    weight_labels = [str(w) for w, _ in weight_results]
    win_rates = [wr * 100 for _, wr in weight_results]
    colors = ['green' if wr == max(win_rates) else 'skyblue' for wr in win_rates]
    bars = ax.bar(range(len(weight_labels)), win_rates, color=colors, alpha=0.7)
    ax.set_xlabel('Weight Configuration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Weight Optimization Results', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(weight_labels)))
    ax.set_xticklabels(range(1, len(weight_labels) + 1))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('tictactoe_results.png', dpi=150, bbox_inches='tight')
    print("  Saved: tictactoe_results.png")
    
    # Additional plot: Log scale comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.semilogy(exp1_depths, exp1_mm_nodes, 'o-', linewidth=2, markersize=8, label='Minimax')
    ax2.semilogy(exp1_depths, exp1_ab_nodes, 's-', linewidth=2, markersize=8, label='Alpha-Beta')
    ax2.set_xlabel('Depth', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Nodes Expanded (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Exponential Growth Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tictactoe_logscale.png', dpi=150, bbox_inches='tight')
    print("  Saved: tictactoe_logscale.png")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_experiments()