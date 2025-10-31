from typing import Dict, List, Tuple
import itertools
import random
import math
import matplotlib.pyplot as plt

# ============================================================================
# PROBLEM SETUP
# ============================================================================

DOMAIN = [0, 1, 2]
OBS = {1: 0, 2: 2, 3: 2}  # Observations: (0, 2, 2)

def o_factor(i: int, x: int) -> float:
    """Observation factor: o_i(x_i) = max(0, 2 - |x_i - obs_i|)"""
    return max(0, 2 - abs(x - OBS[i]))

def t_factor(x: int, y: int) -> float:
    """Transition factor: 2 if equal, 1 if adjacent, 0 otherwise"""
    if x == y:
        return 2
    if abs(x - y) == 1:
        return 1
    return 0

def weight(assignment: Tuple[int, int, int]) -> float:
    """Total weight for assignment (x1, x2, x3)"""
    x1, x2, x3 = assignment
    return (o_factor(1, x1) * o_factor(2, x2) * o_factor(3, x3) *
            t_factor(x1, x2) * t_factor(x2, x3))

# ============================================================================
# PART A: MRF EXACT INFERENCE
# ============================================================================

def enumerate_exact():
    """
    Enumerate all assignments to compute partition function Z and marginals.
    Returns: Z, marginals P(X_i), max-weight assignment
    """
    table = []
    Z = 0.0
    
    for assignment in itertools.product(DOMAIN, repeat=3):
        w = weight(assignment)
        if w > 0:
            table.append((assignment, w))
            Z += w
    
    # Compute marginals for each variable
    marginals = [{}, {}, {}]
    for i in range(3):
        marginals[i] = {v: 0.0 for v in DOMAIN}
    
    for (x1, x2, x3), w in table:
        marginals[0][x1] += w / Z
        marginals[1][x2] += w / Z
        marginals[2][x3] += w / Z
    
    # Find MAP (max-weight) assignment
    max_assignment, max_w = max(table, key=lambda t: t[1])
    
    return Z, marginals, table, max_assignment, max_w

# ============================================================================
# PART A: GIBBS SAMPLING
# ============================================================================

def gibbs_sampling(n_iters: int = 5000, burn_in: int = 1000, seed: int = 0):
    """
    Gibbs sampling on MRF.
    Returns: approximate marginals, convergence history
    """
    random.seed(seed)
    
    # Initialize randomly from support
    x = [random.choice(DOMAIN) for _ in range(3)]
    
    def conditional_prob(i: int, state: List[int]) -> List[float]:
        """Compute P(X_i | X_{-i}) proportional to local factors"""
        probs = []
        for v in DOMAIN:
            temp = state.copy()
            temp[i] = v
            
            # Compute product of factors touching variable i
            if i == 0:
                w = o_factor(1, v) * t_factor(v, state[1])
            elif i == 1:
                w = o_factor(2, v) * t_factor(state[0], v) * t_factor(v, state[2])
            else:
                w = o_factor(3, v) * t_factor(state[1], v)
            
            probs.append(max(0.0, w))
        
        # Normalize
        s = sum(probs)
        if s > 0:
            probs = [p / s for p in probs]
        else:
            probs = [1.0 / len(DOMAIN)] * len(DOMAIN)
        
        return probs
    
    # Track samples after burn-in
    samples = [[] for _ in range(3)]
    history_x2 = []  # For convergence plot
    
    for iteration in range(n_iters):
        # Systematic scan: update each variable in order
        for i in range(3):
            probs = conditional_prob(i, x)
            x[i] = random.choices(DOMAIN, weights=probs)[0]
        
        # After burn-in, collect samples
        if iteration >= burn_in:
            for i in range(3):
                samples[i].append(x[i])
            history_x2.append(x[1])
    
    # Compute empirical marginals
    marginals = []
    for i in range(3):
        counts = {v: 0 for v in DOMAIN}
        for sample in samples[i]:
            counts[sample] += 1
        total = len(samples[i])
        marginals.append({v: counts[v] / total for v in DOMAIN})
    
    return marginals, history_x2

# ============================================================================
# PART B: BAYESIAN NETWORK / HMM
# ============================================================================

def row_normalize(values: List[float]) -> List[float]:
    """Normalize a row to sum to 1"""
    s = sum(values)
    if s > 0:
        return [v / s for v in values]
    return [1.0 / len(values)] * len(values)

def forward_backward(evidence: Dict[int, int]):
    """
    Forward-Backward algorithm for HMM.
    Returns: forward messages, backward messages, smoothed posteriors
    """
    # Compute position-specific likelihoods
    lik = {t: {h: o_factor(t, h) for h in DOMAIN} for t in [1, 2, 3]}
    
    # Forward messages (unnormalized)
    alpha = [{} for _ in range(4)]  # Index 0 unused, 1-3 for timesteps
    
    # Initialize alpha_1
    for h in DOMAIN:
        alpha[1][h] = lik[1][h]
    
    # Forward recursion (no normalization)
    for t in [2, 3]:
        for h in DOMAIN:
            alpha[t][h] = lik[t][h] * sum(
                alpha[t-1][h_prev] * t_factor(h_prev, h)
                for h_prev in DOMAIN
            )
    
    # Backward messages (unnormalized)
    beta = [{} for _ in range(4)]
    for h in DOMAIN:
        beta[3][h] = 1.0
    
    for t in [2, 1]:
        for h in DOMAIN:
            beta[t][h] = sum(
                beta[t+1][h_next] * t_factor(h, h_next) * lik[t+1][h_next]
                for h_next in DOMAIN
            )
    
    # Smoothed posteriors
    posteriors = []
    for t in [1, 2, 3]:
        post = {h: alpha[t][h] * beta[t][h] for h in DOMAIN}
        s = sum(post.values())
        post = {h: post[h] / s for h in DOMAIN} if s > 0 else {h: 1.0 / len(DOMAIN) for h in DOMAIN}
        posteriors.append(post)
    
    return alpha, beta, posteriors

def normalize_dict(d: Dict) -> Dict:
    """Normalize dictionary values to sum to 1"""
    s = sum(d.values())
    if s > 0:
        return {k: v / s for k, v in d.items()}
    return {k: 1.0 / len(d) for k in d}

# ============================================================================
# PART C: EXPLAINING AWAY (ALARM NETWORK)
# ============================================================================

def alarm_explaining_away(eps: float = 0.05):
    """
    Alarm network: B, E -> A where A = B OR E
    Compute P(B=1|A=1) and P(B=1|A=1,E=1) to show explaining away
    """
    # P(A=1) = P(B=1)P(E=0) + P(B=0)P(E=1) + P(B=1)P(E=1)
    #        = eps(1-eps) + (1-eps)eps + eps*eps
    #        = 2*eps - eps^2
    p_a1 = 2 * eps - eps * eps
    
    # P(B=1|A=1) = P(B=1,A=1) / P(A=1)
    #            = [eps(1-eps) + eps*eps] / P(A=1)
    #            = eps / P(A=1)
    p_b1_given_a1 = eps / p_a1
    
    # P(B=1|A=1,E=1) = P(B=1,E=1,A=1) / P(A=1,E=1)
    #                = P(B=1)P(E=1)P(A=1|B=1,E=1) / P(A=1,E=1)
    #                = eps * eps * 1 / [eps * 1]
    #                = eps
    p_b1_given_a1_e1 = eps
    
    return p_b1_given_a1, p_b1_given_a1_e1

# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_mrf_exact():
    """Experiment 1: Exact inference on MRF"""
    print("=" * 80)
    print("EXPERIMENT 1: MRF Exact Inference")
    print("=" * 80)
    
    Z, marginals, table, max_assign, max_w = enumerate_exact()
    
    print(f"\nPartition function Z: {Z:.4f}")
    
    print(f"\nAll positive-weight assignments:")
    print(f"{'Assignment':>15s} {'Weight':>10s} {'Probability':>12s}")
    print("-" * 80)
    for assignment, w in sorted(table, key=lambda x: x[1], reverse=True):
        prob = w / Z
        print(f"{str(assignment):>15s} {w:10.2f} {prob:12.4f}")
    
    print(f"\nMarginal Distributions:")
    for i in range(3):
        print(f"\nP(X{i+1}):")
        for v in DOMAIN:
            print(f"  X{i+1}={v}: {marginals[i][v]:.4f}")
    
    print(f"\nMAP (Maximum Weight) Assignment:")
    print(f"  Assignment: {max_assign}")
    print(f"  Weight: {max_w:.2f}")
    print(f"  Probability: {max_w/Z:.4f}")

def experiment_gibbs():
    """Experiment 2: Gibbs sampling convergence"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Gibbs Sampling")
    print("=" * 80)
    
    # Get exact marginals for comparison
    Z, exact_marginals, _, _, _ = enumerate_exact()
    
    # Run Gibbs with different numbers of iterations
    print(f"{'Iterations':>12s} ", end="")
    for v in DOMAIN:
        print(f"P(X2={v}):>12s{{}}", end="")
    print(f"{'Linf Error':>12s}")  # Using "Linf" instead of ∞ for compatibility
    print("-" * 80)
    
    for n_iters in [500, 1000, 2000, 5000]:
        marginals, _ = gibbs_sampling(n_iters=n_iters, burn_in=n_iters//5, seed=42)
        
        # L-infinity error for X2
        errors = [abs(marginals[1][v] - exact_marginals[1][v]) for v in DOMAIN]
        l_inf_error = max(errors)
        
        print(f"{n_iters:12d} ", end="")
        for v in DOMAIN:
            print(f"{marginals[1][v]:12.4f} ", end="")
        print(f"{l_inf_error:12.4f}")
    
    print(f"\nExact P(X2):     ", end="")
    for v in DOMAIN:
        print(f"{exact_marginals[1][v]:12.4f} ", end="")
    print()
    
    print("\nConclusion: Gibbs converges to exact marginals with sufficient iterations")

def experiment_bn_hmm():
    """Experiment 3: Bayesian Network / HMM inference"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Bayesian Network (HMM) Inference")
    print("=" * 80)
    
    evidence = {1: 0, 2: 2, 3: 2}
    alpha, beta, posteriors = forward_backward(evidence)
    
    print(f"\nEvidence: E = (0, 2, 2)")
    
    print(f"\nForward Messages (alpha):")
    for t in [1, 2, 3]:
        print(f"  alpha_{t}: {{0: {alpha[t][0]:.4f}, 1: {alpha[t][1]:.4f}, 2: {alpha[t][2]:.4f}}}")
    
    print(f"\nBackward Messages (beta):")
    for t in [1, 2, 3]:
        print(f"  beta_{t}: {{0: {beta[t][0]:.4f}, 1: {beta[t][1]:.4f}, 2: {beta[t][2]:.4f}}}")
    
    print(f"\nSmoothed Posteriors P(H_t | E):")
    for t, post in enumerate(posteriors, 1):
        print(f"\n  P(H_{t} | E):")
        for h in DOMAIN:
            print(f"    H_{t}={h}: {post[h]:.4f}")
    
    # Compare with MRF marginals
    print(f"\nComparison with MRF P(X2):")
    _, mrf_marginals, _, _, _ = enumerate_exact()
    print(f"  MRF:  {{0: {mrf_marginals[1][0]:.4f}, 1: {mrf_marginals[1][1]:.4f}, 2: {mrf_marginals[1][2]:.4f}}}")
    print(f"  HMM:  {{0: {posteriors[1][0]:.4f}, 1: {posteriors[1][1]:.4f}, 2: {posteriors[1][2]:.4f}}}")
    print(f"\nConclusion: MRF and HMM give identical results (same model)")

def experiment_explaining_away():
    """Experiment 4: Explaining away in alarm network"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Explaining Away (Alarm Network)")
    print("=" * 80)
    
    eps = 0.05
    p_b1_a1, p_b1_a1_e1 = alarm_explaining_away(eps)
    
    print(f"\nAlarm Network: B, E -> A (A = B OR E)")
    print(f"Prior: P(B=1) = P(E=1) = epsilon = {eps}")
    
    print(f"\nP(B=1 | A=1) = {p_b1_a1:.4f}")
    print(f"P(B=1 | A=1, E=1) = {p_b1_a1_e1:.4f}")
    
    ratio = p_b1_a1 / p_b1_a1_e1
    print(f"\nRatio: {ratio:.2f}x decrease")
    
    print(f"\nExplaining Away Interpretation:")
    print(f"  - Observing A=1 raises belief in B (from {eps:.3f} to {p_b1_a1:.3f})")
    print(f"  - But observing E=1 'explains away' the alarm")
    print(f"  - So P(B=1) drops back to prior level {p_b1_a1_e1:.3f}")
    print(f"  - This is characteristic of common-effect structures (V-structures)")

def plot_gibbs_convergence():
    iterations_list = [500, 1000, 2000, 5000]
    errors = []
    
    for n_iters in iterations_list:
        marginals, _ = gibbs_sampling(n_iters=n_iters, burn_in=n_iters//5, seed=42)
        Z, exact_marginals, _, _, _ = enumerate_exact()
        error = max([abs(marginals[1][v] - exact_marginals[1][v]) for v in DOMAIN])
        errors.append(error)
    
    plt.plot(iterations_list, errors, marker='o')
    plt.xlabel('Number of Iterations')
    plt.ylabel('L-infinity Error')
    plt.title('Gibbs Sampling Convergence')
    plt.grid(True)
    plt.savefig('gibbs_convergence.png')
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("WEEK 7-1: MARKOV RANDOM FIELDS AND BAYESIAN NETWORKS")
    print("=" * 80)
    
    experiment_mrf_exact()
    experiment_gibbs()
    experiment_bn_hmm()
    experiment_explaining_away()
    plot_gibbs_convergence()
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED")
    print("=" * 80)