"""
Week 7-3: Learning Bayesian Networks with MLE, Laplace Smoothing, and EM
Movie Ratings Example: G → R1, R2
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math

# ============================================================================
# PROBLEM SETUP
# ============================================================================

G_VALS = ["c", "d"]  # comedy, drama
R_VALS = [1, 2, 3, 4, 5]  # ratings 1-5

# Type alias
Example = Tuple[Optional[str], int, int]  # (G or None, R1, R2)

# Data
SUPERVISED = [
    ("d", 4, 5),
    ("d", 4, 4),
    ("d", 5, 3),
    ("c", 1, 2),
    ("c", 5, 4),
]

UNSUPERVISED = [
    (None, 2, 2),
    (None, 1, 2),
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_dict(d: Dict) -> Dict:
    """Normalize dictionary values to sum to 1"""
    s = sum(d.values())
    if s == 0:
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: d[k] / s for k in d}

# ============================================================================
# MLE WITH PARAMETER SHARING
# ============================================================================

def fit_mle(supervised: List[Example], lambda_: float = 0.0, 
            share_R: bool = True) -> Tuple[Dict, Dict]:
    """
    Fit BN using MLE with optional Laplace smoothing.
    
    Args:
        supervised: Fully observed examples
        lambda_: Laplace smoothing parameter
        share_R: If True, use p_R(·|g) for both R1 and R2
    
    Returns:
        (pG, pR) where pG is dict and pR is dict g -> {r: prob}
    """
    # Initialize counts with Laplace pseudocounts
    count_G = {g: lambda_ for g in G_VALS}
    
    if share_R:
        count_R = {g: {r: lambda_ for r in R_VALS} for g in G_VALS}
    else:
        count_R1 = {g: {r: lambda_ for r in R_VALS} for g in G_VALS}
        count_R2 = {g: {r: lambda_ for r in R_VALS} for g in G_VALS}
    
    # Count observed data
    for g, r1, r2 in supervised:
        assert g is not None, "fit_mle requires fully observed data"
        count_G[g] += 1
        
        if share_R:
            count_R[g][r1] += 1
            count_R[g][r2] += 1
        else:
            count_R1[g][r1] += 1
            count_R2[g][r2] += 1
    
    # Normalize to probabilities
    pG = normalize_dict(count_G)
    
    if share_R:
        pR = {g: normalize_dict(count_R[g]) for g in G_VALS}
    else:
        pR = {
            "R1": {g: normalize_dict(count_R1[g]) for g in G_VALS},
            "R2": {g: normalize_dict(count_R2[g]) for g in G_VALS}
        }
    
    return pG, pR

# ============================================================================
# LOG-LIKELIHOOD
# ============================================================================

def log_likelihood(mixed: List[Example], pG: Dict, pR: Dict, 
                   share_R: bool = True) -> float:
    """Compute log-likelihood of data under current parameters"""
    ll = 0.0
    
    for g_obs, r1, r2 in mixed:
        if g_obs is None:
            # Marginalize over G
            prob = 0.0
            for g in G_VALS:
                if share_R:
                    prob += pG[g] * pR[g][r1] * pR[g][r2]
                else:
                    prob += pG[g] * pR["R1"][g][r1] * pR["R2"][g][r2]
            ll += math.log(max(prob, 1e-12))
        else:
            # G is observed
            g = g_obs
            if share_R:
                prob = pG[g] * pR[g][r1] * pR[g][r2]
            else:
                prob = pG[g] * pR["R1"][g][r1] * pR["R2"][g][r2]
            ll += math.log(max(prob, 1e-12))
    
    return ll

# ============================================================================
# EM ALGORITHM
# ============================================================================

def e_step(mixed: List[Example], pG: Dict, pR: Dict, 
           share_R: bool = True) -> List[Dict]:
    """
    E-step: Compute posterior P(G | R1, R2) for each example.
    
    Returns: List of posterior distributions (one per example)
    """
    posteriors = []
    
    for g_obs, r1, r2 in mixed:
        if g_obs is not None:
            # G is observed - posterior is deterministic
            post = {g: 1.0 if g == g_obs else 0.0 for g in G_VALS}
        else:
            # G is missing - compute posterior
            unnorm = {}
            for g in G_VALS:
                if share_R:
                    unnorm[g] = pG[g] * pR[g][r1] * pR[g][r2]
                else:
                    unnorm[g] = pG[g] * pR["R1"][g][r1] * pR["R2"][g][r2]
            
            post = normalize_dict(unnorm)
        
        posteriors.append(post)
    
    return posteriors

def m_step(mixed: List[Example], posteriors: List[Dict], 
           lambda_: float = 0.0, share_R: bool = True) -> Tuple[Dict, Dict]:
    """
    M-step: Update parameters using fractional counts from posteriors.
    """
    # Initialize with Laplace pseudocounts
    count_G = {g: lambda_ for g in G_VALS}
    
    if share_R:
        count_R = {g: {r: lambda_ for r in R_VALS} for g in G_VALS}
    else:
        count_R1 = {g: {r: lambda_ for r in R_VALS} for g in G_VALS}
        count_R2 = {g: {r: lambda_ for r in R_VALS} for g in G_VALS}
    
    # Add fractional counts weighted by posteriors
    for (g_obs, r1, r2), post in zip(mixed, posteriors):
        for g in G_VALS:
            weight = post[g]
            count_G[g] += weight
            
            if share_R:
                count_R[g][r1] += weight
                count_R[g][r2] += weight
            else:
                count_R1[g][r1] += weight
                count_R2[g][r2] += weight
    
    # Normalize
    pG = normalize_dict(count_G)
    
    if share_R:
        pR = {g: normalize_dict(count_R[g]) for g in G_VALS}
    else:
        pR = {
            "R1": {g: normalize_dict(count_R1[g]) for g in G_VALS},
            "R2": {g: normalize_dict(count_R2[g]) for g in G_VALS}
        }
    
    return pG, pR

def fit_em(mixed: List[Example], init: Optional[Tuple] = None, 
           lambda_: float = 0.0, n_iters: int = 10, 
           share_R: bool = True) -> Tuple[Tuple[Dict, Dict], List[float]]:
    """
    Fit BN using EM algorithm.
    
    Returns:
        ((pG, pR), ll_history)
    """
    if init is None:
        # Uniform initialization
        pG = {g: 1.0 / len(G_VALS) for g in G_VALS}
        if share_R:
            pR = {g: {r: 1.0 / len(R_VALS) for r in R_VALS} for g in G_VALS}
        else:
            pR = {
                "R1": {g: {r: 1.0 / len(R_VALS) for r in R_VALS} for g in G_VALS},
                "R2": {g: {r: 1.0 / len(R_VALS) for r in R_VALS} for g in G_VALS}
            }
    else:
        pG, pR = init
    
    ll_history = [log_likelihood(mixed, pG, pR, share_R=share_R)]
    
    for iteration in range(n_iters):
        # E-step
        posteriors = e_step(mixed, pG, pR, share_R=share_R)
        
        # M-step
        pG, pR = m_step(mixed, posteriors, lambda_=lambda_, share_R=share_R)
        
        # Record likelihood
        ll = log_likelihood(mixed, pG, pR, share_R=share_R)
        ll_history.append(ll)
    
    return (pG, pR), ll_history

# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_mle():
    """Experiment 1: MLE on supervised data"""
    print("=" * 80)
    print("EXPERIMENT 1: Maximum Likelihood Estimation")
    print("=" * 80)
    
    print(f"\nSupervised Data (N={len(SUPERVISED)}):")
    for i, (g, r1, r2) in enumerate(SUPERVISED, 1):
        print(f"  {i}. G={g}, R1={r1}, R2={r2}")
    
    # MLE without smoothing
    pG, pR = fit_mle(SUPERVISED, lambda_=0.0, share_R=True)
    
    print(f"\nMLE P(G):")
    for g in G_VALS:
        print(f"  P(G={g}) = {pG[g]:.4f}")
    
    print(f"\nMLE P(R|G) with parameter sharing:")
    for g in G_VALS:
        print(f"  P(R|G={g}):")
        for r in R_VALS:
            print(f"    R={r}: {pR[g][r]:.4f}")
    
    # Identify zero probabilities
    print(f"\nZero Probabilities (will cause issues with unsupervised data):")
    for g in G_VALS:
        zero_ratings = [r for r in R_VALS if pR[g][r] == 0.0]
        if zero_ratings:
            print(f"  P(R={zero_ratings}|G={g}) = 0")

def experiment_laplace():
    """Experiment 2: Laplace smoothing"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Laplace Smoothing")
    print("=" * 80)
    
    print(f"\n{'λ':>6s} {'Zero→Pos':>12s} {'Example Changes':>30s}")
    print("-" * 80)
    
    # MLE baseline
    pG_mle, pR_mle = fit_mle(SUPERVISED, lambda_=0.0, share_R=True)
    
    for lambda_ in [0.0, 0.5, 1.0, 2.0]:
        pG_lap, pR_lap = fit_mle(SUPERVISED, lambda_=lambda_, share_R=True)
        
        # Count zero→positive transitions
        flips = 0
        examples = []
        for g in G_VALS:
            for r in R_VALS:
                if pR_mle[g][r] == 0.0 and pR_lap[g][r] > 0.0:
                    flips += 1
                    if len(examples) < 2:
                        examples.append(f"P(R={r}|G={g})")
        
        ex_str = ", ".join(examples) if examples else "None"
        print(f"{lambda_:6.1f} {flips:12d} {ex_str:>30s}")
    
    # Detailed view for λ=1
    print(f"\nDetailed Comparison (λ=0 vs λ=1):")
    pG_0, pR_0 = fit_mle(SUPERVISED, lambda_=0.0, share_R=True)
    pG_1, pR_1 = fit_mle(SUPERVISED, lambda_=1.0, share_R=True)
    
    for g in G_VALS:
        print(f"\n  P(R|G={g}):")
        print(f"    {'R':>3s} {'λ=0':>10s} {'λ=1':>10s} {'Change':>10s}")
        for r in R_VALS:
            change = pR_1[g][r] - pR_0[g][r]
            print(f"    {r:3d} {pR_0[g][r]:10.4f} {pR_1[g][r]:10.4f} {change:+10.4f}")

def experiment_em():
    """Experiment 3: EM algorithm"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: EM Algorithm")
    print("=" * 80)
    
    mixed = SUPERVISED + UNSUPERVISED
    
    print(f"\nMixed Data (N={len(mixed)}):")
    print(f"  Supervised:   {len(SUPERVISED)} examples")
    print(f"  Unsupervised: {len(UNSUPERVISED)} examples")
    for i, (g, r1, r2) in enumerate(UNSUPERVISED, 1):
        print(f"    {i}. G=?, R1={r1}, R2={r2}")
    
    # Initialize with MLE on supervised data + Laplace
    pG_init, pR_init = fit_mle(SUPERVISED, lambda_=1.0, share_R=True)
    
    print(f"\nInitialization (MLE + Laplace λ=1):")
    print(f"  P(G): {pG_init}")
    
    # Run EM
    (pG_em, pR_em), ll_history = fit_em(
        mixed, init=(pG_init, pR_init), lambda_=1.0, n_iters=10, share_R=True
    )
    
    print(f"\n{'Iter':>6s} {'Log-Likelihood':>16s} {'ΔLL':>12s}")
    print("-" * 80)
    for i, ll in enumerate(ll_history):
        delta = ll - ll_history[i-1] if i > 0 else 0.0
        print(f"{i:6d} {ll:16.4f} {delta:+12.4f}")
    
    print(f"\nFinal Parameters after EM:")
    print(f"  P(G):")
    for g in G_VALS:
        print(f"    G={g}: {pG_em[g]:.4f}")
    
    print(f"\n  P(R|G):")
    for g in G_VALS:
        print(f"    P(R|G={g}): {[f'{pR_em[g][r]:.3f}' for r in R_VALS]}")

def experiment_posteriors():
    """Experiment 4: Posterior inference on unsupervised data"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Posterior Inference P(G | R1, R2)")
    print("=" * 80)
    
    # Get EM-trained parameters
    mixed = SUPERVISED + UNSUPERVISED
    pG_init, pR_init = fit_mle(SUPERVISED, lambda_=1.0, share_R=True)
    (pG_em, pR_em), _ = fit_em(mixed, init=(pG_init, pR_init), 
                                lambda_=1.0, n_iters=10, share_R=True)
    
    print(f"\nPosteriors for Unsupervised Examples:")
    
    for i, (_, r1, r2) in enumerate(UNSUPERVISED, 1):
        # Compute posterior
        unnorm = {}
        for g in G_VALS:
            unnorm[g] = pG_em[g] * pR_em[g][r1] * pR_em[g][r2]
        
        post = normalize_dict(unnorm)
        
        print(f"\n  Example {i}: R1={r1}, R2={r2}")
        for g in G_VALS:
            print(f"    P(G={g} | R1={r1}, R2={r2}) = {post[g]:.4f}")
        
        # Interpretation
        likely = max(G_VALS, key=lambda g: post[g])
        print(f"    → Most likely: G={likely}")

def experiment_lambda_sweep():
    """Experiment 5: Effect of λ on EM"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Laplace Smoothing Effect on EM")
    print("=" * 80)
    
    mixed = SUPERVISED + UNSUPERVISED
    
    print(f"\n{'λ':>6s} {'Final LL':>12s} {'P(G=c)':>10s} {'P(G=d)':>10s}")
    print("-" * 80)
    
    for lambda_ in [0.0, 0.5, 1.0, 2.0, 5.0]:
        pG_init, pR_init = fit_mle(SUPERVISED, lambda_=lambda_, share_R=True)
        (pG_em, pR_em), ll_hist = fit_em(
            mixed, init=(pG_init, pR_init), lambda_=lambda_, 
            n_iters=10, share_R=True
        )
        
        print(f"{lambda_:6.1f} {ll_hist[-1]:12.4f} {pG_em['c']:10.4f} {pG_em['d']:10.4f}")
    
    print("\nConclusion: Higher λ provides more regularization")
    print("            but may prevent overfitting to small data")

def experiment_convergence():
    """Experiment 6: EM convergence analysis"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: EM Convergence Analysis")
    print("=" * 80)
    
    mixed = SUPERVISED + UNSUPERVISED
    pG_init, pR_init = fit_mle(SUPERVISED, lambda_=1.0, share_R=True)
    
    # Run EM for more iterations
    (pG_em, pR_em), ll_history = fit_em(
        mixed, init=(pG_init, pR_init), lambda_=1.0, 
        n_iters=20, share_R=True
    )
    
    print(f"\nEM Convergence:")
    print(f"{'Iter':>6s} {'Log-Likelihood':>16s} {'ΔLL':>12s} {'Status':>15s}")
    print("-" * 80)
    
    for i, ll in enumerate(ll_history):
        delta = ll - ll_history[i-1] if i > 0 else 0.0
        
        if i == 0:
            status = "Initial"
        elif abs(delta) < 1e-4:
            status = "Converged"
        else:
            status = "Improving"
        
        print(f"{i:6d} {ll:16.6f} {delta:+12.6f} {status:>15s}")
    
    print("\nConclusion: EM converges when ΔLL < threshold (e.g., 1e-4)")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("WEEK 7-3: LEARNING BAYESIAN NETWORKS")
    print("=" * 80)
    
    experiment_mle()
    experiment_laplace()
    experiment_em()
    experiment_posteriors()
    experiment_lambda_sweep()
    experiment_convergence()
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED")
    print("=" * 80)