"""
CS221 Week 8-1: A Tiny Propositional Logic Engine
Truth Tables, Satisfiability, Entailment, and Forward Chaining
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, Iterable, Tuple, List, Optional
import itertools

# =====================================================================
# PART 1: AST FOR PROPOSITIONAL LOGIC
# =====================================================================

@dataclass(frozen=True)
class Var:
    """Propositional variable (atom)"""
    name: str

@dataclass(frozen=True)
class Not:
    """Negation"""
    f: object

@dataclass(frozen=True)
class And:
    """Conjunction"""
    a: object
    b: object

@dataclass(frozen=True)
class Or:
    """Disjunction"""
    a: object
    b: object

@dataclass(frozen=True)
class Imp:
    """Implication (a → b)"""
    a: object
    b: object

@dataclass(frozen=True)
class Iff:
    """Biconditional (a ↔ b)"""
    a: object
    b: object


# =====================================================================
# PART 2: FORMULA UTILITIES
# =====================================================================

def atoms_in(f) -> Set[str]:
    """
    Extract all atom names from a formula.
    
    Args:
        f: Formula (AST node)
    
    Returns:
        Set of atom names (strings)
    """
    if isinstance(f, Var):
        return {f.name}
    
    if isinstance(f, Not):
        return atoms_in(f.f)
    
    if isinstance(f, (And, Or, Imp, Iff)):
        return atoms_in(f.a) | atoms_in(f.b)
    
    raise TypeError(f"Unknown node type: {type(f)}")


def eval_formula(f, w: Dict[str, int]) -> int:
    """
    Evaluate formula f in world w.
    
    Args:
        f: Formula (AST node)
        w: World (dictionary mapping atom names to 0 or 1)
    
    Returns:
        0 or 1 (False or True)
    """
    if isinstance(f, Var):
        return 1 if w.get(f.name, 0) else 0
    
    if isinstance(f, Not):
        return 1 - eval_formula(f.f, w)
    
    if isinstance(f, And):
        return eval_formula(f.a, w) & eval_formula(f.b, w)
    
    if isinstance(f, Or):
        return max(eval_formula(f.a, w), eval_formula(f.b, w))
    
    if isinstance(f, Imp):
        # A → B is false only when A is true and B is false
        ea = eval_formula(f.a, w)
        eb = eval_formula(f.b, w)
        return 1 if (ea == 0 or eb == 1) else 0
    
    if isinstance(f, Iff):
        # A ↔ B is true when A and B have the same truth value
        ea = eval_formula(f.a, w)
        eb = eval_formula(f.b, w)
        return 1 if ea == eb else 0
    
    raise TypeError(f"Unknown node type: {type(f)}")


# =====================================================================
# PART 3: SATISFIABILITY VIA TRUTH TABLES
# =====================================================================

def models_of_KB(KB: Iterable[object]) -> List[Dict[str, int]]:
    """
    Find all models (satisfying assignments) of KB using truth table enumeration.
    
    Args:
        KB: Knowledge base (list of formulas)
    
    Returns:
        List of models (each model is a dict mapping atom names to 0 or 1)
    """
    # Collect all atoms
    atoms = sorted(set().union(*[atoms_in(f) for f in KB])) if KB else []
    
    models = []
    
    # Enumerate all possible truth assignments
    for values in itertools.product([0, 1], repeat=len(atoms)):
        w = dict(zip(atoms, values))
        
        # Check if this assignment satisfies all formulas in KB
        if all(eval_formula(f, w) == 1 for f in KB):
            models.append(w)
    
    return models


def satisfiable(KB: Iterable[object]) -> Tuple[bool, Optional[Dict[str, int]]]:
    """
    Check if KB is satisfiable.
    
    Args:
        KB: Knowledge base (list of formulas)
    
    Returns:
        (is_satisfiable, model_or_None)
    """
    models = models_of_KB(KB)
    
    if models:
        return True, models[0]
    else:
        return False, None


# =====================================================================
# PART 4: ENTAILMENT VIA SAT
# =====================================================================

def entails(KB: Iterable[object], f) -> bool:
    """
    Check if KB entails f using the relationship:
        KB |= f  iff  KB ∪ {¬f} is UNSATISFIABLE
    
    Args:
        KB: Knowledge base (list of formulas)
        f: Query formula
    
    Returns:
        True if KB entails f, False otherwise
    """
    # Check if KB ∪ {¬f} is satisfiable
    sat, _ = satisfiable(list(KB) + [Not(f)])
    
    # KB entails f iff KB ∪ {¬f} is unsatisfiable
    return not sat


# =====================================================================
# PART 5: FORWARD CHAINING WITH MODUS PONENS
# =====================================================================

def forward_chain_modus_ponens(KB: Iterable[object]) -> Set[object]:
    """
    Apply forward chaining using only modus ponens.
    
    Modus Ponens: From p and (p → q), derive q.
    
    Args:
        KB: Knowledge base (list of formulas)
    
    Returns:
        Set of all derived formulas (including original KB)
    """
    derived = set(KB)
    changed = True
    
    while changed:
        changed = False
        
        # Get current formulas
        current_formulas = list(derived)
        
        # Look for implications
        for imp in current_formulas:
            if isinstance(imp, Imp):
                antecedent = imp.a
                consequent = imp.b
                
                # Check if antecedent is in derived formulas
                if antecedent in current_formulas and consequent not in derived:
                    # Apply modus ponens: derive consequent
                    derived.add(consequent)
                    changed = True
    
    return derived


# =====================================================================
# PART 6: HELPER FUNCTIONS FOR DISPLAY
# =====================================================================

def formula_to_string(f) -> str:
    """Convert formula to readable string"""
    if isinstance(f, Var):
        return f.name
    if isinstance(f, Not):
        return f"¬{formula_to_string(f.f)}"
    if isinstance(f, And):
        return f"({formula_to_string(f.a)} ∧ {formula_to_string(f.b)})"
    if isinstance(f, Or):
        return f"({formula_to_string(f.a)} ∨ {formula_to_string(f.b)})"
    if isinstance(f, Imp):
        return f"({formula_to_string(f.a)} → {formula_to_string(f.b)})"
    if isinstance(f, Iff):
        return f"({formula_to_string(f.a)} ↔ {formula_to_string(f.b)})"
    return str(f)


# =====================================================================
# MAIN EXPERIMENTS
# =====================================================================

def run_experiments():
    """Run all Week 8-1 experiments"""
    
    print("=" * 70)
    print("WEEK 8-1: PROPOSITIONAL LOGIC ENGINE")
    print("=" * 70)
    
    # Define atoms
    Rain = Var("Rain")
    Wet = Var("Wet")
    Slippery = Var("Slippery")
    Snow = Var("Snow")
    
    # Class KB: Rain, Rain → Wet, Wet → Slippery
    KB = [Rain, Imp(Rain, Wet), Imp(Wet, Slippery)]
    
    print("\nKnowledge Base:")
    for f in KB:
        print(f"  {formula_to_string(f)}")
    
    # ========== TASK 1: FORWARD CHAINING ==========
    print("\n" + "=" * 70)
    print("TASK 1: FORWARD CHAINING (Modus Ponens Only)")
    print("=" * 70)
    
    FC = forward_chain_modus_ponens(KB)
    
    print(f"\nDerived formulas ({len(FC)} total):")
    for f in FC:
        print(f"  {formula_to_string(f)}")
    
    # Show which are atoms (derived facts)
    derived_atoms = [f for f in FC if isinstance(f, Var)]
    print(f"\nDerived atoms: {[f.name for f in derived_atoms]}")
    
    # ========== TASK 2: ENTAILMENT VIA SAT ==========
    print("\n" + "=" * 70)
    print("TASK 2: ASK/TELL VIA SAT REASONING")
    print("=" * 70)
    
    # 2a) Is Wet entailed?
    print("\n2a) Is Wet entailed by KB?")
    wet_entailed = entails(KB, Wet)
    print(f"    Answer: {wet_entailed}")
    print(f"    Check: KB ∪ {{¬Wet}} satisfiable? {satisfiable([*KB, Not(Wet)])[0]}")
    print(f"    Therefore: KB |= Wet is {wet_entailed}")
    
    # 2b) Is Rain → Slippery entailed?
    print("\n2b) Is Rain → Slippery entailed by KB?")
    rain_slip_entailed = entails(KB, Imp(Rain, Slippery))
    print(f"    Answer: {rain_slip_entailed}")
    print(f"    This shows transitive implication is entailed")
    
    # 2c) Is ¬Rain contradictory with KB?
    print("\n2c) Is ¬Rain contradictory with KB?")
    not_rain_sat, _ = satisfiable([*KB, Not(Rain)])
    contradictory = not not_rain_sat
    print(f"    KB ∪ {{¬Rain}} satisfiable? {not_rain_sat}")
    print(f"    Therefore: ¬Rain is {'CONTRADICTORY' if contradictory else 'NOT contradictory'} with KB")
    print(f"    (Rain is explicitly asserted in KB)")
    
    # ========== TASK 3: CONTINGENCY ==========
    print("\n" + "=" * 70)
    print("TASK 3: CONTINGENCY ANALYSIS")
    print("=" * 70)
    
    print("\nIs Snow contingent w.r.t. KB?")
    
    # Check KB ∪ {Snow}
    snow_sat, snow_model = satisfiable([*KB, Snow])
    print(f"\n  KB ∪ {{Snow}} satisfiable? {snow_sat}")
    if snow_model:
        print(f"    Sample model: {snow_model}")
    
    # Check KB ∪ {¬Snow}
    not_snow_sat, not_snow_model = satisfiable([*KB, Not(Snow)])
    print(f"\n  KB ∪ {{¬Snow}} satisfiable? {not_snow_sat}")
    if not_snow_model:
        print(f"    Sample model: {not_snow_model}")
    
    is_contingent = snow_sat and not_snow_sat
    print(f"\n  Conclusion: Snow is {'CONTINGENT' if is_contingent else 'NOT contingent'}")
    print(f"  (Snow is neither entailed nor contradicted by KB)")
    
    # ========== TASK 4: MODELS OF KB ==========
    print("\n" + "=" * 70)
    print("TASK 4: MODELS OF KB")
    print("=" * 70)
    
    models = models_of_KB(KB)
    print(f"\nKB has {len(models)} model(s):")
    for i, model in enumerate(models, 1):
        print(f"  Model {i}: {model}")
    
    print("\nInterpretation:")
    print("  KB forces Rain=1, which forces Wet=1, which forces Slippery=1")
    print("  There is exactly one model satisfying all constraints")
    
    # ========== SOUNDNESS & COMPLETENESS ==========
    print("\n" + "=" * 70)
    print("SOUNDNESS & COMPLETENESS ANALYSIS")
    print("=" * 70)
    
    print("\nSOUNDNESS: Every derived formula is entailed by KB")
    print("-" * 70)
    
    for f in FC:
        if isinstance(f, Var):
            is_entailed = entails(KB, f)
            print(f"  {f.name}: derived={True}, entailed={is_entailed} {'✓' if is_entailed else '✗'}")
    
    print("\nAll derived atoms are entailed → Forward chaining is SOUND")
    
    print("\n\nINCOMPLETENESS: Modus ponens cannot derive all entailed formulas")
    print("-" * 70)
    
    # Example showing incompleteness
    print("\nExample: KB = {Rain, (Rain ∨ Snow) → Wet}")
    KB_incomplete = [Rain, Imp(Or(Rain, Snow), Wet)]
    
    print(f"  KB: {[formula_to_string(f) for f in KB_incomplete]}")
    
    # Semantic entailment
    wet_entailed_semantic = entails(KB_incomplete, Wet)
    print(f"\n  Is Wet entailed (semantically)? {wet_entailed_semantic}")
    
    # Forward chaining
    FC_incomplete = forward_chain_modus_ponens(KB_incomplete)
    wet_derived = Wet in FC_incomplete
    print(f"  Is Wet derived by modus ponens? {wet_derived}")
    
    print("\n  Analysis:")
    print("    - Semantically: Rain is true → (Rain ∨ Snow) is true → Wet is true")
    print("    - Syntactically: Modus ponens needs exact match of antecedent")
    print("    - It cannot match 'Rain' with '(Rain ∨ Snow)'")
    print("    - Therefore: Wet is ENTAILED but NOT DERIVED")
    print("\n  Conclusion: Modus ponens alone is INCOMPLETE for propositional logic")
    
    # ========== COMPLEXITY ANALYSIS ==========
    print("\n" + "=" * 70)
    print("COMPLEXITY ANALYSIS")
    print("=" * 70)
    
    print("\nTruth Table Method:")
    print(f"  - Time: O(2^n × m) where n = # atoms, m = |KB|")
    print(f"  - Space: O(n + m)")
    print(f"  - Complete: YES (checks all possibilities)")
    print(f"  - For this KB: n=3 atoms → 2^3 = 8 truth assignments checked")
    
    print("\nForward Chaining (Modus Ponens):")
    print(f"  - Time: O(k × |KB|^2) where k = # iterations")
    print(f"  - Space: O(|KB|)")
    print(f"  - Complete: NO (only handles certain patterns)")
    print(f"  - For this KB: 2 iterations, derived 5 formulas from 3")
    
    print("\n" + "=" * 70)
    print("✓ Week 8-1 experiments completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_experiments()