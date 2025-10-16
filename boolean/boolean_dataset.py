import argparse
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
import json
import numpy as np
from itertools import product, combinations
from datetime import datetime
import sympy as sp
from sympy import symbols, Not, And, Or, Xor, to_cnf
from sympy import evaluate

# Constants for mechanistic key function
COMMUTATIVE_OPS = {"AND", "OR", "XOR", "NAND", "NOR"}
IDEMPOTENT_OPS = {"AND", "OR"}

def _contains_constant(expr):
    """Return True if expr or any subexpr is True/False constant."""
    if expr in (sp.true, sp.false):
        return True
    return any(_contains_constant(arg) for arg in getattr(expr, "args", ()))

def _is_leaf(expr: sp.Expr) -> bool:
    """Check if expression is a leaf node (variable or constant)."""
    return isinstance(expr, sp.Symbol) or (expr in (sp.true, sp.false)) or isinstance(expr, (int, bool))

def _op_name(expr: sp.Expr) -> str:
    # detect NOR first (Not(Or(...)))
    if isinstance(expr, sp.Not) and isinstance(expr.args[0], sp.Or):
        return "NOR"
    if isinstance(expr, sp.Not):
        return "NOT"
    elif isinstance(expr, sp.And):
        return "AND"
    elif isinstance(expr, sp.Or):
        return "OR"
    elif isinstance(expr, sp.Xor):
        return "XOR"
    else:
        return "UNKNOWN"

def mechanistic_key_sympy(
    expr: sp.Expr,
    *,
    apply_commutativity: bool = True,
    apply_idempotence_and_or: bool = True,
    flatten_associativity: bool = True
) -> Tuple:
    """
    Canonicalize expr using biology-aware symmetries ONLY:
      - commutativity (AND/OR/XOR) (local)
      - optional idempotence (AND/OR)
      - optional associativity flattening (on by default in CLI, flat=True)
    NO absorption/distributivity/De Morgan.
    Parental swap is intentionally omitted for (x,y) because it adds nothing beyond commutativity.
    """

    def to_tuple(e: sp.Expr) -> Tuple:
        if _is_leaf(e):
            if isinstance(e, sp.Symbol):
                return ("VAR", str(e))
            return ("CONST", bool(e))  # BooleanTrue/False if any appear

        name = _op_name(e)

        if name == "NOT":
            return ("NOT", to_tuple(e.args[0]))
        

        if name == "NOR":
            # e is Not( Or(a,b,...) )
            or_node = e.args[0]
            kids = [to_tuple(arg) for arg in or_node.args]
            if apply_commutativity:
                kids = sorted(kids)
            return ("NOR", tuple(kids))

        # children (SymPy keeps args as-is when evaluate=False)
        kids = [to_tuple(arg) for arg in e.args]

        # Optional associativity flattening (off by default): turn cascades into multi-input gate
        if flatten_associativity and name in (IDEMPOTENT_OPS | {"XOR"}):
            flat = []
            for k in kids:
                if isinstance(k, tuple) and k and k[0] == name:
                    flat.extend(k[1:])
                else:
                    flat.append(k)
            kids = flat

        # Commutativity: sort children at commutative ops
        if apply_commutativity and name in COMMUTATIVE_OPS:
            kids = sorted(kids)

        # Idempotence (optional) for AND/OR: remove exact duplicates
        if apply_idempotence_and_or and name in IDEMPOTENT_OPS:
            uniq = []
            for k in kids:
                if not uniq or uniq[-1] != k:
                    uniq.append(k)
            kids = uniq


        return (name, tuple(kids))

    return to_tuple(expr)

class BooleanExpression:
    """Represents a Boolean expression with its symbolic formula."""
    def __init__(self, formula: str, variables: List[str], operators: Set[str]):
        self.formula = formula
        self.variables = sorted(variables)   # fixed to ['x','y'] in this script
        self.operators = operators
        self.sympy_expr = self._parse_formula(formula)
        self.truth_table = self._compute_truth_table()

    def _parse_formula(self, formula: str):
        """Parse to SymPy without auto-simplification."""
        var_symbols = {v: symbols(v) for v in self.variables}

        # NOR(a,b,...)  ->  ~(a | b | ...)
        def _nor_repl(m):
            inner = m.group(1)
            parts = [p.strip() for p in inner.split(',')]
            return "~(" + " | ".join(parts) + ")"

        formula = re.sub(r'\bNOR\s*\(([^)]*)\)', _nor_repl, formula, flags=re.IGNORECASE)

        parsed = (formula
                .replace('AND', '&')
                .replace('OR', '|')
                .replace('NOT', '~')
                .replace('XOR', '^'))
        try:
            with evaluate(False):
                expr = sp.sympify(parsed, locals=var_symbols, evaluate=False)
            return expr
        except Exception:
            return None

    def _compute_truth_table(self) -> Dict[Tuple[int, ...], int]:
        truth_table = {}
        if self.sympy_expr is None:
            return truth_table
        syms = [symbols(v) for v in self.variables]
        for values in product([0, 1], repeat=len(self.variables)):
            subs = {s: val for s, val in zip(syms, values)}
            try:
                res = self.sympy_expr.subs(subs)
                res = int(bool(res))
            except Exception:
                res = 0
            truth_table[values] = res
        return truth_table

    def evaluate(self, inputs: Dict[str, int]) -> int:
        values = tuple(inputs[var] for var in self.variables)
        return self.truth_table.get(values, 0)

    def to_string(self) -> str:
        return self.formula

    def get_canonical_form(self) -> str:
        """CNF for display only (semantic), not used for dedup."""
        if self.sympy_expr is not None:
            try:
                return str(to_cnf(self.sympy_expr, simplify=False))
            except Exception:
                pass
        return self.formula

    def mechanistic_key(
        self,
        *,
        apply_commutativity: bool = True,
        apply_idempotence_and_or: bool = True,
        flatten_associativity: bool = True
    ) -> Tuple:
        return mechanistic_key_sympy(
            self.sympy_expr,
            apply_commutativity=apply_commutativity,
            apply_idempotence_and_or=apply_idempotence_and_or,
            flatten_associativity=flatten_associativity
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, BooleanExpression):
            return False
        return self.truth_table == other.truth_table

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.truth_table.items())))

class BooleanObservation:
    """Represents a Boolean input-output observation."""
    def __init__(self, inputs: Dict[str, int], output: int):
        self.inputs = inputs
        self.output = output

    def to_tuple(self) -> Tuple:
        sorted_inputs = tuple(sorted(self.inputs.items()))
        return (sorted_inputs, self.output)

    def to_string(self) -> str:
        s = ", ".join([f"{k}={v}" for k, v in sorted(self.inputs.items())])
        return f"({s}) -> {self.output}"

    def __eq__(self, other) -> bool:
        return self.to_tuple() == other.to_tuple()

    def __hash__(self) -> int:
        return hash(self.to_tuple())

class BooleanDiscoveryGame:
    """Task for discovering Boolean functions from partial observations (x from mother, y from father)."""

    OPERATOR_SETS = {
        'basic': {'AND', 'OR'},
        'extended': {'AND', 'OR', 'NOT'},
        # 'full': {'AND', 'OR', 'NOT', 'XOR'}
        'full': {'AND', 'OR', 'NOT', 'NOR'}
    }

    # -------- Random expression (kept for completeness) --------
    @staticmethod
    def generate_random_expression(
        variables: List[str],
        operators: Set[str],
        max_depth: int = 2,
        rng: Optional[np.random.Generator] = None
    ) -> BooleanExpression:
        if rng is None:
            rng = np.random.default_rng()
        var_symbols = [symbols(v) for v in variables]

        def gen(depth: int):
            if depth >= max_depth or rng.random() < 0.3:
                return rng.choice(var_symbols)
            op = rng.choice(list(operators))
            with evaluate(False):
                if op == 'NOT':
                    child = gen(depth + 1)
                    return Not(child, evaluate=False)
                left = gen(depth + 1)
                right = gen(depth + 1)
                if op == 'AND':
                    return And(left, right, evaluate=False)
                if op == 'OR':
                    return Or(left, right, evaluate=False)
                if op == 'XOR':
                    return Xor(left, right, evaluate=False)
                if op == 'NOR':
                    left = gen(depth + 1)
                    right = gen(depth + 1)
                    return Not(Or(left, right, evaluate=False), evaluate=False)
                # Fallback: return left if op unsupported
                return left

        sym = gen(0)
        formula_str = (str(sym)
               .replace('&', ' AND ')
               .replace('|', ' OR ')
               .replace('~', 'NOT ')
               .replace('^', ' XOR '))
        # NOR pretty print
        if isinstance(sym, sp.Not) and isinstance(sym.args[0], sp.Or):
            args_str = " , ".join(str(arg) for arg in sym.args[0].args)
            formula_str = f"NOR({args_str})"

        expr = BooleanExpression.__new__(BooleanExpression)
        expr.formula = formula_str
        expr.variables = sorted(variables)
        expr.operators = operators
        expr.sympy_expr = sym
        expr.truth_table = BooleanExpression._compute_truth_table(expr)
        return expr

    # -------- Exhaustive enumeration with mechanistic dedup --------
    @staticmethod
    def generate_all_expressions(
        variables: List[str],
        operators: Set[str],
        max_depth: int,
        mechanistic_opts: Dict[str, Any]
    ) -> List[BooleanExpression]:
        var_symbols = {v: symbols(v) for v in variables}

        def rec(depth: int) -> List[sp.Expr]:
            if depth == 0:
                return list(var_symbols.values())
            prev = rec(depth - 1)
            exprs = list(prev)  # include lower depths
            with evaluate(False):
                if 'NOT' in operators:
                    exprs += [Not(e, evaluate=False) for e in prev]
                if 'AND' in operators:
                    exprs += [And(a, b, evaluate=False) for a in prev for b in prev]
                if 'OR' in operators:
                    exprs += [Or(a, b, evaluate=False) for a in prev for b in prev]
                if 'XOR' in operators:
                    exprs += [Xor(a, b, evaluate=False) for a in prev for b in prev]
                if 'NOR' in operators:
                    exprs += [Not(Or(a, b, evaluate=False), evaluate=False) for a in prev for b in prev]
            return exprs

        sym_exprs = rec(max_depth)

        seen_keys: Set[Tuple] = set()
        results: List[BooleanExpression] = []
        for s in sym_exprs:
            if _contains_constant(s):
                continue
            be = BooleanExpression.__new__(BooleanExpression)
            be.formula = (str(s)
                          .replace('&', ' AND ')
                          .replace('|', ' OR ')
                          .replace('~', 'NOT ')
                          .replace('^', ' XOR '))
            be.variables = sorted(variables)
            be.operators = operators
            be.sympy_expr = s
            be.truth_table = BooleanExpression._compute_truth_table(be)
            
            # Filter out constant expressions (those that don't depend on variables)
            # Check if all truth table values are the same (constant True or False)
            if be.sympy_expr in (sp.true, sp.false):
                continue
            
            key = be.mechanistic_key(**mechanistic_opts)
            if key not in seen_keys:
                seen_keys.add(key)
                results.append(be)
        return results

    # -------- Observations from expressions (consistent across set) --------
    @staticmethod
    def generate_observations_from_expressions(
        expressions: List[BooleanExpression],
        n_observations: int,
        rng: Optional[np.random.Generator] = None
    ) -> List[BooleanObservation]:
        if rng is None:
            rng = np.random.default_rng()
        if not expressions:
            return []

        variables = expressions[0].variables
        all_inputs = list(product([0, 1], repeat=len(variables)))
        consistent: List[BooleanObservation] = []

        for input_values in all_inputs:
            inputs_dict = {var: val for var, val in zip(variables, input_values)}
            outs = [e.evaluate(inputs_dict) for e in expressions]
            if len(set(outs)) == 1:
                consistent.append(BooleanObservation(inputs_dict, outs[0]))

        if len(consistent) <= n_observations:
            return consistent

        idx = rng.choice(len(consistent), size=n_observations, replace=False)
        return [consistent[i] for i in idx]

    # -------- Filter compatible expressions --------
    @staticmethod
    def find_all_compatible_expressions(
        observations: List[BooleanObservation],
        variables: List[str],
        operators: Set[str],
        max_depth: int,
        mechanistic_opts: Dict[str, Any],
        pregenerated_exprs: Optional[List[BooleanExpression]] = None
    ) -> List[BooleanExpression]:
        if pregenerated_exprs is None:
            all_exprs = BooleanDiscoveryGame.generate_all_expressions(
                variables, operators, max_depth, mechanistic_opts
            )
        else:
            all_exprs = pregenerated_exprs
        compatible: List[BooleanExpression] = []
        for e in all_exprs:
            ok = True
            for obs in observations:
                if e.evaluate(obs.inputs) != obs.output:
                    ok = False
                    break
            if ok:
                compatible.append(e)
        return compatible

    # -------- Dataset generation (x,y only) --------
    @staticmethod
    def generate_game_dataset(
        n_observations: Optional[int] = None,
        operator_set: str = 'full',
        max_depth: int = 2,
        seed: Optional[int] = None,
        k_ground_truth: int = 3,
        mechanistic_opts: Optional[Dict[str, Any]] = None
    ) -> Dict:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        variables = ['x', 'y']
        operators = BooleanDiscoveryGame.OPERATOR_SETS.get(operator_set, {'AND', 'OR', 'NOT'})

        # Default: leave at least one input unobserved if possible
        if n_observations is None:
            n_observations = min(2 ** len(variables) - 1, 3)

        print("-" * 50)
        print("Generating Boolean Discovery Dataset (x: mother, y: father)")
        print("-" * 50)
        print(f"Variables: {', '.join(variables)}")
        print(f"Operators: {operators}")
        print(f"Max depth: {max_depth}")
        print(f"Target observations: {n_observations}")

        if mechanistic_opts is None:
            mechanistic_opts = dict(
                apply_commutativity=True,
                apply_idempotence_and_or=True,
                flatten_associativity=True
            )

        # Build a diverse pool of semantic-unique candidates to choose GTs from
        # (We use the semantic check here just to diversify the seed set.)
        candidates = BooleanDiscoveryGame.generate_all_expressions(
            variables, operators, max_depth, mechanistic_opts
        )
        print(f"Total number of Hypothesis Space: {len(candidates)}")

        # de-duplicate semantically to ensure different functions in candidates
        sem_seen: Set[Tuple] = set()
        gt_candidates: List[BooleanExpression] = []
        for e in candidates:
            tt = tuple(sorted(e.truth_table.items()))
            if tt not in sem_seen:
                sem_seen.add(tt)
                gt_candidates.append(e)
        
        
        # Pick k functions; generate observations consistent across all of them
        best_exprs: List[BooleanExpression] = []
        best_obs: List[BooleanObservation] = []
        attempts = 100

        for _ in range(attempts):
            if len(gt_candidates) == 0:
                break
            take = min(k_ground_truth, len(gt_candidates))
            idx = rng.choice(len(gt_candidates), size=take, replace=False)
            chosen = [gt_candidates[i] for i in idx]
            obs = BooleanDiscoveryGame.generate_observations_from_expressions(chosen, n_observations, rng)
            if len(obs) >= n_observations:
                best_exprs, best_obs = chosen, obs
                break
            if len(best_obs) < len(obs):
                best_exprs, best_obs = chosen, obs

        if not best_obs:
            # Fallback: pick one GT and generate observations from it
            e = rng.choice(gt_candidates) if gt_candidates else BooleanDiscoveryGame.generate_random_expression(
                variables, operators, max_depth, rng
            )
            all_inputs = list(product([0, 1], repeat=len(variables)))
            obs_list = []
            for inputs in all_inputs[:n_observations]:
                inputs_dict = {v: val for v, val in zip(variables, inputs)}
                obs_list.append(BooleanObservation(inputs_dict, e.evaluate(inputs_dict)))
            best_exprs = [e]
            best_obs = obs_list

        print(f"Selected {len(best_exprs)} ground truth candidates")
        print(f"Generated {len(best_obs)} observations")

        print("Finding all compatible expressions (mechanistic dedup)...")
        all_compatible = BooleanDiscoveryGame.find_all_compatible_expressions(
            best_obs, variables, operators, max_depth, mechanistic_opts,
            pregenerated_exprs=candidates
        )
        print(f"Found {len(all_compatible)} compatible expressions")

        # Serialize
        obs_out = [{"inputs": o.inputs, "output": o.output, "string": o.to_string()} for o in best_obs]
        gt_out = [{
            "formula": e.formula,
            "canonical_form": e.get_canonical_form(),
            "truth_table": {str(k): v for k, v in e.truth_table.items()}
        } for e in all_compatible]

        return {
            "ground_truth_expressions": gt_out,
            "observations": obs_out,
            "variables": variables,
            "operators": list(operators),
            "max_depth": max_depth,
            "n_ground_truths": len(all_compatible),
            "n_hypothesis_space": len(candidates),
            "metadata": {
                "description": "Boolean function discovery dataset (x: mother, y: father)",
                "dedup": "mechanistic",
                "mechanistic_opts": mechanistic_opts,
                "note": "Parental swap symmetry omitted (redundant with only two variables)."
            }
        }


class BooleanDatasetGenerator:
    """Generate datasets with all possible observation combinations."""
    
    @staticmethod
    def generate_all_observation_combinations(
        variables: List[str],
        operators: Set[str], 
        max_depth: int,
        n_observations: int,
        mechanistic_opts: Dict[str, Any],
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate all possible datasets with exactly n_observations.
        
        Args:
            variables: List of variable names (e.g., ['x', 'y'])
            operators: Set of allowed operators
            max_depth: Maximum expression depth
            n_observations: Number of observations to include (1, 2, 3, or 4)
            mechanistic_opts: Options for mechanistic deduplication
            seed: Random seed for reproducibility
            
        Returns:
            List of dataset dictionaries, each with a unique observation combination
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
            
        # Generate all expressions
        all_exprs = BooleanDiscoveryGame.generate_all_expressions(
            variables, operators, max_depth, mechanistic_opts
        )
        
        # Generate all possible inputs for 2 variables
        all_inputs = list(product([0, 1], repeat=len(variables)))
        
        # Generate all combinations of n_observations from 4 possible inputs
        observation_combinations = list(combinations(all_inputs, n_observations))
        
        datasets = []
        
        for obs_combo in observation_combinations:
            # For each combination of inputs, try all possible output assignments
            # With n observations, there are 2^n possible output assignments
            for output_assignment in product([0, 1], repeat=n_observations):
                # Create observations
                observations = []
                for inputs, output in zip(obs_combo, output_assignment):
                    inputs_dict = {var: val for var, val in zip(variables, inputs)}
                    observations.append(BooleanObservation(inputs_dict, output))
                
                # Find all compatible expressions
                compatible_exprs = []
                for expr in all_exprs:
                    is_compatible = True
                    for obs in observations:
                        if expr.evaluate(obs.inputs) != obs.output:
                            is_compatible = False
                            break
                    if is_compatible:
                        compatible_exprs.append(expr)
                
                # Only create dataset if there are compatible expressions
                if compatible_exprs:
                    # Create observation description
                    obs_desc = "_".join([f"{inputs[0]}{inputs[1]}" for inputs in obs_combo])
                    out_desc = "".join(map(str, output_assignment))
                    
                    dataset = {
                        "observation_set_id": f"n{n_observations}_inputs_{obs_desc}_outputs_{out_desc}",
                        "n_observations": n_observations,
                        "observation_inputs": [list(inputs) for inputs in obs_combo],
                        "observation_outputs": list(output_assignment),
                        "observations": [{"inputs": obs.inputs, "output": obs.output, "string": obs.to_string()} 
                                       for obs in observations],
                        "ground_truth_expressions": [
                            {
                                "formula": expr.formula,
                                "canonical_form": expr.get_canonical_form(),
                                "mechanistic_key": str(expr.mechanistic_key(**mechanistic_opts)),
                                "truth_table": {str(k): v for k, v in expr.truth_table.items()}
                            } for expr in compatible_exprs
                        ],
                        "n_compatible_expressions": len(compatible_exprs),
                        "variables": variables,
                        "operators": list(operators),
                        "max_depth": max_depth
                    }
                    datasets.append(dataset)
        
        return datasets
    
    @staticmethod
    def generate_complete_dataset_collection(
        mechanistic_opts: Dict[str, Any],
        operator_set: str = 'extended',
        max_depth: int = 1,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Generate complete collection of datasets for all observation counts.
        
        Args:
            operator_set: 'basic', 'extended', or 'full'
            max_depth: Maximum expression depth
            seed: Random seed
            
        Returns:
            Dictionary containing all datasets organized by observation count
        """
        variables = ['x', 'y']
        operators = BooleanDiscoveryGame.OPERATOR_SETS.get(operator_set, {'AND', 'OR', 'NOT'})
        
        # mechanistic_opts = {
        #     'apply_commutativity': True,
        #     'apply_idempotence_and_or': True,
        #     'flatten_associativity': True
        # }
        
        print("=" * 60)
        print("GENERATING COMPLETE BOOLEAN DATASET COLLECTION")
        print("=" * 60)
        print(f"Variables: {variables}")
        print(f"Operators: {operators}")
        print(f"Max depth: {max_depth}")
        print()
        
        # Generate all expressions once to show hypothesis space
        all_exprs = BooleanDiscoveryGame.generate_all_expressions(
            variables, operators, max_depth, mechanistic_opts
        )
        print(f"Total hypothesis space: {len(all_exprs)} expressions")
        
        # Group by semantic equivalence
        semantic_groups = {}
        for expr in all_exprs:
            tt = tuple(sorted(expr.truth_table.items()))
            if tt not in semantic_groups:
                semantic_groups[tt] = []
            semantic_groups[tt].append(expr.formula)
        print(f"Semantically unique functions: {len(semantic_groups)}")
        print()
        
        collection = {
            "metadata": {
                "variables": variables,
                "operators": list(operators),
                "max_depth": max_depth,
                "hypothesis_space_size": len(all_exprs),
                "semantic_unique_count": len(semantic_groups),
                "mechanistic_opts": mechanistic_opts,
                "total_observation_sets": 0  # Will be updated after generation
            },
            "datasets_by_n_observations": {}
        }
        
        # Generate datasets for each observation count
        for n_obs in range(1, 5):  # 1, 2, 3, 4 observations
            print(f"Generating datasets with {n_obs} observation(s)...")
            datasets = BooleanDatasetGenerator.generate_all_observation_combinations(
                variables, operators, max_depth, n_obs, mechanistic_opts, seed
            )
            
            # Statistics
            unique_gt_counts = set()
            for ds in datasets:
                unique_gt_counts.add(ds["n_compatible_expressions"])
            
            print(f"  - Generated {len(datasets)} unique observation sets")
            print(f"  - Compatible expression counts: {sorted(unique_gt_counts)}")
            
            collection["datasets_by_n_observations"][n_obs] = datasets
        
        # Update total observation sets count
        total_sets = sum(len(datasets) for datasets in collection["datasets_by_n_observations"].values())
        collection["metadata"]["total_observation_sets"] = total_sets
        print(f"\nTotal observation sets across all n_observations: {total_sets}")
        
        return collection


def main():
    parser = argparse.ArgumentParser(description="Generate complete Boolean dataset collection with all observation combinations")
    parser.add_argument("--operators", choices=['basic', 'extended', 'full'], default='extended', help="Operator set")
    parser.add_argument("--max-depth", type=int, default=1, help="Maximum expression depth")
    parser.add_argument("--n-observations", type=int, default=None, help="Generate only datasets with this many observations (1-4). If not specified, generate all.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    # Mechanistic knobs (defaults align with your earlier choice)
    parser.add_argument("--mech-no-comm", dest="comm", action="store_false",help="Disable commutativity collapse (ordered children)")
    parser.add_argument("--mech-no-idem", dest="idem", action="store_false",help="Disable idempotence collapse for AND/OR (keep x AND x)")
    parser.add_argument("--mech-no-flat", dest="flat", action="store_false",help="Disable associativity flattening")
    
    parser.set_defaults(comm=True, idem=True, flat=True)

    args = parser.parse_args()
    
    mechanistic_opts = dict(
        apply_commutativity=args.comm,
        apply_idempotence_and_or=args.idem,
        flatten_associativity=args.flat
    )

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"boolean_datasets_complete_{timestamp}.json"
    
    if args.n_observations is not None:
        # Generate only for specific observation count
        if args.n_observations < 1 or args.n_observations > 4:
            print("Error: n_observations must be between 1 and 4")
            return
            
        variables = ['x', 'y']
        operators = BooleanDiscoveryGame.OPERATOR_SETS.get(args.operators, {'AND', 'OR', 'NOT'})
        
        datasets = BooleanDatasetGenerator.generate_all_observation_combinations(
            variables, operators, args.max_depth, args.n_observations, 
            mechanistic_opts, args.seed
        )
        
        result = {
            "metadata": {
                "n_observations": args.n_observations,
                "variables": variables,
                "operators": list(operators),
                "max_depth": args.max_depth,
                "n_datasets": len(datasets),
                "total_observation_sets": len(datasets),
                "mechanistic_opts": mechanistic_opts
            },
            "datasets": datasets
        }
    else:
        # Generate complete collection
        result = BooleanDatasetGenerator.generate_complete_dataset_collection(
            operator_set=args.operators,
            max_depth=args.max_depth,
            seed=args.seed,
            mechanistic_opts=mechanistic_opts
        )
    
    # Save to file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDataset saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()