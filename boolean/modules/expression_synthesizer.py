import re
from typing import List, Tuple, Dict, Any

from boolean_dataset import BooleanObservation, BooleanDiscoveryGame


class BooleanExpressionSynthesizer:
    """
    Enumerate the finite Boolean hypothesis space once and reuse it to find
    expressions compatible with a set of observations. This acts as a fast
    heuristic layer that can propose high-quality candidates before we fall
    back to querying an LLM.
    """

    def __init__(
        self,
        variables: List[str],
        operators: Any,
        max_depth: int,
        mechanistic_opts: Dict[str, Any]
    ) -> None:
        self.variables = list(variables)
        self.operators = set(operators)
        self.max_depth = max_depth
        self.mechanistic_opts = mechanistic_opts or {}
        self._expr_items: List[Tuple[str, Any, int, int]] = []
        self._cache: Dict[Tuple, Tuple[str, ...]] = {}
        self._build_expression_pool()

    def _build_expression_pool(self) -> None:
        expressions = BooleanDiscoveryGame.generate_all_expressions(
            self.variables,
            self.operators,
            self.max_depth,
            self.mechanistic_opts
        )
        for expr in expressions:
            normalized = self._normalize_formula(expr.formula)
            complexity = self._count_ops(expr.sympy_expr)
            depth = self._compute_depth(expr.sympy_expr)
            self._expr_items.append((normalized, expr, complexity, depth))
        self._expr_items.sort(key=lambda item: (item[3], item[2], len(item[0])))

    def find_candidates(self, observations: List[BooleanObservation]) -> List[str]:
        if not observations:
            return [item[0] for item in self._expr_items]

        key = self._make_obs_key(observations)
        if key in self._cache:
            return list(self._cache[key])

        matches: List[str] = []
        for normalized, expr, _, _ in self._expr_items:
            consistent = True
            for obs in observations:
                if expr.evaluate(obs.inputs) != obs.output:
                    consistent = False
                    break
            if consistent:
                matches.append(normalized)

        self._cache[key] = tuple(matches)
        return list(matches)

    def _make_obs_key(self, observations: List[BooleanObservation]) -> Tuple:
        keyed = []
        for obs in observations:
            ordered_inputs = tuple((var, obs.inputs[var]) for var in self.variables)
            keyed.append((ordered_inputs, obs.output))
        keyed.sort()
        return tuple(keyed)

    @staticmethod
    def _normalize_formula(formula: str) -> str:
        compact = re.sub(r"\s+", " ", formula.strip())
        compact = compact.replace("( ", "(").replace(" )", ")")
        compact = compact.replace(" ,", ",")
        compact = compact.replace(", ", ", ")
        return compact

    @staticmethod
    def _count_ops(sym_expr: Any) -> int:
        if sym_expr is None:
            return 0
        try:
            return int(sym_expr.count_ops())
        except Exception:
            return 0

    def _compute_depth(self, sym_expr: Any) -> int:
        if sym_expr is None or not getattr(sym_expr, "args", None):
            return 0
        child_depths = [self._compute_depth(arg) for arg in sym_expr.args]
        return 1 + max(child_depths)
