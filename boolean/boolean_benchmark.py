import sys
import json
import argparse
import os
import re
import yaml
import random
import numpy as np
import ast
import sympy as sp
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from textwrap import dedent
from collections import Counter
from scipy import stats
import traceback
from itertools import product

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent))

from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM
from boolean_dataset import BooleanExpression, BooleanObservation


class BooleanBenchmarkRefined:
    def __init__(self, complete_dataset_path: str):
        """
        Initialize benchmark with a complete dataset.
        
        Args:
            complete_dataset_path: Path to the complete Boolean dataset JSON file
        """
        with open(complete_dataset_path, 'r') as f:
            self.complete_dataset = json.load(f)
        
        # Extract metadata
        self.mechanistic_opts = self.complete_dataset['metadata']['mechanistic_opts']
        self.metadata = self.complete_dataset['metadata']
        self.variables = self.metadata['variables']
        self.operators = set(self.metadata['operators'])
        self.max_depth = self.metadata['max_depth']
        
        # Flatten all datasets into a single list for sampling
        self.all_observation_sets = []
        for n_obs, datasets in self.complete_dataset['datasets_by_n_observations'].items():
            self.all_observation_sets.extend(datasets)
        
        print(f"Loaded complete dataset with {len(self.all_observation_sets)} observation sets")
        print(f"Variables: {', '.join(self.variables)}")
        print(f"Operators: {', '.join(self.operators)}")
        print(f"Max depth: {self.max_depth}")
    
    def sample_observation_sets(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        """
        Sample n observation sets from the complete dataset.
        
        Args:
            n_samples: Number of observation sets to sample
            seed: Random seed for reproducibility
        
        Returns:
            List of sampled observation sets
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Sample without replacement
        n_available = len(self.all_observation_sets)
        n_to_sample = min(n_samples, n_available)
        
        if n_to_sample < n_samples:
            print(f"Warning: Requested {n_samples} samples but only {n_available} available")
        
        sampled = random.sample(self.all_observation_sets, n_to_sample)
        return sampled

    def create_prompt(
        self,
        observations: List[BooleanObservation],
        prior_attempts: List[Dict[str, Any]],
        recovered_stats: Optional[Dict[str, int]] = None
    ) -> str:
        """Build a structured prompt with explicit feedback for the LLM."""
        obs_lines = [obs.to_string() for obs in observations]
        if obs_lines:
            obs_block = "        " + "\n        ".join(obs_lines)
        else:
            obs_block = "        (no observations provided)"

        if prior_attempts:
            history_lines: List[str] = []
            for attempt in prior_attempts:
                if isinstance(attempt, dict):
                    idx = attempt.get('index', len(history_lines) + 1)
                    status = attempt.get('status', 'unknown').upper()
                    expr = attempt.get('expression') or attempt.get('raw_response') or '(no expression captured)'
                    reason = attempt.get('reason')
                    extras = []
                    if attempt.get('duplicate'):
                        extras.append('duplicate structure')
                    if attempt.get('recovered'):
                        extras.append('recovers ground truth')
                    extra_text = f" ({'; '.join(extras)})" if extras else ''
                    line = f"{idx}. [{status}] {expr}{extra_text}"
                    if reason:
                        line += f" -> {reason}"
                    history_lines.append(line)
                else:
                    history_lines.append(str(attempt))
            prior_block = "        " + "\n        ".join(history_lines)
        else:
            prior_block = "        (no previous attempts)"

        op_descriptions = []
        if 'AND' in self.operators:
            op_descriptions.append('AND (conjunction)')
        if 'OR' in self.operators:
            op_descriptions.append('OR (disjunction)')
        if 'NOT' in self.operators:
            op_descriptions.append('NOT (negation)')
        if 'XOR' in self.operators:
            op_descriptions.append('XOR (exclusive or)')
        if 'NOR' in self.operators:
            op_descriptions.append('NOR (NOT (x OR y))')
        operators_str = ', '.join(op_descriptions) if op_descriptions else '(no operators)'

        mech = self.complete_dataset.get('metadata', {}).get('mechanistic_opts', {})
        comm = mech.get('apply_commutativity', True)
        idem = mech.get('apply_idempotence_and_or', True)
        flat = mech.get('flatten_associativity', True)

        eq_rules: List[str] = []
        if comm:
            eq_rules.append('- Commutativity: reorderings are the SAME (x AND y = y AND x).')
        if idem:
            eq_rules.append('- Idempotence: duplicates collapse (x AND x = x; x OR x = x).')
        if flat:
            eq_rules.append('- Associativity flattening: cascades of the SAME operator are the SAME ((x AND y) AND x = x AND y AND x).')
        else:
            eq_rules.append('- No associativity flattening: different parentheses produce DIFFERENT expressions.')
        eq_rules.append('- No distributivity / absorption / De Morgan rewrites: mixed-operator forms stay DIFFERENT.')
        rules_text = "\n        ".join(eq_rules)

        combos = list(product([0, 1], repeat=len(self.variables)))
        combo_str = ', '.join(''.join(str(bit) for bit in combo) for combo in combos)

        if recovered_stats:
            total = recovered_stats.get('total', 0)
            recovered = recovered_stats.get('recovered', 0)
            if total:
                remaining = max(total - recovered, 0)
                recovered_text = f"{recovered}/{total} ground-truth structures covered; {remaining} still unrecovered."
            else:
                recovered_text = 'Ground-truth structure count unavailable.'
        else:
            recovered_text = 'No ground-truth structures recovered yet; seek a new candidate.'

        prompt = dedent(f"""
        You are a boolean logic expert. Produce a hypothesis expression consistent with every observation and structurally novel relative to prior attempts.

        Observations (input -> output):
{obs_block}

        Previous attempts (status :: feedback):
{prior_block}

        Progress summary: {recovered_text}

        Allowed variables: {', '.join(self.variables)}
        Allowed operators: {operators_str}

        Treat two expressions as equivalent only if all of the following apply:
        {rules_text}

        Workflow instructions:
        1. Enumerate every input combination ({combo_str}) and make sure your expression matches the provided outputs before replying.
        2. Do not repeat expressions that are equivalent to the history under the rules above.
        3. Keep expression depth <= {self.max_depth} and do not use literal True/False constants.
        4. If an attempt fails, self-correct and only share the corrected final answer.

        Output format (plain text):
        Expression: <final expression>
        Truth table:
          <inputs> -> <output>   # list all {len(combos)} combinations
        Reasoning: <one sentence on why the expression fits and is novel>

        Do not add extra sections beyond the three shown above.
        """).strip()
        return prompt

    def parse_llm_response(self, response: str) -> Optional[str]:
        """Parse LLM response to extract Boolean expression."""
        if not isinstance(response, str):
            return None

        response = response.strip()

        # Clean common formatting issues
        def clean_expression(expr: str) -> str:
            expr = re.sub(r'\[()\[\]]', '', expr)
            expr = re.sub(r'\$+', '', expr)
            expr = expr.strip('`')
            replacements = {
                '\land': 'AND',
                '\lor': 'OR',
                '\lnot': 'NOT',
                '\neg': 'NOT',
                '\wedge': 'AND',
                '\vee': 'OR',
                '¬': 'NOT',
            }
            for src, dst in replacements.items():
                expr = expr.replace(src, dst)
            expr = re.sub(r'\text\{([^}]+)\}', r'\1', expr)
            expr = re.sub(r'\*+$', '', expr)
            expr = expr.strip("'\" ").rstrip('.,;:')
            return expr.strip()

        # Attempt to parse JSON-style answers
        possible_json_strings = [response]
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if fence_match:
            possible_json_strings.append(fence_match.group(1).strip())
        brace_match = re.search(r"\{.*\}", response, re.DOTALL)
        if brace_match:
            possible_json_strings.append(brace_match.group(0))

        for candidate in possible_json_strings:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue

            if isinstance(parsed, dict):
                keys_to_try = ['expression', 'Expression', 'expr', 'final_expression', 'boolean_expression']
                expr_value = None
                for key in keys_to_try:
                    if key in parsed:
                        expr_value = parsed[key]
                        break

                if expr_value is None:
                    for container_key in ['answer', 'result', 'data']:
                        inner = parsed.get(container_key)
                        if isinstance(inner, dict):
                            for key in keys_to_try:
                                if key in inner:
                                    expr_value = inner[key]
                                    break
                        if expr_value is not None:
                            break

                if expr_value:
                    cleaned = clean_expression(str(expr_value))
                    if cleaned:
                        return cleaned

            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str):
                        cleaned = clean_expression(item)
                        if cleaned:
                            return cleaned

        # Look for explicit "Expression:" lines
        lines = response.splitlines()
        for line in lines:
            if 'Expression:' in line or 'expression:' in line:
                parts = re.split(r'[Ee]xpression:', line)
                if len(parts) >= 2:
                    expr = clean_expression(parts[1])
                    if expr:
                        return expr

        # Fallback: scan for the first line that resembles a boolean expression
        for line in lines:
            candidate = clean_expression(line.strip())
            if not candidate:
                continue
            if any(op in candidate.upper() for op in ['AND', 'OR', 'NOT', 'XOR', 'NOR']):
                return candidate
            if all(c.isalnum() or c in '() ' for c in candidate):
                return candidate

        return None

    def get_expression_mechanistic_key(self, expr_str: str) -> Optional[Tuple]:
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            return expr.mechanistic_key(**self.mechanistic_opts)
        except:
            return None
    
    def _in_space(self, expr_str: str) -> bool:
        """Check if expression meets all space constraints (variables, operators, depth, no constants)."""
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            
            if expr.sympy_expr is None:
                return False
            
            # Check variables
            allowed_symbols = {sp.symbols(v) for v in self.variables}
            if not expr.sympy_expr.free_symbols.issubset(allowed_symbols):
                return False
            
            # Check operators
            if not self._check_operators_in_ast(expr.sympy_expr, self.operators):
                return False
            
            # Check no constants
            for node in sp.preorder_traversal(expr.sympy_expr):
                if isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                    return False
            
            # Check depth
            if self._get_ast_depth(expr.sympy_expr) > self.max_depth:
                return False
            
            return True
        except:
            return False
    
    def _get_ast_depth(self, sympy_expr) -> int:
        """Calculate the true AST depth of an expression."""
        if sympy_expr is None:
            return 0
        
        # Leaf nodes (variables, constants) have depth 0
        if isinstance(sympy_expr, sp.Symbol):
            return 0
        if isinstance(sympy_expr, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
            return 0
        
        # For operators, depth is 1 + max depth of children
        if hasattr(sympy_expr, 'args') and len(sympy_expr.args) > 0:
            child_depths = [self._get_ast_depth(arg) for arg in sympy_expr.args]
            return 1 + max(child_depths)
        
        return 0
    
    def _check_operators_in_ast(self, sympy_expr, allowed_ops: Set[str]) -> bool:
        if sympy_expr is None:
            return False
        stack = [sympy_expr]
        while stack:
            node = stack.pop()

            # NOR pattern: Not(Or(...))
            if isinstance(node, sp.Not) and isinstance(node.args[0], sp.Or):
                if 'NOR' not in allowed_ops:
                    return False
                # treat as primitive; do NOT traverse inside
                continue

            # primitive checks
            if isinstance(node, sp.Not) and 'NOT' not in allowed_ops:
                return False
            elif isinstance(node, sp.And) and 'AND' not in allowed_ops:
                return False
            elif isinstance(node, sp.Or) and 'OR' not in allowed_ops:
                return False
            elif isinstance(node, sp.Xor) and 'XOR' not in allowed_ops:
                return False
            elif isinstance(node, sp.Implies) and 'IMPLIES' not in allowed_ops:
                return False
            elif isinstance(node, sp.Equivalent) and 'EQUIV' not in allowed_ops:
                return False

            if hasattr(node, 'args'):
                stack.extend(node.args)
        return True
    
    def validate_expression(self, expr_str: str, observations: List[BooleanObservation]) -> Tuple[bool, Dict[str, Any]]:
        """Validate if expression is consistent with observations and meets constraints."""
        info: Dict[str, Any] = {}
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)
        except Exception as exc:
            return False, {'reason': f'Failed to parse expression: {exc}'}

        if expr.sympy_expr is None:
            return False, {'reason': 'Expression could not be parsed into a symbolic form.'}

        allowed_symbols = {sp.symbols(v) for v in self.variables}
        free_symbols = expr.sympy_expr.free_symbols
        if not free_symbols.issubset(allowed_symbols):
            invalid_symbols = sorted({str(s) for s in free_symbols - allowed_symbols})
            return False, {'reason': f'Uses unsupported variables: {", ".join(invalid_symbols)}'}

        if not self._check_operators_in_ast(expr.sympy_expr, self.operators):
            return False, {'reason': 'Uses operators outside the allowed set.'}

        for node in sp.preorder_traversal(expr.sympy_expr):
            if isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                return False, {'reason': 'Contains boolean constants (True/False), which are not allowed.'}

        ast_depth = self._get_ast_depth(expr.sympy_expr)
        if ast_depth > self.max_depth:
            return False, {'reason': f'Expression depth {ast_depth} exceeds allowed maximum of {self.max_depth}.'}

        try:
            for obs in observations:
                predicted = expr.evaluate(obs.inputs)
                if predicted != obs.output:
                    return False, {
                        'reason': f'Mismatch for inputs {obs.inputs}: expected {obs.output}, got {predicted}.',
                        'truth_table': expr.truth_table
                    }
        except Exception as exc:
            return False, {'reason': f'Failed while evaluating expression: {exc}'}

        info['truth_table'] = expr.truth_table
        return True, info

    def _classify_error(self, error_message: str) -> str:
        """Classify the type of error from the error message with more detail."""
        if "Expecting value" in error_message:
            # Extract line and column info if available
            match = re.search(r'line (\d+) column (\d+)', error_message)
            if match:
                return f"json_parse_error (line {match.group(1)}, col {match.group(2)})"
            return "json_parse_error"
        elif "Rate limit" in error_message.lower():
            return "rate_limit"
        elif "timeout" in error_message.lower():
            return "timeout"
        elif "401" in error_message or "unauthorized" in error_message.lower():
            return "auth_error"
        elif "403" in error_message or "forbidden" in error_message.lower():
            return "forbidden_error"
        elif "404" in error_message:
            return "not_found_error"
        elif "429" in error_message:
            return "rate_limit_429"
        elif "500" in error_message or "internal server error" in error_message.lower():
            return "server_error_500"
        elif "502" in error_message or "bad gateway" in error_message.lower():
            return "bad_gateway_502"
        elif "503" in error_message or "service unavailable" in error_message.lower():
            return "service_unavailable_503"
        elif "connection" in error_message.lower():
            return "connection_error"
        elif "JSONDecodeError" in error_message:
            return "json_decode_error"
        else:
            # Try to extract HTTP status code
            match = re.search(r'\b(\d{3})\b', error_message)
            if match:
                return f"http_error_{match.group(1)}"
            return "unknown_error"
    
    def get_expression_depth(self, expr_str: str) -> int:
        """Calculate the depth of a Boolean expression."""
        # Simple heuristic: count maximum nesting level of parentheses
        max_depth = 0
        current_depth = 0
        
        for char in expr_str:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        # Also check for operators without parentheses
        if max_depth == 0:
            # Count operators to estimate depth
            expr_upper = expr_str.upper()
            if any(op in expr_upper for op in ['AND','OR','NOT','XOR','NOR']):
                max_depth = 1
        
        return max_depth
    
    def evaluate_single_observation_set(
        self,
        llm: LLMInterface,
        observation_set: Dict,
        n_queries: int = 10,
        verbose: bool = True,
        max_retries: int = 5
    ) -> Dict:
        """Evaluate LLM on a single observation set."""
        observations = [
            BooleanObservation(obs['inputs'], obs['output'])
            for obs in observation_set['observations']
        ]

        ground_truth_expressions = observation_set['ground_truth_expressions']
        gt_truth_tables = []
        gt_mechanistic_keys = []
        for gt in ground_truth_expressions:
            truth_table = {}
            for key_str, value in gt['truth_table'].items():
                key_tuple = ast.literal_eval(key_str)
                truth_table[key_tuple] = value
            gt_truth_tables.append(truth_table)
            gt_mech_key = gt.get('mechanistic_key', None)
            if gt_mech_key:
                gt_mechanistic_keys.append(ast.literal_eval(gt_mech_key) if isinstance(gt_mech_key, str) else gt_mech_key)

        all_hypotheses: List[str] = []
        valid_hypotheses: List[str] = []
        unique_mechanistic_keys: Set[Tuple] = set()
        unique_valid_expressions: List[str] = []
        all_unique_mechanistic_keys: Set[Tuple] = set()
        unique_all_expressions: List[str] = []
        attempt_history: List[Dict[str, Any]] = []
        recovered_gt_mech_keys: Set[Tuple] = set()

        parse_success_count = 0
        in_space_count = 0

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0

        def stringify_truth_table(truth_table: Dict[Tuple, Any]) -> Dict[str, Any]:
            """Convert tuple-keyed truth tables to JSON-serializable form."""
            converted = {}
            for key, value in truth_table.items():
                if isinstance(key, tuple):
                    key_str = "".join(str(bit) for bit in key)
                else:
                    key_str = str(key)
                converted[key_str] = value
            return converted

        errors: List[Dict[str, Any]] = []
        error_counts: Dict[str, int] = {}

        for i in range(n_queries):
            prompt = self.create_prompt(
                observations,
                attempt_history,
                {
                    'recovered': len(recovered_gt_mech_keys),
                    'total': len(gt_mechanistic_keys)
                }
            )

            hypothesis_str: Optional[str] = None
            query_error: Optional[Dict[str, Any]] = None
            last_response: Optional[str] = None

            for attempt in range(max_retries):
                if hasattr(llm, 'query_with_usage'):
                    result = llm.query_with_usage(prompt)
                    response = result['response']
                    last_response = response if isinstance(response, str) else str(response)
                    usage = result.get('usage', {})
                    total_prompt_tokens += usage.get('prompt_tokens', 0)
                    total_completion_tokens += usage.get('completion_tokens', 0)
                    total_tokens += usage.get('total_tokens', 0)
                    total_cost += result.get('cost', 0.0)
                else:
                    response = llm.query(prompt)
                    last_response = response if isinstance(response, str) else str(response)

                if isinstance(response, str) and response.startswith("Error querying"):
                    error_type = self._classify_error(response)
                    query_error = {
                        'query_index': i,
                        'attempt': attempt + 1,
                        'error_message': response,
                        'error_type': error_type
                    }
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    continue

                hypothesis_str = self.parse_llm_response(response)
                if hypothesis_str:
                    break

                query_error = {
                    'query_index': i,
                    'attempt': attempt + 1,
                    'error_message': 'Unable to parse model output.',
                    'error_type': 'parse_failure'
                }

            attempt_record: Dict[str, Any] = {
                'index': len(attempt_history) + 1,
                'expression': hypothesis_str,
            }
            if last_response:
                attempt_record['raw_response'] = last_response[:500]

            if not hypothesis_str:
                if query_error:
                    errors.append(query_error)
                    attempt_record['status'] = query_error['error_type']
                    attempt_record['reason'] = query_error['error_message']
                else:
                    attempt_record['status'] = 'parse_failure'
                    attempt_record['reason'] = 'Unable to interpret model output.'
                    error_counts['parse_failure'] = error_counts.get('parse_failure', 0) + 1
                all_hypotheses.append(f"[ERROR] {attempt_record.get('reason', 'Unknown failure')}")
                attempt_history.append(attempt_record)
                continue

            parse_success_count += 1

            in_space = self._in_space(hypothesis_str)
            is_valid, validation_info = self.validate_expression(hypothesis_str, observations)

            if in_space:
                in_space_count += 1
                mech_key_all = self.get_expression_mechanistic_key(hypothesis_str)
                if mech_key_all and mech_key_all not in all_unique_mechanistic_keys:
                    all_unique_mechanistic_keys.add(mech_key_all)
                    unique_all_expressions.append(hypothesis_str)
            else:
                attempt_record['status'] = 'out_of_space'
                attempt_record['reason'] = validation_info.get('reason', 'Violates search-space constraints.') if isinstance(validation_info, dict) else 'Violates search-space constraints.'
                all_hypotheses.append(f"[INVALID:SPACE] {hypothesis_str} :: {attempt_record['reason']}")
                attempt_history.append(attempt_record)
                continue

            if not is_valid:
                attempt_record['status'] = 'invalid'
                attempt_record['reason'] = validation_info.get('reason', 'Failed to satisfy observations.')
                if isinstance(validation_info, dict) and 'truth_table' in validation_info:
                    attempt_record['truth_table'] = stringify_truth_table(validation_info['truth_table'])
                all_hypotheses.append(f"[INVALID] {hypothesis_str} :: {attempt_record['reason']}")
                attempt_history.append(attempt_record)
                continue

            attempt_record['status'] = 'valid'
            attempt_record['reason'] = 'Matches all provided observations.'
            if isinstance(validation_info, dict) and 'truth_table' in validation_info:
                attempt_record['truth_table'] = stringify_truth_table(validation_info['truth_table'])

            valid_hypotheses.append(hypothesis_str)

            mech_key = self.get_expression_mechanistic_key(hypothesis_str)
            duplicate_flag = False
            if mech_key:
                attempt_record['mechanistic_key'] = repr(mech_key)
                if mech_key not in unique_mechanistic_keys:
                    unique_mechanistic_keys.add(mech_key)
                    unique_valid_expressions.append(hypothesis_str)
                else:
                    duplicate_flag = True

                if mech_key in gt_mechanistic_keys:
                    if mech_key not in recovered_gt_mech_keys:
                        recovered_gt_mech_keys.add(mech_key)
                        attempt_record['recovered'] = True
                    else:
                        attempt_record['recovered'] = False

            attempt_record['duplicate'] = duplicate_flag
            if duplicate_flag:
                attempt_record['reason'] = attempt_record.get('reason', '') + ' (structure already seen)'
                all_hypotheses.append(f"[VALID DUPLICATE] {hypothesis_str}")
            else:
                all_hypotheses.append(f"[VALID] {hypothesis_str}")

            attempt_history.append(attempt_record)

        parse_success_rate = parse_success_count / n_queries if n_queries > 0 else 0
        in_space_rate = in_space_count / n_queries if n_queries > 0 else 0
        valid_rate = len(valid_hypotheses) / n_queries if n_queries > 0 else 0
        novelty_rate = len(unique_all_expressions) / n_queries if n_queries > 0 else 0

        recovered_gts = set()
        generated_mech_keys = set()
        for expr in unique_valid_expressions:
            mech_key = self.get_expression_mechanistic_key(expr)
            if mech_key:
                generated_mech_keys.add(mech_key)
        for j, gt_key in enumerate(gt_mechanistic_keys):
            if gt_key in generated_mech_keys:
                recovered_gts.add(j)

        recovery_rate = len(recovered_gts) / len(gt_mechanistic_keys) if gt_mechanistic_keys else 0

        return {
            'observation_set_id': observation_set['observation_set_id'],
            'n_observations': observation_set['n_observations'],
            'n_ground_truths': len(ground_truth_expressions),
            'n_queries': n_queries,
            'n_valid': len(valid_hypotheses),
            'n_unique_valid': len(unique_valid_expressions),
            'n_unique_all': len(unique_all_expressions),
            'n_recovered_gts': len(recovered_gts),
            'parse_success_rate': parse_success_rate,
            'in_space_rate': in_space_rate,
            'valid_rate': valid_rate,
            'novelty_rate': novelty_rate,
            'recovery_rate': recovery_rate,
            'all_hypotheses': all_hypotheses,
            'valid_hypotheses': valid_hypotheses,
            'unique_valid_expressions': unique_valid_expressions,
            'unique_all_expressions': unique_all_expressions,
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'cost': total_cost,
            'attempt_history': attempt_history,
            'errors': errors,
            'error_summary': {
                'total_errors': len(errors),
                'error_types': error_counts
            }
        }

    def run_benchmark(
        self,
        llm: LLMInterface,
        n_samples: int = 10,
        n_queries_per_sample: Optional[int] = None,
        query_multiplier: float = 2.0,
        seed: Optional[int] = None,
        verbose: bool = True,
        checkpoint_dir: str = "checkpoints",
        run_id: Optional[str] = None
    ) -> Dict:
        """
        Run the benchmark with sampling.
        
        Args:
            llm: LLM interface to use
            n_samples: Number of observation sets to sample
            n_queries_per_sample: Fixed number of queries per observation set (if None, uses query_multiplier)
            query_multiplier: Multiplier for n_gt to determine queries (default 2.0 means 2x ground truths)
            seed: Random seed
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints
        
        Returns:
            Dictionary with complete benchmark results
        """
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID and safe filename
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize LLM name for filename
        safe_llm_name = llm.get_name().replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        checkpoint_file = checkpoint_path / f"checkpoint_{safe_llm_name}_{run_id}.json"
        
        print(f"\nRunning refined Boolean benchmark")
        print(f"LLM: {llm.get_name()}")
        print(f"Sampling {n_samples} observation sets")
        if n_queries_per_sample is not None:
            print(f"Queries per sample: {n_queries_per_sample} (fixed)")
        else:
            print(f"Queries per sample: {query_multiplier}x number of ground truths (adaptive)")
        print(f"Checkpoint file: {checkpoint_file}")
        print("-" * 50)
        
        # Sample observation sets
        sampled_sets = self.sample_observation_sets(n_samples, seed)
        
        # Initialize results tracking
        all_results = []
        valid_rates = []
        novelty_rates = []
        recovery_rates = []
        
        # Initialize token and cost tracking
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Initialize error tracking
        all_errors = []
        total_error_counts = {}
        
        # Load existing checkpoint if it exists
        start_idx = 0
        if checkpoint_file.exists():
            print(f"Using Recovered checkpoint file: {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                all_results = checkpoint_data['results']
                start_idx = len(all_results)
                print(f"Resuming from checkpoint: {start_idx}/{n_samples} completed")

                # Recalculate rates from checkpoint
                for result in all_results:
                    valid_rates.append(result['valid_rate'])
                    novelty_rates.append(result['novelty_rate'])
                    recovery_rates.append(result['recovery_rate'])

                # RESTORE TOKEN AND COST TOTALS
                if 'total_token_usage' in checkpoint_data:
                    total_prompt_tokens = checkpoint_data['total_token_usage'].get('prompt_tokens', 0)
                    total_completion_tokens = checkpoint_data['total_token_usage'].get('completion_tokens', 0)
                    total_tokens = checkpoint_data['total_token_usage'].get('total_tokens', 0)
                if 'total_cost' in checkpoint_data:
                    total_cost = checkpoint_data['total_cost']
        
        # Process each sampled observation set
        for idx in range(start_idx, len(sampled_sets)):
            obs_set = sampled_sets[idx]
            
            if verbose:
                print(f"\nSample {idx + 1}/{n_samples}")
                print(f"  Observation set ID: {obs_set['observation_set_id']}")
                print(f"  Number of observations: {obs_set['n_observations']}")
                print(f"  Number of ground truths: {obs_set['n_compatible_expressions']}")
            
            try:
                # Determine number of queries for this observation set
                if n_queries_per_sample is not None:
                    # Use fixed number
                    n_queries = n_queries_per_sample
                else:
                    # Adaptive: use multiplier of ground truths
                    n_gt = obs_set['n_compatible_expressions']
                    n_queries = max(1, int(n_gt * query_multiplier))
                    if verbose:
                        print(f"  Using {n_queries} queries ({query_multiplier}x {n_gt} ground truths)")
                
                # Evaluate on this observation set
                result = self.evaluate_single_observation_set(
                    llm, obs_set, n_queries, verbose=False
                )
                
                all_results.append(result)
                valid_rates.append(result['valid_rate'])
                novelty_rates.append(result['novelty_rate'])
                recovery_rates.append(result['recovery_rate'])
                
                # Aggregate token usage and costs
                if 'token_usage' in result:
                    total_prompt_tokens += result['token_usage']['prompt_tokens']
                    total_completion_tokens += result['token_usage']['completion_tokens']
                    total_tokens += result['token_usage']['total_tokens']
                if 'cost' in result:
                    total_cost += result['cost']
                
                # Aggregate errors
                if 'errors' in result and result['errors']:
                    all_errors.extend(result['errors'])
                    # Update error type counts
                    if 'error_summary' in result:
                        for error_type, count in result['error_summary']['error_types'].items():
                            total_error_counts[error_type] = total_error_counts.get(error_type, 0) + count
                
                if verbose:
                    print(f"  Valid rate: {result['valid_rate']:.2%}")
                    print(f"  Novelty rate: {result['novelty_rate']:.2%}")
                    print(f"  Recovery rate: {result['recovery_rate']:.2%}")
                
                # Save checkpoint after each sample
                checkpoint_data = {
                    'run_id': run_id,
                    'llm_name': llm.get_name(),
                    'n_samples': n_samples,
                    'n_queries_per_sample': n_queries_per_sample,
                    'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                    'results': all_results,
                    'total_token_usage': {
                        'prompt_tokens': total_prompt_tokens,
                        'completion_tokens': total_completion_tokens,
                        'total_tokens': total_tokens
                    },
                    'total_cost': total_cost,
                    'total_errors': len(all_errors),
                    'error_types': total_error_counts
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

            except Exception as e:
                print(f"  Error processing sample {idx + 1}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Calculate statistics
        def calculate_stats(rates):
            if not rates:
                return {'mean': 0, 'std': 0, 'var': 0, 'min': 0, 'max': 0}
            return {
                'mean': np.mean(rates),
                'std': np.std(rates),
                'var': np.var(rates),
                'min': np.min(rates),
                'max': np.max(rates)
            }
        
        # Calculate p-values (one-sample t-test against null hypothesis of 0)
        def calculate_p_value(rates):
            if not rates or len(rates) < 2:
                return None
            t_stat, p_val = stats.ttest_1samp(rates, 0)
            return p_val
        
        # Compile final results
        final_results = {
            'run_id': run_id,
            'llm_name': llm.get_name(),
            'n_samples': len(all_results),
            'n_queries_per_sample': n_queries_per_sample,
            'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
            'query_mode': 'fixed' if n_queries_per_sample is not None else f'adaptive_{query_multiplier}x',
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'valid_rate': {
                    **calculate_stats(valid_rates),
                    'p_value': calculate_p_value(valid_rates)
                },
                'novelty_rate': {
                    **calculate_stats(novelty_rates),
                    'p_value': calculate_p_value(novelty_rates)
                },
                'recovery_rate': {
                    **calculate_stats(recovery_rates),
                    'p_value': calculate_p_value(recovery_rates)
                }
            },
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens,
                'avg_tokens_per_sample': total_tokens / len(all_results) if all_results else 0,
                'avg_tokens_per_query': total_tokens / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'cost': {
                'total_cost': total_cost,
                'avg_cost_per_sample': total_cost / len(all_results) if all_results else 0,
                'avg_cost_per_query': total_cost / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'error_summary': {
                'total_errors': len(all_errors),
                'error_types': total_error_counts,
                'error_rate': len(all_errors) / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'per_sample_results': all_results
        }
        
        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 50)
        print(f"Samples evaluated: {len(all_results)}/{n_samples}")
        
        for metric_name, metric_key in [('Valid Rate', 'valid_rate'), 
                                        ('Novelty Rate', 'novelty_rate'), 
                                        ('Recovery Rate', 'recovery_rate')]:
            stats_dict = final_results['statistics'][metric_key]
            print(f"\n{metric_name}:")
            print(f"  Mean ± Std: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Variance: {stats_dict['var']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            if stats_dict['p_value'] is not None:
                print(f"  p-value: {stats_dict['p_value']:.4f}")
        
        # Print token usage and cost summary
        print(f"\nToken Usage:")
        print(f"  Total tokens: {final_results['token_usage']['total_tokens']:,}")
        print(f"  Prompt tokens: {final_results['token_usage']['prompt_tokens']:,}")
        print(f"  Completion tokens: {final_results['token_usage']['completion_tokens']:,}")
        print(f"  Avg tokens/sample: {final_results['token_usage']['avg_tokens_per_sample']:.1f}")
        print(f"  Avg tokens/query: {final_results['token_usage']['avg_tokens_per_query']:.1f}")
        
        print(f"\nCost:")
        print(f"  Total cost: ${final_results['cost']['total_cost']:.4f}")
        print(f"  Avg cost/sample: ${final_results['cost']['avg_cost_per_sample']:.4f}")
        print(f"  Avg cost/query: ${final_results['cost']['avg_cost_per_query']:.6f}")
        
        # Print error summary if there were errors
        if final_results['error_summary']['total_errors'] > 0:
            print(f"\nErrors:")
            print(f"  Total errors: {final_results['error_summary']['total_errors']}")
            print(f"  Error rate: {final_results['error_summary']['error_rate']:.2%}")
            if final_results['error_summary']['error_types']:
                print(f"  Error types:")
                for error_type, count in final_results['error_summary']['error_types'].items():
                    print(f"    - {error_type}: {count}")
        
        print("=" * 50)
        
        return final_results


def setup_llm(llm_type: str, **kwargs) -> LLMInterface:
    """Set up the LLM interface based on type."""
    
    if llm_type == "openai":
        api_key = kwargs.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        return OpenAILLM(
            model=kwargs.get('model', 'gpt-4'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    elif llm_type == "anthropic":
        api_key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key required")
        
        return AnthropicLLM(
            model=kwargs.get('model', 'claude-3-opus-20240229'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    elif llm_type == "openrouter":
        api_key = kwargs.get('api_key') or os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key required")
        
        return OpenRouterLLM(
            model=kwargs.get('model', 'anthropic/claude-3.5-sonnet'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run refined Boolean discovery benchmark with sampling")
    parser.add_argument("--dataset", required=True, help="Path to complete Boolean dataset JSON file")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of observation sets to sample")
    parser.add_argument("--n-queries", type=int, default=None, help="Fixed number of queries per observation set (if not set, uses adaptive)")
    parser.add_argument("--query-multiplier", type=float, default=1.0, help="Multiplier for adaptive queries (n_queries = n_gt * multiplier)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    llm_type = config.get('llm', {}).get('type', 'openrouter')
    
    model = config.get('llm', {}).get('models', {}).get(llm_type)
    if not model:
        default_models = {
            'openrouter': 'openai/gpt-3.5-turbo',
            'openai': 'gpt-4',
            'anthropic': 'claude-3-opus-20240229'
        }
        model = default_models.get(llm_type)
    
    api_key = config.get('llm', {}).get('api_keys', {}).get(llm_type)
    if not api_key:
        env_vars = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY'
        }
        if llm_type in env_vars:
            api_key = os.environ.get(env_vars[llm_type])
    
    temperature = config.get('llm', {}).get('temperature', 0.7)
    checkpoint_dir = config.get('benchmark', {}).get('checkpoint', 'checkpoints')
    run_id = config.get('benchmark', {}).get('run_id', None)
    verbose = config.get('benchmark', {}).get('verbose', True)

    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    dataset_name = Path(args.dataset).stem
    model_name = Path(model).stem if model else llm_type
    output_pattern = config.get('benchmark', {}).get('output_pattern', 'results_{dataset_name}_{model}.json')
    output = output_pattern.format(dataset_name=dataset_name, model=model_name, llm_type=llm_type)

    # Generate output filename if not specified
    if output is None:
        dataset_name = Path(args.dataset).stem
        model_name = Path(model).stem if model else llm_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"results/{dataset_name}_{model_name}_{timestamp}.json"

    # Initialize benchmark
    benchmark = BooleanBenchmarkRefined(args.dataset)
     
    # Print configuration
    print("\n" + "=" * 60)
    print("REFINED BOOLEAN BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"LLM Type: {llm_type}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Samples: {args.n_samples}")
    print(f"Queries per sample: {args.n_queries}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output}")
    print("=" * 60)
    
    # Set up LLM
    llm = setup_llm(
        llm_type,
        model=model,
        api_key=api_key,
        temperature=temperature
    )
    
    # Run benchmark
    results = benchmark.run_benchmark(
        llm=llm,
        n_samples=args.n_samples,
        n_queries_per_sample=args.n_queries,
        query_multiplier=args.query_multiplier,
        seed=args.seed,
        verbose=verbose,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id
    )
    
    # Save final results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal results saved to: {output}")


if __name__ == "__main__":
    main()
