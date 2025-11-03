from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Iterable
from collections import Counter

from modules.models import CausalGraph


@dataclass
class ConstraintSummary:
    """Summarises downstream relationships implied by perturbation observations."""

    descendant_map: Dict[str, Set[str]]
    non_descendant_map: Dict[str, Set[str]]
    forbidden_edges: Set[Tuple[str, str]]


def _normalise_effects(effects: Dict[str, int]) -> Dict[str, int]:
    return {k: int(v) for k, v in effects.items()}


def build_constraint_summary(nodes: Iterable[str], observations: List[Dict[str, Dict[str, int]]]) -> ConstraintSummary:
    """Derive deterministic descendant/non-descendant information from observations."""
    node_list = list(nodes)
    descendants: Dict[str, Set[str]] = {n: set() for n in node_list}
    non_descendants: Dict[str, Set[str]] = {n: set() for n in node_list}

    for obs in observations:
        perturbed = obs["perturbed_node"]
        effects = _normalise_effects(obs["effects"])
        for target, val in effects.items():
            if target == perturbed:
                continue
            if val == 1:
                descendants[perturbed].add(target)
                non_descendants[target].add(perturbed)
            else:
                non_descendants[perturbed].add(target)

    forbidden_edges = set()
    for src, targets in non_descendants.items():
        for dst in targets:
            if src != dst:
                forbidden_edges.add((src, dst))

    return ConstraintSummary(descendants, non_descendants, forbidden_edges)


def _format_edge_list(edges: Iterable[Tuple[str, str]]) -> str:
    formatted = [f"{a}->{b}" for a, b in edges]
    if not formatted:
        return "(no edges)"
    return ", ".join(sorted(formatted))


def format_constraint_lines(summary: ConstraintSummary) -> List[str]:
    """Convert constraint summary into human-readable bullet points."""
    lines: List[str] = []

    for src, targets in sorted(summary.descendant_map.items()):
        if targets:
            lines.append(f"{src} must influence: {', '.join(sorted(targets))}")

    for src, targets in sorted(summary.non_descendant_map.items()):
        filtered = {t for t in targets if t not in summary.descendant_map.get(src, set())}
        if filtered:
            lines.append(f"{src} cannot reach: {', '.join(sorted(filtered))}")

    return lines


def summarise_graph_bank(nodes: Iterable[str], graphs: List[CausalGraph], coverage_threshold: float = 0.65) -> Dict[str, List[Tuple[str, str]]]:
    """Generate quick heuristics from the set of compatible graphs."""
    if not graphs:
        return {"always": [], "frequent": [], "forbidden": []}

    total = len(graphs)
    edge_counts: Counter = Counter()
    for graph in graphs:
        edge_counts.update(graph.edges)

    always_edges = [edge for edge, count in edge_counts.items() if count == total]
    frequent_edges = [
        edge for edge, count in edge_counts.items()
        if count < total and (count / total) >= coverage_threshold
    ]

    all_possible_edges = {(a, b) for a in nodes for b in nodes if a != b}
    forbidden_edges = sorted(all_possible_edges - set(edge_counts.keys()))

    return {
        "always": sorted(always_edges),
        "frequent": sorted(frequent_edges),
        "forbidden": forbidden_edges,
    }


def format_edge_pattern_lines(patterns: Dict[str, List[Tuple[str, str]]]) -> List[str]:
    """Produce readable summaries of edge frequency heuristics."""
    lines: List[str] = []
    if patterns.get("always"):
        lines.append(f"Always include: {_format_edge_list(patterns['always'])}")
    if patterns.get("frequent"):
        lines.append(f"Prefer edges (>=65%): {_format_edge_list(patterns['frequent'])}")
    if patterns.get("forbidden"):
        lines.append(f"Prohibited edges: {_format_edge_list(patterns['forbidden'])}")
    return lines


def format_graph_string(graph: CausalGraph) -> str:
    """Return canonical string representation suitable for prompt options."""
    if not graph.edges:
        return "Graph: No edges"
    edge_str = ", ".join(f"{src}->{dst}" for src, dst in sorted(graph.edges))
    return f"Graph: {edge_str}"