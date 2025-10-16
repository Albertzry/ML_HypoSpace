import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import json
import numpy as np
import networkx as nx
from datetime import datetime
import random
from collections import defaultdict
from itertools import combinations
from modules.models import CausalGraph
from generate_causal_dataset import PerturbationObservation

class HybridDatasetGenerator:
    """Hybrid generator: exhaustive GT generation + efficient observation sampling."""

    @staticmethod
    def generate_all_dags(nodes: List[str], max_edges: Optional[int] = None) -> List[CausalGraph]:
        """
        Generate ALL possible DAGs with the given nodes (exhaustive).

        Args:
            nodes: List of node names
            max_edges: Maximum number of edges (None for no limit)

        Returns:
            List of all possible DAGs
        """
        all_dags = []
        all_possible_edges = [(i, j) for i in nodes for j in nodes if i != j]

        if max_edges is None:
            max_edges = len(all_possible_edges)

        print(f"Generating all DAGs with ≤{max_edges} edges...")
        total_combinations = sum(1 for k in range(min(max_edges + 1, len(all_possible_edges) + 1))
                                for _ in combinations(all_possible_edges, k))
        print(f"Checking {total_combinations} potential edge combinations...")

        checked = 0
        # Try all possible edge combinations
        for edge_count in range(min(max_edges + 1, len(all_possible_edges) + 1)):
            for edge_combo in combinations(all_possible_edges, edge_count):
                checked += 1
                if checked % 1000000 == 0:
                    print(f"  Checked {checked}/{total_combinations} combinations...")

                # Check if this forms a valid DAG
                test_graph = CausalGraph(nodes=nodes, edges=list(edge_combo))
                G = test_graph.to_networkx()

                if nx.is_directed_acyclic_graph(G):
                    all_dags.append(test_graph)

        print(f"Found {len(all_dags)} valid DAGs")
        return all_dags

    @staticmethod
    def get_perturbation_effects(
        graph: CausalGraph,
        perturbed_node: str,
        desc_map: Optional[Dict[str, Set[str]]] = None
    ) -> PerturbationObservation:
        """Get effects of perturbing a node."""
        G = graph.to_networkx()
        if desc_map is None:
            desc_map = {n: nx.descendants(G, n) for n in G.nodes}

        effects = {n: (1 if n in desc_map[perturbed_node] else 0) for n in graph.nodes}
        effects[perturbed_node] = 0
        return PerturbationObservation(perturbed_node, effects)

    @staticmethod
    def generate_observation_sets_from_all_dags(
        nodes: List[str],
        all_dags: List[CausalGraph],
        n_observations: int,
        n_observation_sets: int = 100,
        seed: Optional[int] = None,
        ensure_diversity: bool = True
    ) -> List[Dict]:
        """
        Generate observation sets by sampling from ALL possible DAGs.

        Args:
            nodes: List of node names
            all_dags: ALL possible DAGs (complete set)
            n_observations: Number of observations per set
            n_observation_sets: Target number of observation sets
            seed: Random seed
            ensure_diversity: Try to get diverse GT counts

        Returns:
            List of dataset dictionaries with complete GT coverage
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        print(f"Generating observation sets from {len(all_dags)} DAGs...")

        # Step 1: Precompute all possible observations from ALL DAGs
        print("Computing all possible observations...")
        observation_signatures = {}  # signature -> PerturbationObservation
        dag_to_obs_sigs = {}  # dag_hash -> {node -> obs_signature}

        for i, dag in enumerate(all_dags):
            if (i + 1) % 100000 == 0:
                print(f"  Processing DAG {i+1}/{len(all_dags)}...")

            G = dag.to_networkx()
            desc_map = {n: nx.descendants(G, n) for n in G.nodes}
            dag_hash = dag.get_hash()
            dag_to_obs_sigs[dag_hash] = {}

            for node in nodes:
                obs = HybridDatasetGenerator.get_perturbation_effects(dag, node, desc_map)
                sig = obs.to_tuple()
                observation_signatures[sig] = obs
                dag_to_obs_sigs[dag_hash][node] = sig

        print(f"Found {len(observation_signatures)} unique observations across all DAGs")

        # Step 2: Generate observation sets
        print(f"Generating {n_observation_sets} observation sets...")
        datasets = []
        dataset_counter = 0
        used_combinations = set()

        # Strategy: Sample from different DAGs to ensure diversity
        attempts = 0
        max_attempts = n_observation_sets * 50

        # Create pools of DAGs by edge count for diversity
        dags_by_edge_count = defaultdict(list)
        for dag in all_dags:
            edge_count = len(dag.edges)
            dags_by_edge_count[edge_count].append(dag)

        while len(datasets) < n_observation_sets and attempts < max_attempts:
            attempts += 1

            # Select a random edge count (weighted by availability)
            edge_counts = list(dags_by_edge_count.keys())
            if not edge_counts:
                break

            # Prefer diversity in edge counts
            if ensure_diversity and len(datasets) > 0:
                # Try to pick an edge count we haven't used much
                edge_count_usage = defaultdict(int)
                for ds in datasets:
                    for gt in ds['ground_truth_graphs']:
                        edge_count_usage[len(gt['edges'])] += 1

                # Weight towards less-used edge counts
                weights = [1.0 / (edge_count_usage[ec] + 1) for ec in edge_counts]
                edge_count = random.choices(edge_counts, weights=weights)[0]
            else:
                edge_count = random.choice(edge_counts)

            # Pick a random DAG from this edge count
            if not dags_by_edge_count[edge_count]:
                continue

            primary_dag = random.choice(dags_by_edge_count[edge_count])

            # Generate observation set from this DAG
            if n_observations > len(nodes):
                continue

            # Randomly select nodes to perturb
            selected_nodes = tuple(sorted(random.sample(nodes, n_observations)))

            # Get observations for these nodes from the primary DAG
            primary_hash = primary_dag.get_hash()
            obs_sigs = tuple(dag_to_obs_sigs[primary_hash][node] for node in selected_nodes)

            # Check if we've used this combination
            combo_key = (selected_nodes, obs_sigs)
            if combo_key in used_combinations:
                continue
            used_combinations.add(combo_key)

            # Get actual observations
            observations = [observation_signatures[sig] for sig in obs_sigs]

            # Find ALL compatible DAGs from the complete set
            compatible_dags = []
            for dag in all_dags:
                dag_hash = dag.get_hash()
                # Check if this DAG predicts the same observations
                all_match = all(
                    dag_to_obs_sigs[dag_hash].get(node) == sig
                    for node, sig in zip(selected_nodes, obs_sigs)
                )
                if all_match:
                    compatible_dags.append(dag)

            if compatible_dags:
                dataset_counter += 1
                obs_set_id = f"n{n_observations}_{dataset_counter:03d}"

                dataset = {
                    "observation_set_id": obs_set_id,
                    "n_observations": n_observations,
                    "observations": [
                        {
                            "perturbed_node": o.perturbed_node,
                            "effects": o.effects,
                            "string": o.to_string(),
                        } for o in observations
                    ],
                    "ground_truth_graphs": [dag.to_dict() for dag in compatible_dags],
                    "n_compatible_graphs": len(compatible_dags),
                    "nodes": nodes,
                    "max_edges": max([len(d.edges) for d in all_dags])
                }
                datasets.append(dataset)

        print(f"Generated {len(datasets)} observation sets")

        # Print GT distribution
        gt_counts = defaultdict(int)
        for ds in datasets:
            gt_counts[ds['n_compatible_graphs']] += 1
        print(f"GT distribution: {dict(sorted(gt_counts.items()))}")

        return datasets

    @staticmethod
    def generate_dataset_collection_hybrid(
        nodes: List[str],
        n_observations: Optional[int] = None,
        fixed: bool = False,
        max_edges: Optional[int] = None,
        seed: Optional[int] = None,
        n_observation_sets: int = 100
    ) -> Dict:
        """
        Generate dataset collection using hybrid approach.

        Args:
            nodes: List of node names
            n_observations: Number of observations (None defaults to n_nodes)
            fixed: If True, generate only n_observations exactly
            max_edges: Maximum edges in DAGs
            seed: Random seed
            n_observation_sets: Target observation sets per n_observations value

        Returns:
            Dataset collection with COMPLETE ground truth coverage
        """
        n_nodes = len(nodes)

        if n_observations is None:
            n_observations = n_nodes

        print("=" * 60)
        print("HYBRID CAUSAL DATASET GENERATION")
        print("(Exhaustive GT + Efficient Observation Sampling)")
        print("=" * 60)
        print(f"Nodes: {nodes} ({len(nodes)} nodes)")
        print(f"Max edges: {max_edges}")
        print(f"Target observation sets: {n_observation_sets}")

        if fixed:
            print(f"Observations: exactly {n_observations}")
        else:
            print(f"Observations: 1 to {n_observations}")
        print()

        # Generate ALL possible DAGs (exhaustive)
        all_dags = HybridDatasetGenerator.generate_all_dags(nodes, max_edges)
        print(f"Total hypothesis space: {len(all_dags)} DAGs")
        print(f"This represents ALL possible DAGs with ≤{max_edges} edges")
        print()

        if fixed:
            # Generate only for exact observation count
            print(f"Generating datasets with exactly {n_observations} observations...")
            datasets = HybridDatasetGenerator.generate_observation_sets_from_all_dags(
                nodes, all_dags, n_observations, n_observation_sets, seed
            )

            result = {
                "metadata": {
                    "nodes": nodes,
                    "n_observations": n_observations,
                    "max_edges": max_edges,
                    "hypothesis_space_size": len(all_dags),
                    "n_datasets": len(datasets),
                    "fixed": True,
                    "generation_method": "hybrid_exhaustive_gt",
                    "complete_gt_coverage": True
                },
                "datasets": datasets
            }
        else:
            # Generate for range 1 to n_observations
            collection = {
                "metadata": {
                    "nodes": nodes,
                    "max_n_observations": n_observations,
                    "max_edges": max_edges,
                    "hypothesis_space_size": len(all_dags),
                    "total_observation_sets": 0,
                    "generation_method": "hybrid_exhaustive_gt",
                    "complete_gt_coverage": True
                },
                "datasets_by_n_observations": {}
            }

            # Generate datasets for each observation count
            for n_obs in range(1, min(n_observations, n_nodes) + 1):
                print(f"\nGenerating datasets with {n_obs} observation(s)...")
                datasets = HybridDatasetGenerator.generate_observation_sets_from_all_dags(
                    nodes, all_dags, n_obs, n_observation_sets, seed
                )

                if not datasets:
                    print(f"  No valid observation sets found for n={n_obs}")
                    continue

                collection["datasets_by_n_observations"][n_obs] = datasets

            # Update total
            total_sets = sum(len(datasets) for datasets in collection["datasets_by_n_observations"].values())
            collection["metadata"]["total_observation_sets"] = total_sets
            print(f"\nTotal observation sets: {total_sets}")

            result = collection

        return result


def main():
    parser = argparse.ArgumentParser(description="Hybrid causal dataset generator with complete GT coverage")
    parser.add_argument("--nodes", type=int, default=5,
                       help="Number of nodes in graphs (recommend ≤7 for exhaustive)")
    parser.add_argument("--n-observations", type=int, default=None,
                       help="Number of observations (default: number of nodes)")
    parser.add_argument("--fixed", action="store_true",
                       help="Generate only exact n-observations count")
    parser.add_argument("--max-edges", type=int, default=None,
                       help="Maximum edges in DAGs")
    parser.add_argument("--n-observation-sets", type=int, default=100,
                       help="Target number of observation sets (default: 100)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")

    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.fixed and args.n_observations:
            args.output = f"datasets/causal_hybrid_n{args.nodes}_obs{args.n_observations}_fixed_{timestamp}.json"
        else:
            args.output = f"datasets/causal_hybrid_n{args.nodes}_{timestamp}.json"

    # Generate node names
    nodes = [chr(65 + i) for i in range(args.nodes)]  # A, B, C, ...

    # Set default max_edges based on node count
    if args.max_edges is None:
        # Default: reasonable limit for exhaustive generation
        if args.nodes <= 5:
            args.max_edges = args.nodes * (args.nodes - 1) // 2  # ~50% of max
        else:
            args.max_edges = min(10, args.nodes * 2)  # More conservative for larger graphs
        print(f"Using default max_edges={args.max_edges}")

    # Warn for large graphs
    if args.nodes > 7:
        print("WARNING: Exhaustive DAG generation for >7 nodes can be very slow!")
        print("Consider using generate_causal_dataset_gt_first.py for larger graphs.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Generate dataset
    result = HybridDatasetGenerator.generate_dataset_collection_hybrid(
        nodes=nodes,
        n_observations=args.n_observations,
        fixed=args.fixed,
        max_edges=args.max_edges,
        seed=args.seed,
        n_observation_sets=args.n_observation_sets
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