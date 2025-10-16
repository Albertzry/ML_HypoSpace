import hashlib
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import networkx as nx


@dataclass
class CausalGraph:
    """
    Represents a causal graph with nodes and directed edges.
    
    Attributes:
        nodes: List of node names in the graph
        edges: List of directed edges as (from_node, to_node) tuples
    """
    nodes: List[str]
    edges: List[Tuple[str, str]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation for serialization."""
        return {
            "nodes": self.nodes,
            "edges": self.edges
        }
    
    def to_string(self) -> str:
        """
        Convert to string representation for comparison.
        Sorts nodes and edges to ensure consistent representation.
        """
        sorted_nodes = sorted(self.nodes)
        sorted_edges = sorted(self.edges)
        return f"Nodes: {sorted_nodes}, Edges: {sorted_edges}"
    
    def get_hash(self) -> str:
        """
        Get unique hash for this causal graph.
        Used for comparing graphs and detecting duplicates.
        """
        return hashlib.md5(self.to_string().encode()).hexdigest()
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph for graph operations."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        return G
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CausalGraph':
        """Create CausalGraph from dictionary representation."""
        return cls(
            nodes=data["nodes"], 
            edges=[(e[0], e[1]) for e in data["edges"]]
        )
    
    def __str__(self) -> str:
        """String representation for printing."""
        edges_str = ", ".join([f"{a}->{b}" for a, b in self.edges])
        return f"CausalGraph(nodes={self.nodes}, edges=[{edges_str}])"
    
    def __eq__(self, other) -> bool:
        """Check equality based on hash."""
        if not isinstance(other, CausalGraph):
            return False
        return self.get_hash() == other.get_hash()


@dataclass
class Observation:
    """
    Represents a set of observations from a causal system.
    
    Attributes:
        data: Dictionary mapping variable names to lists of observed values
        metadata: Additional metadata about the observations
    """
    data: Dict[str, List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation for serialization."""
        return {
            "data": self.data,
            "metadata": self.metadata
        }
    
    def to_natural_language(self) -> str:
        """
        Convert observations to natural language description.
        Includes statistical summaries and correlations between variables.
        """
        descriptions = []
        
        for var, values in self.data.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Analyze correlations with other variables
            correlations = []
            for other_var, other_values in self.data.items():
                if other_var != var and len(values) == len(other_values):
                    corr = np.corrcoef(values, other_values)[0, 1]
                    if abs(corr) > 0.3:  # Significant correlation threshold
                        if corr > 0:
                            correlations.append(
                                f"positively correlated with {other_var} (r={corr:.2f})"
                            )
                        else:
                            correlations.append(
                                f"negatively correlated with {other_var} (r={corr:.2f})"
                            )
            
            desc = f"Variable '{var}': mean={mean_val:.2f}, std={std_val:.2f}"
            if correlations:
                desc += f", {', '.join(correlations)}"
            descriptions.append(desc)
        
        return "Observations:\n" + "\n".join(descriptions)
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all variables.
        
        Returns:
            Dictionary mapping variable names to their statistics
        """
        stats = {}
        for var, values in self.data.items():
            stats[var] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        return stats
    
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Calculate correlation matrix between all variables.
        
        Returns:
            Numpy array containing correlation coefficients
        """
        vars_list = sorted(self.data.keys())
        n_vars = len(vars_list)
        corr_matrix = np.zeros((n_vars, n_vars))
        
        for i, var1 in enumerate(vars_list):
            for j, var2 in enumerate(vars_list):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr = np.corrcoef(self.data[var1], self.data[var2])[0, 1]
                    corr_matrix[i, j] = corr
        
        return corr_matrix
    
    def __str__(self) -> str:
        """String representation for printing."""
        n_samples = len(next(iter(self.data.values())))
        return f"Observation(variables={list(self.data.keys())}, n_samples={n_samples})"


@dataclass
class BenchmarkResult:
    """
    Results from a single benchmark run.
    
    Attributes:
        n_queries: Number of LLM queries made
        n_ground_truths: Number of ground truth graphs
        n_valid: Number of valid hypotheses generated
        n_unique_valid: Number of unique valid hypotheses
        creativity_rate: Proportion of valid hypotheses (n_valid / n_queries)
        recovery_rate: Proportion of ground truths recovered (n_unique_valid / n_ground_truths)
        all_hypotheses: List of all generated hypotheses
        valid_hypotheses: List of valid hypotheses
        ground_truth_graphs: List of ground truth graphs
    """
    n_queries: int
    n_ground_truths: int
    n_valid: int
    n_unique_valid: int
    creativity_rate: float
    recovery_rate: float
    all_hypotheses: List[CausalGraph]
    valid_hypotheses: List[CausalGraph]
    ground_truth_graphs: List[CausalGraph]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "n_queries": self.n_queries,
            "n_ground_truths": self.n_ground_truths,
            "n_valid": self.n_valid,
            "n_unique_valid": self.n_unique_valid,
            "creativity_rate": self.creativity_rate,
            "recovery_rate": self.recovery_rate,
            "all_hypotheses": [h.to_dict() for h in self.all_hypotheses],
            "valid_hypotheses": [h.to_dict() for h in self.valid_hypotheses],
            "ground_truth_graphs": [g.to_dict() for g in self.ground_truth_graphs]
        }
    
    def __str__(self) -> str:
        """String representation for printing."""
        return (f"BenchmarkResult(creativity_rate={self.creativity_rate:.2%}, "
                f"recovery_rate={self.recovery_rate:.2%}, "
                f"n_valid={self.n_valid}/{self.n_queries})")