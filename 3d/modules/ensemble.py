"""
Ensemble methods for 3D structure generation.

Implements:
1. Self-Consistency Ensemble - Generate multiple candidates and vote
2. Beam Search - Maintain top-k candidates during generation
3. Constraint-Based Scoring - Multi-criteria evaluation and ranking
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict


class SelfConsistencyEnsemble:
    """
    Self-consistency ensemble method.
    Generate multiple candidates and select most consistent ones.
    """
    
    def __init__(self, n_candidates: int = 3, voting_method: str = "frequency"):
        """
        Initialize ensemble.
        
        Args:
            n_candidates: Number of candidates to generate per query
            voting_method: Method for voting ("frequency", "similarity", "weighted")
        """
        self.n_candidates = n_candidates
        self.voting_method = voting_method
    
    def generate_ensemble(self, 
                         llm_interface,
                         prompt_fn: Callable,
                         parse_fn: Callable,
                         validate_fn: Callable,
                         observation,
                         prior_structures=None) -> List:
        """
        Generate ensemble of candidates and select best.
        
        Args:
            llm_interface: LLM interface for generation
            prompt_fn: Function to create prompt
            parse_fn: Function to parse LLM response
            validate_fn: Function to validate structure
            observation: Target observation
            prior_structures: Previously generated structures
        
        Returns:
            List of structures sorted by consistency score
        """
        candidates = []
        
        # Generate multiple candidates
        for i in range(self.n_candidates):
            prompt = prompt_fn(observation, prior_structures)
            
            try:
                # Query LLM
                if hasattr(llm_interface, 'query_with_usage'):
                    result = llm_interface.query_with_usage(prompt)
                    response = result['response']
                else:
                    response = llm_interface.query(prompt)
                
                # Parse response
                structure = parse_fn(response)
                
                if structure:
                    # Normalize
                    structure = structure.normalize()
                    
                    # Validate
                    is_valid = validate_fn(structure, observation)
                    
                    candidates.append({
                        'structure': structure,
                        'is_valid': is_valid,
                        'response': response
                    })
            except Exception as e:
                # Don't print error for every candidate to reduce noise
                # Only print first few errors
                if i < 2:
                    print(f"Error generating candidate {i+1}: {e}")
                continue
        
        if not candidates:
            return []
        
        # Apply voting/consistency check
        ranked_candidates = self._apply_voting(candidates)
        
        return [c['structure'] for c in ranked_candidates]
    
    def _apply_voting(self, candidates: List[Dict]) -> List[Dict]:
        """Apply voting to rank candidates."""
        if self.voting_method == "frequency":
            return self._frequency_voting(candidates)
        elif self.voting_method == "similarity":
            return self._similarity_voting(candidates)
        elif self.voting_method == "weighted":
            return self._weighted_voting(candidates)
        else:
            # Default: prioritize valid structures
            return sorted(candidates, key=lambda x: x['is_valid'], reverse=True)
    
    def _frequency_voting(self, candidates: List[Dict]) -> List[Dict]:
        """Vote by hash frequency."""
        # Count frequency of each structure
        hash_counts = defaultdict(int)
        hash_to_candidate = {}
        
        for candidate in candidates:
            structure_hash = candidate['structure'].get_hash()
            hash_counts[structure_hash] += 1
            if structure_hash not in hash_to_candidate:
                hash_to_candidate[structure_hash] = candidate
        
        # Sort by frequency
        sorted_hashes = sorted(hash_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return candidates in order of frequency
        result = []
        seen = set()
        for structure_hash, count in sorted_hashes:
            if structure_hash not in seen:
                candidate = hash_to_candidate[structure_hash]
                candidate['consistency_score'] = count / len(candidates)
                result.append(candidate)
                seen.add(structure_hash)
        
        return result
    
    def _similarity_voting(self, candidates: List[Dict]) -> List[Dict]:
        """Vote by similarity to others."""
        scores = []
        
        for i, candidate in enumerate(candidates):
            # Compute average similarity to all other candidates
            similarity_sum = 0.0
            for j, other in enumerate(candidates):
                if i != j:
                    sim = self._structure_similarity(
                        candidate['structure'], 
                        other['structure']
                    )
                    similarity_sum += sim
            
            avg_similarity = similarity_sum / (len(candidates) - 1) if len(candidates) > 1 else 0
            scores.append((candidate, avg_similarity))
        
        # Sort by similarity score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for candidate, score in scores:
            candidate['consistency_score'] = score
            result.append(candidate)
        
        return result
    
    def _weighted_voting(self, candidates: List[Dict]) -> List[Dict]:
        """Weighted voting combining validity and similarity."""
        scores = []
        
        for i, candidate in enumerate(candidates):
            score = 0.0
            
            # Validity weight
            if candidate['is_valid']:
                score += 0.5
            
            # Similarity weight
            similarity_sum = 0.0
            for j, other in enumerate(candidates):
                if i != j:
                    sim = self._structure_similarity(
                        candidate['structure'],
                        other['structure']
                    )
                    similarity_sum += sim
            
            avg_similarity = similarity_sum / (len(candidates) - 1) if len(candidates) > 1 else 0
            score += 0.5 * avg_similarity
            
            scores.append((candidate, score))
        
        # Sort by combined score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for candidate, score in scores:
            candidate['consistency_score'] = score
            result.append(candidate)
        
        return result
    
    def _structure_similarity(self, s1, s2) -> float:
        """Compute Jaccard similarity between structures."""
        # Quick check: same hash
        if s1.get_hash() == s2.get_hash():
            return 1.0
        
        # Align heights
        max_height = max(len(s1.layers), len(s2.layers))
        if max_height == 0:
            return 1.0
        
        # Compute layer-wise similarity
        similarities = []
        for z in range(max_height):
            l1 = s1.layers[z] if z < len(s1.layers) else np.zeros_like(s1.layers[0])
            l2 = s2.layers[z] if z < len(s2.layers) else np.zeros_like(s2.layers[0])
            
            intersection = np.sum((l1 == 1) & (l2 == 1))
            union = np.sum((l1 == 1) | (l2 == 1))
            
            if union > 0:
                similarities.append(intersection / union)
            else:
                similarities.append(1.0)
        
        return np.mean(similarities)


class BeamSearch:
    """
    Beam search for structured generation.
    Maintains top-k candidates and expands them iteratively.
    """
    
    def __init__(self, beam_size: int = 3, max_iterations: int = 5):
        """
        Initialize beam search.
        
        Args:
            beam_size: Number of candidates to keep in beam
            max_iterations: Maximum number of expansion iterations
        """
        self.beam_size = beam_size
        self.max_iterations = max_iterations
    
    def search(self,
               llm_interface,
               prompt_fn: Callable,
               parse_fn: Callable,
               score_fn: Callable,
               observation,
               initial_structures=None) -> List[Tuple]:
        """
        Perform beam search to find top structures.
        
        Args:
            llm_interface: LLM interface
            prompt_fn: Function to create prompt
            parse_fn: Function to parse response
            score_fn: Function to score structures (higher is better)
            observation: Target observation
            initial_structures: Initial beam (if None, starts from scratch)
        
        Returns:
            List of (structure, score) tuples sorted by score
        """
        # Initialize beam
        if initial_structures:
            beam = [(s, score_fn(s, observation)) for s in initial_structures[:self.beam_size]]
        else:
            beam = []
        
        # Iteratively expand beam
        for iteration in range(self.max_iterations):
            candidates = []
            
            # Generate candidates from each beam element
            for structure, score in beam if beam else [(None, 0)]:
                # Create prompt (possibly conditioned on current structure)
                prompt = prompt_fn(observation, [structure] if structure else None)
                
                try:
                    # Query LLM
                    if hasattr(llm_interface, 'query_with_usage'):
                        result = llm_interface.query_with_usage(prompt)
                        response = result['response']
                    else:
                        response = llm_interface.query(prompt)
                    
                    # Parse
                    new_structure = parse_fn(response)
                    
                    if new_structure:
                        new_structure = new_structure.normalize()
                        new_score = score_fn(new_structure, observation)
                        candidates.append((new_structure, new_score))
                
                except Exception as e:
                    print(f"Error in beam search iteration {iteration}: {e}")
                    continue
            
            if not candidates:
                break
            
            # Select top-k candidates for next beam
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.beam_size]
            
            # Early stopping if we have high-scoring valid structures
            if beam[0][1] > 0.95:  # Threshold for "good enough"
                break
        
        return beam


class ConstraintBasedScoring:
    """
    Multi-criteria scoring for structure evaluation.
    Combines multiple constraints and objectives.
    """
    
    def __init__(self,
                 validity_weight: float = 0.4,
                 physics_weight: float = 0.3,
                 quality_weight: float = 0.2,
                 diversity_weight: float = 0.1):
        """
        Initialize scorer.
        
        Args:
            validity_weight: Weight for observation validity
            physics_weight: Weight for physical constraints
            quality_weight: Weight for structural quality
            diversity_weight: Weight for novelty
        """
        self.validity_weight = validity_weight
        self.physics_weight = physics_weight
        self.quality_weight = quality_weight
        self.diversity_weight = diversity_weight
        self.known_hashes = set()
    
    def score(self, structure, observation: np.ndarray, is_valid: bool = None) -> float:
        """
        Compute multi-criteria score for structure.
        Score is in range [0, 1], higher is better.
        
        Args:
            structure: Structure3D object
            observation: Target observation
            is_valid: Pre-computed validity (if None, will compute)
        
        Returns:
            Weighted score
        """
        total_score = 0.0
        
        # 1. Validity score
        if is_valid is None:
            is_valid = self._check_validity(structure, observation)
        validity_score = 1.0 if is_valid else self._partial_validity_score(structure, observation)
        total_score += self.validity_weight * validity_score
        
        # 2. Physics score (constraint satisfaction)
        physics_score = self._physics_score(structure)
        total_score += self.physics_weight * physics_score
        
        # 3. Quality score (structural properties)
        quality_score = self._quality_score(structure)
        total_score += self.quality_weight * quality_score
        
        # 4. Diversity score
        diversity_score = self._diversity_score(structure)
        total_score += self.diversity_weight * diversity_score
        
        return total_score
    
    def _check_validity(self, structure, observation: np.ndarray) -> bool:
        """Check if structure matches observation."""
        top_view = structure.get_top_view()
        return (top_view.shape == observation.shape and 
                np.array_equal(top_view, observation))
    
    def _partial_validity_score(self, structure, observation: np.ndarray) -> float:
        """Compute partial validity score (IoU)."""
        top_view = structure.get_top_view()
        
        if top_view.shape != observation.shape:
            return 0.0
        
        intersection = np.sum((top_view == 1) & (observation == 1))
        union = np.sum((top_view == 1) | (observation == 1))
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def _physics_score(self, structure) -> float:
        """Score based on physical constraint satisfaction."""
        if not structure.layers or len(structure.layers) < 2:
            return 1.0
        
        total_blocks = 0
        violations = 0
        
        for z in range(1, len(structure.layers)):
            current = structure.layers[z]
            below = structure.layers[z-1]
            
            current_blocks = np.sum(current == 1)
            total_blocks += current_blocks
            
            # Count floating blocks
            floating = np.sum((current == 1) & (below == 0))
            violations += floating
        
        if total_blocks == 0:
            return 1.0
        
        # Score decreases with violations
        score = 1.0 - (violations / (total_blocks + 1))
        return max(0.0, score)
    
    def _quality_score(self, structure) -> float:
        """Score based on structural quality metrics."""
        if not structure.layers:
            return 0.0
        
        score = 0.0
        
        # Factor 1: Connectivity (30%)
        connectivity = self._compute_connectivity(structure)
        score += 0.3 * connectivity
        
        # Factor 2: Stability (40%)
        stability = self._compute_stability(structure)
        score += 0.4 * stability
        
        # Factor 3: Compactness (30%)
        compactness = self._compute_compactness(structure)
        score += 0.3 * compactness
        
        return score
    
    def _diversity_score(self, structure) -> float:
        """Score based on novelty."""
        structure_hash = structure.get_hash()
        
        if structure_hash in self.known_hashes:
            return 0.0  # Not novel
        
        self.known_hashes.add(structure_hash)
        return 1.0  # Novel
    
    def _compute_connectivity(self, structure) -> float:
        """Measure block connectivity."""
        if not structure.layers:
            return 0.0
        
        total_blocks = sum(np.sum(layer) for layer in structure.layers)
        if total_blocks <= 1:
            return 1.0
        
        connections = 0
        
        # Horizontal and vertical connections within layers
        for layer in structure.layers:
            for r in range(layer.shape[0]):
                for c in range(layer.shape[1]):
                    if layer[r, c] == 1:
                        if c < layer.shape[1] - 1 and layer[r, c+1] == 1:
                            connections += 1
                        if r < layer.shape[0] - 1 and layer[r+1, c] == 1:
                            connections += 1
        
        # Vertical connections between layers
        for z in range(len(structure.layers) - 1):
            connections += np.sum((structure.layers[z] == 1) & (structure.layers[z+1] == 1))
        
        # Normalize (rough estimate)
        max_connections = total_blocks * 2
        return min(1.0, connections / max_connections)
    
    def _compute_stability(self, structure) -> float:
        """Measure structural stability."""
        if not structure.layers or len(structure.layers) < 2:
            return 1.0
        
        # Check if all blocks have support
        for z in range(1, len(structure.layers)):
            current = structure.layers[z]
            below = structure.layers[z-1]
            
            # Any floating blocks?
            if np.any((current == 1) & (below == 0)):
                return 0.0
        
        return 1.0
    
    def _compute_compactness(self, structure) -> float:
        """Measure structural compactness."""
        if not structure.layers:
            return 0.0
        
        total_blocks = sum(np.sum(layer) for layer in structure.layers)
        if total_blocks == 0:
            return 0.0
        
        # Compute bounding box volume
        height = structure.height
        rows, cols = structure.shape
        
        bounding_volume = height * rows * cols
        if bounding_volume == 0:
            return 0.0
        
        # Compactness = blocks / bounding_volume
        compactness = total_blocks / bounding_volume
        
        return min(1.0, compactness * 2)  # Scale for better range
    
    def reset(self):
        """Reset known hashes."""
        self.known_hashes.clear()

