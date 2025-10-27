"""
Optimization module for 3D structure generation.

Implements three core optimization approaches:
1. Loss Function Design - Quantitative evaluation of structures
2. Search Algorithm - Local optimization via neighbor exploration
3. Reward Signal - RL-inspired reward shaping
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Set
from copy import deepcopy


class StructureLoss:
    """
    Loss function for evaluating 3D structures.
    Similar to supervised learning loss: L = L_obs + λ1*L_physics + λ2*L_reg
    """
    
    def __init__(self, 
                 obs_weight: float = 1.0,
                 physics_weight: float = 0.5,
                 complexity_weight: float = 0.2,
                 diversity_weight: float = 0.1):
        """
        Initialize loss function with weights.
        
        Args:
            obs_weight: Weight for observation matching loss
            physics_weight: Weight for physical constraint violations
            complexity_weight: Weight for structural complexity regularization
            diversity_weight: Weight for diversity encouragement
        """
        self.obs_weight = obs_weight
        self.physics_weight = physics_weight
        self.complexity_weight = complexity_weight
        self.diversity_weight = diversity_weight
        self.known_structures: List = []
    
    def compute_loss(self, structure, observation: np.ndarray) -> float:
        """
        Compute total loss for a structure.
        Lower loss is better.
        
        Args:
            structure: Structure3D object
            observation: Top view observation (2D numpy array)
        
        Returns:
            Total loss value
        """
        loss = 0.0
        
        # 1. Observation matching loss (reconstruction error)
        obs_loss = self._observation_loss(structure, observation)
        loss += self.obs_weight * obs_loss
        
        # 2. Physics constraint loss (penalty for violations)
        physics_loss = self._physics_loss(structure)
        loss += self.physics_weight * physics_loss
        
        # 3. Complexity regularization (avoid unnecessarily complex structures)
        complexity_loss = self._complexity_loss(structure)
        loss += self.complexity_weight * complexity_loss
        
        # 4. Diversity loss (penalty for similarity to known structures)
        if self.known_structures:
            diversity_loss = self._diversity_loss(structure)
            loss += self.diversity_weight * diversity_loss
        
        return loss
    
    def _observation_loss(self, structure, observation: np.ndarray) -> float:
        """L2 loss between generated top view and observation."""
        top_view = structure.get_top_view()
        
        # Ensure same shape
        if top_view.shape != observation.shape:
            return 1000.0  # Large penalty for shape mismatch
        
        # Mean squared error
        mse = np.mean((top_view - observation) ** 2)
        return mse
    
    def _physics_loss(self, structure) -> float:
        """Penalty for violating gravity constraints."""
        penalty = 0.0
        
        if not structure.layers:
            return 0.0
        
        # Count floating blocks (blocks without support below)
        for z in range(1, len(structure.layers)):
            current = structure.layers[z]
            below = structure.layers[z-1]
            
            # Each block in current layer must have support in layer below
            floating_blocks = np.sum((current == 1) & (below == 0))
            penalty += floating_blocks
        
        return penalty
    
    def _complexity_loss(self, structure) -> float:
        """Regularization based on structure complexity."""
        if not structure.layers:
            return 0.0
        
        # Total number of blocks
        total_blocks = sum(np.sum(layer) for layer in structure.layers)
        
        # Height of structure
        height = structure.height
        
        # Simple heuristic: penalize unnecessarily tall structures with few blocks
        if total_blocks > 0 and height > 0:
            avg_blocks_per_layer = total_blocks / height
            if avg_blocks_per_layer < 1.5 and height > 2:
                return height * 0.5  # Penalty for tall but sparse structures
        
        return 0.0
    
    def _diversity_loss(self, structure) -> float:
        """Penalty for similarity to known structures."""
        max_similarity = 0.0
        structure_hash = structure.get_hash()
        
        for known in self.known_structures:
            if structure_hash == known.get_hash():
                return 10.0  # Large penalty for exact duplicates
            
            # Compute layer-wise similarity
            similarity = self._structure_similarity(structure, known)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _structure_similarity(self, s1, s2) -> float:
        """Compute similarity between two structures (0-1)."""
        # Align heights
        max_height = max(len(s1.layers), len(s2.layers))
        
        # Pad shorter structure
        layers1 = s1.layers + [np.zeros_like(s1.layers[0])] * (max_height - len(s1.layers))
        layers2 = s2.layers + [np.zeros_like(s2.layers[0])] * (max_height - len(s2.layers))
        
        # Compute Jaccard similarity for each layer
        similarities = []
        for l1, l2 in zip(layers1, layers2):
            intersection = np.sum((l1 == 1) & (l2 == 1))
            union = np.sum((l1 == 1) | (l2 == 1))
            if union > 0:
                similarities.append(intersection / union)
            else:
                similarities.append(1.0)  # Both empty
        
        return np.mean(similarities)
    
    def update_known_structures(self, structure):
        """Add structure to known structures for diversity computation."""
        self.known_structures.append(structure)


class StructureOptimizer:
    """
    Local search optimizer for 3D structures.
    Uses simulated annealing-like approach to escape local minima.
    """
    
    def __init__(self, 
                 loss_fn: StructureLoss,
                 max_iterations: int = 30,
                 initial_temp: float = 1.0):
        """
        Initialize optimizer.
        
        Args:
            loss_fn: Loss function to minimize
            max_iterations: Maximum optimization iterations
            initial_temp: Initial temperature for simulated annealing
        """
        self.loss_fn = loss_fn
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
    
    def optimize(self, initial_structure, observation: np.ndarray) -> Tuple:
        """
        Optimize structure using local search.
        
        Args:
            initial_structure: Starting structure
            observation: Target observation
        
        Returns:
            (optimized_structure, final_loss)
        """
        current_structure = initial_structure
        current_loss = self.loss_fn.compute_loss(current_structure, observation)
        
        best_structure = current_structure
        best_loss = current_loss
        
        for iteration in range(self.max_iterations):
            # Generate neighbor structures
            neighbors = self._generate_neighbors(current_structure, observation)
            
            if not neighbors:
                break  # No valid neighbors
            
            # Evaluate all neighbors
            neighbor_losses = []
            for neighbor in neighbors:
                loss = self.loss_fn.compute_loss(neighbor, observation)
                neighbor_losses.append((neighbor, loss))
            
            # Select best neighbor
            best_neighbor, best_neighbor_loss = min(neighbor_losses, key=lambda x: x[1])
            
            # Accept if better or with probability (simulated annealing)
            if best_neighbor_loss < current_loss:
                current_structure = best_neighbor
                current_loss = best_neighbor_loss
                
                # Update global best
                if current_loss < best_loss:
                    best_structure = current_structure
                    best_loss = current_loss
            else:
                # Simulated annealing acceptance
                temperature = self.initial_temp * (1 - iteration / self.max_iterations)
                acceptance_prob = np.exp(-(best_neighbor_loss - current_loss) / (temperature + 0.01))
                
                if random.random() < acceptance_prob:
                    current_structure = best_neighbor
                    current_loss = best_neighbor_loss
        
        return best_structure, best_loss
    
    def _generate_neighbors(self, structure, observation: np.ndarray) -> List:
        """Generate neighbor structures by local modifications."""
        neighbors = []
        
        if not structure.layers:
            return neighbors
        
        # Get positions with blocks in observation
        block_positions = list(zip(*np.where(observation == 1)))
        
        # Operation 1: Add a block at a valid position
        for r, c in block_positions:
            for z in range(min(len(structure.layers), 3)):  # Limit to reasonable heights
                # Check if we can add a block here
                if z < len(structure.layers):
                    if structure.layers[z][r, c] == 0:
                        # Check support (if z > 0, must have block below)
                        if z == 0 or structure.layers[z-1][r, c] == 1:
                            neighbor = self._add_block(structure, z, r, c)
                            if neighbor:
                                neighbors.append(neighbor)
        
        # Operation 2: Remove a block from top layers
        for z in range(len(structure.layers) - 1, 0, -1):  # Start from top
            for r, c in zip(*np.where(structure.layers[z] == 1)):
                # Check if we can remove (no blocks above depend on it)
                can_remove = True
                if z < len(structure.layers) - 1:
                    if structure.layers[z+1][r, c] == 1:
                        can_remove = False
                
                if can_remove:
                    neighbor = self._remove_block(structure, z, r, c)
                    if neighbor:
                        neighbors.append(neighbor)
        
        return neighbors
    
    def _add_block(self, structure, z: int, r: int, c: int):
        """Add a block at (z, r, c) and return new structure."""
        try:
            # Deep copy layers
            new_layers = [layer.copy() for layer in structure.layers]
            new_layers[z][r, c] = 1
            
            # Import Structure3D locally to avoid circular import
            from run_3d_benchmark import Structure3D
            return Structure3D([layer.tolist() for layer in new_layers])
        except:
            return None
    
    def _remove_block(self, structure, z: int, r: int, c: int):
        """Remove a block at (z, r, c) and return new structure."""
        try:
            # Deep copy layers
            new_layers = [layer.copy() for layer in structure.layers]
            new_layers[z][r, c] = 0
            
            # Import Structure3D locally to avoid circular import
            from run_3d_benchmark import Structure3D
            return Structure3D([layer.tolist() for layer in new_layers])
        except:
            return None


class RewardFunction:
    """
    RL-inspired reward function for structure evaluation.
    Higher reward is better.
    """
    
    def __init__(self,
                 valid_reward: float = 10.0,
                 match_reward: float = 5.0,
                 gravity_penalty: float = 2.0,
                 diversity_bonus: float = 1.0):
        """
        Initialize reward function.
        
        Args:
            valid_reward: Reward for valid structures
            match_reward: Reward for observation matching
            gravity_penalty: Penalty per gravity violation
            diversity_bonus: Bonus for novel structures
        """
        self.valid_reward = valid_reward
        self.match_reward = match_reward
        self.gravity_penalty = gravity_penalty
        self.diversity_bonus = diversity_bonus
        self.known_hashes: Set[str] = set()
    
    def compute_reward(self, structure, observation: np.ndarray, is_valid: bool) -> float:
        """
        Compute reward for a structure.
        
        Args:
            structure: Structure3D object
            observation: Target observation
            is_valid: Whether structure matches observation
        
        Returns:
            Total reward value
        """
        reward = 0.0
        
        # 1. Validity reward
        if is_valid:
            reward += self.valid_reward
        
        # 2. Observation matching reward (partial credit)
        match_score = self._compute_match_score(structure, observation)
        reward += self.match_reward * match_score
        
        # 3. Gravity penalty
        gravity_violations = self._count_gravity_violations(structure)
        reward -= self.gravity_penalty * gravity_violations
        
        # 4. Diversity bonus
        structure_hash = structure.get_hash()
        if structure_hash not in self.known_hashes:
            reward += self.diversity_bonus
            self.known_hashes.add(structure_hash)
        
        # 5. Structure quality bonus
        quality = self._structure_quality(structure)
        reward += quality
        
        return reward
    
    def _compute_match_score(self, structure, observation: np.ndarray) -> float:
        """Compute how well structure matches observation (0-1)."""
        top_view = structure.get_top_view()
        
        if top_view.shape != observation.shape:
            return 0.0
        
        # Intersection over union (IoU)
        intersection = np.sum((top_view == 1) & (observation == 1))
        union = np.sum((top_view == 1) | (observation == 1))
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def _count_gravity_violations(self, structure) -> int:
        """Count number of floating blocks."""
        violations = 0
        
        if not structure.layers or len(structure.layers) < 2:
            return 0
        
        for z in range(1, len(structure.layers)):
            current = structure.layers[z]
            below = structure.layers[z-1]
            violations += np.sum((current == 1) & (below == 0))
        
        return violations
    
    def _structure_quality(self, structure) -> float:
        """Compute structural quality score."""
        if not structure.layers:
            return 0.0
        
        quality = 0.0
        
        # Reward connected structures
        connectivity = self._compute_connectivity(structure)
        quality += connectivity * 0.5
        
        # Reward balanced structures
        balance = self._compute_balance(structure)
        quality += balance * 0.3
        
        return quality
    
    def _compute_connectivity(self, structure) -> float:
        """Measure how connected blocks are (0-1)."""
        if not structure.layers:
            return 0.0
        
        total_blocks = sum(np.sum(layer) for layer in structure.layers)
        if total_blocks == 0:
            return 0.0
        
        connections = 0
        
        # Count horizontal and vertical connections
        for layer in structure.layers:
            for r in range(layer.shape[0]):
                for c in range(layer.shape[1]):
                    if layer[r, c] == 1:
                        # Check right neighbor
                        if c < layer.shape[1] - 1 and layer[r, c+1] == 1:
                            connections += 1
                        # Check bottom neighbor
                        if r < layer.shape[0] - 1 and layer[r+1, c] == 1:
                            connections += 1
        
        # Count vertical connections between layers
        for z in range(len(structure.layers) - 1):
            connections += np.sum((structure.layers[z] == 1) & (structure.layers[z+1] == 1))
        
        # Normalize
        max_possible_connections = total_blocks * 3  # rough estimate
        return min(1.0, connections / max_possible_connections)
    
    def _compute_balance(self, structure) -> float:
        """Measure structural balance (0-1)."""
        if not structure.layers:
            return 0.0
        
        # Compute center of mass for each layer
        centroids = []
        for layer in structure.layers:
            positions = np.where(layer == 1)
            if len(positions[0]) > 0:
                centroid_r = np.mean(positions[0])
                centroid_c = np.mean(positions[1])
                centroids.append((centroid_r, centroid_c))
        
        if len(centroids) < 2:
            return 1.0  # Single layer is balanced
        
        # Compute variance of centroids
        centroid_array = np.array(centroids)
        variance = np.var(centroid_array, axis=0).sum()
        
        # Convert to score (lower variance = better balance)
        balance_score = 1.0 / (1.0 + variance)
        
        return balance_score
    
    def reset(self):
        """Reset known hashes."""
        self.known_hashes.clear()

