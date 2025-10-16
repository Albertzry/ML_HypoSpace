import json
import argparse
import numpy as np
import itertools
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
import hashlib
import random


class Structure3D:
    """Represents a 3D structure with layers."""
    
    def __init__(self, layers: List[np.ndarray]):
        """
        layers: List of 2D numpy arrays representing each layer (height level)
        """
        self.layers = [np.array(layer, dtype=int) for layer in layers]
        self.height = len(self.layers)
        self.shape = self.layers[0].shape if self.layers else (0, 0)
        self.hash = self._compute_hash()
        
    def _compute_hash(self) -> str:
        """Compute hash for the structure."""
        if not self.layers:
            return "empty000"
        # Stack layers into 3D array for stable hashing
        arr = np.stack(self.layers, axis=0).astype(np.uint8)  # (H, R, C)
        h = hashlib.md5()
        h.update(arr.tobytes())
        h.update(np.array(arr.shape, dtype=np.int64).tobytes())
        return h.hexdigest()[:8]
    
    def get_top_view(self) -> np.ndarray:
        """Get top view (OR of all layers)."""
        if not self.layers:
            return np.array([])
        
        top_view = np.zeros_like(self.layers[0])
        for layer in self.layers:
            top_view = np.logical_or(top_view, layer).astype(int)
        return top_view
    
    
    def to_dict(self) -> Dict:
        """Convert to compact dictionary representation."""
        # Flatten layers into a single string per layer for compactness
        layers_compact = []
        for layer in self.layers:
            # Convert each layer to a string of 0s and 1s
            layer_str = ''.join(''.join(str(cell) for cell in row) for row in layer)
            layers_compact.append(layer_str)
        
        return {
            "layers": layers_compact,
            "hash": self.hash
        }


class Observation3D:
    """Represents a 3D observation (view of a structure)."""
    
    def __init__(self, structure: Structure3D, view_type: str = "top"):
        self.structure = structure
        self.view_type = view_type
        self.observation = self._get_observation()
        self.observation_id = f"{structure.hash}_{view_type}"
        
    def _get_observation(self) -> np.ndarray:
        """Get the observation based on view type."""
        if self.view_type == "top":
            return self.structure.get_top_view()
        else:
            raise ValueError(f"Unknown view type: {self.view_type}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "observation_id": self.observation_id,
            "structure_hash": self.structure.hash,
            "view_type": self.view_type,
            "observation": self.observation.tolist()
        }


def enumerate_all_structures_by_heights(top_view: np.ndarray, max_height: int = 3) -> List[Structure3D]:
    """
    Given a top view, enumerate all possible 3D structures.
    
    top_view: 2D numpy array (N, N) with 0/1 values
    max_height: Maximum height of blocks
    
    Returns: List of Structure3D objects
    """
    coords = [(r, c) for r, c in zip(*np.where(top_view == 1))]
    k = len(coords)
    
    structures = []
    
    # For each position with a block, enumerate heights from 1 to max_height
    for heights_combo in itertools.product(range(1, max_height + 1), repeat=k):
        layers = [np.zeros_like(top_view) for _ in range(max_height)]
        
        # Fill layers based on heights
        for (r, c), h in zip(coords, heights_combo):
            for z in range(h):
                layers[z][r, c] = 1
        
        # Create structure (only keep non-empty layers)
        non_empty_layers = []
        for layer in layers:
            if np.any(layer):
                non_empty_layers.append(layer)
            else:
                break  # Stop at first empty layer
        
        if non_empty_layers:
            structures.append(Structure3D(non_empty_layers))
    
    return structures


class Dataset3DGenerator:
    """Generate complete 3D structure discovery datasets."""
    
    def __init__(self, grid_size: int = 3, max_blocks: int = 3, max_height: int = 3):
        self.grid_size = grid_size
        self.max_blocks = max_blocks
        self.max_height = max_height
        self.all_top_views = []
        self.all_structures = []
        self.observation_sets = []
        
    def generate_all_top_views(self, fixed_blocks: bool = False) -> List[np.ndarray]:
        """Generate all possible top views with specified constraints.
        
        Args:
            fixed_blocks: If True, generate only top views with exactly max_blocks blocks.
                         If False, generate top views with 1 to max_blocks blocks.
        """
        top_views = []
        total_cells = self.grid_size * self.grid_size
        
        # Determine the range of blocks to generate
        if fixed_blocks:
            # Generate only with exact number of blocks
            block_range = [self.max_blocks]
        else:
            # Generate from 1 to max_blocks
            block_range = range(1, min(self.max_blocks + 1, total_cells + 1))
        
        # Generate all combinations of positions for blocks
        for num_blocks in block_range:
            for positions in itertools.combinations(range(total_cells), num_blocks):
                top_view = np.zeros((self.grid_size, self.grid_size), dtype=int)
                for pos in positions:
                    row, col = divmod(pos, self.grid_size)
                    top_view[row, col] = 1
                top_views.append(top_view)
        
        self.all_top_views = top_views
        return top_views
    
    def generate_all_structures(self) -> List[Structure3D]:
        """Generate all possible 3D structures from all top views."""
        all_structures = []
        
        for top_view in self.all_top_views:
            structures = enumerate_all_structures_by_heights(top_view, self.max_height)
            all_structures.extend(structures)
        
        self.all_structures = all_structures
        return all_structures
    
    def generate_observation_sets(self) -> List[Dict]:
        """Generate observation sets for benchmark."""
        observation_sets = []
        seen_obs_ids = set()  # Track seen observation IDs to avoid duplicates
        
        # Group structures by their top view
        top_view_groups = {}
        for struct in self.all_structures:
            top_view_key = tuple(struct.get_top_view().flatten())
            if top_view_key not in top_view_groups:
                top_view_groups[top_view_key] = []
            top_view_groups[top_view_key].append(struct)
        
        # Create observation sets with systematic numbering
        obs_counter = {}  # Track counter for each blocks configuration
        
        for top_view_key, structures in top_view_groups.items():
            top_view = np.array(top_view_key).reshape(self.grid_size, self.grid_size)
            
            # Create observation set with just top view
            # Convert top view to compact string format
            top_view_str = ''.join(''.join(str(cell) for cell in row) for row in top_view)
            
            # Generate systematic ID
            num_blocks = int(np.sum(top_view))
            key = f"b{num_blocks}"
            if key not in obs_counter:
                obs_counter[key] = 0
            obs_counter[key] += 1
            
            obs_set = {
                "observation_id": f"g{self.grid_size}_h{self.max_height}_b{num_blocks}_{obs_counter[key]:03d}",
                "observation": top_view_str,  # Compact string representation
                "ground_truth_structures": [s.to_dict() for s in structures],
                "n_compatible_structures": len(structures)
            }
            if obs_set["observation_id"] not in seen_obs_ids:
                observation_sets.append(obs_set)
                seen_obs_ids.add(obs_set["observation_id"])
        
        self.observation_sets = observation_sets
        return observation_sets
    
    def save_dataset(self, output_path: str):
        """Save complete dataset to JSON file."""
        dataset = {
            "metadata": {
                "grid_size": self.grid_size,
                "max_blocks": self.max_blocks,
                "max_height": self.max_height,
                "n_observation_sets": len(self.observation_sets),
                "generated_at": datetime.now().isoformat()
            },
            "observation_sets": self.observation_sets
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Use indent for human-readable formatting
            json.dump(dataset, f, indent=2)
        
        # Calculate file size
        file_size = output_path.stat().st_size
        size_kb = file_size / 1024
        
        print(f"Dataset saved to: {output_path}")
        print(f"  - Top views: {len(self.all_top_views)}")
        print(f"  - Total structures: {len(self.all_structures)}")
        print(f"  - Observation sets: {len(self.observation_sets)}")
        print(f"  - File size: {size_kb:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description="Generate complete 3D structure dataset")
    parser.add_argument("--grid-size", type=int, default=3,
                       help="Size of the grid (NxN)")
    parser.add_argument("--max-blocks", type=int, default=3,
                       help="Maximum number of blocks in top view (or exact number if --fixed is used)")
    parser.add_argument("--fixed", action="store_true",
                       help="If set, generate only structures with exactly max-blocks blocks (not 1 to max-blocks)")
    parser.add_argument("--max-height", type=int, default=3,
                       help="Maximum height of structures")
    parser.add_argument("--output", type=str, default="datasets/3d_complete.json",
                       help="Output file path")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create generator
    generator = Dataset3DGenerator(
        grid_size=args.grid_size,
        max_blocks=args.max_blocks,
        max_height=args.max_height
    )
    
    # Generate dataset
    if args.fixed:
        print(f"Generating all top views for {args.grid_size}x{args.grid_size} grid with EXACTLY {args.max_blocks} blocks...")
    else:
        print(f"Generating all top views for {args.grid_size}x{args.grid_size} grid with 1 to {args.max_blocks} blocks...")
    generator.generate_all_top_views(fixed_blocks=args.fixed)
    
    print("Generating all 3D structures...")
    generator.generate_all_structures()
    
    print("Creating observation sets...")
    generator.generate_observation_sets()
    
    # Save dataset
    generator.save_dataset(args.output)


if __name__ == "__main__":
    main()