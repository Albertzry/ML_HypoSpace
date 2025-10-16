import sys
import json
import argparse
import os
import re
import yaml
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from scipy import stats
import traceback

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent))

from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM


class Structure3D:
    """Represents a 3D structure (same as in dataset generator)."""
    
    def __init__(self, layers: List[List[List[int]]]):
        # Validate and normalize layers to ensure consistent shapes
        if not layers:
            self.layers = []
            self.height = 0
            self.shape = (0, 0)
            return
            
        # Find the maximum dimensions across all layers
        max_rows = 0
        max_cols = 0
        for layer in layers:
            if layer:  # Non-empty layer
                max_rows = max(max_rows, len(layer))
                for row in layer:
                    if row:  # Non-empty row
                        max_cols = max(max_cols, len(row))
        
        # Pad all layers to have consistent shape
        self.layers = []
        for layer in layers:
            # Create padded layer
            padded_layer = np.zeros((max_rows, max_cols), dtype=int)
            for i, row in enumerate(layer[:max_rows]):
                for j, val in enumerate(row[:max_cols]):
                    padded_layer[i, j] = val
            self.layers.append(padded_layer)
        
        self.height = len(self.layers)
        self.shape = (max_rows, max_cols) if self.layers else (0, 0)
        
    def to_string(self) -> str:
        """Convert to string representation for LLM."""
        result = []
        for i, layer in enumerate(self.layers):
            result.append(f"Layer {i+1}:")
            for row in layer:
                result.append(" ".join(str(cell) for cell in row))
        return "\n".join(result)
    
    def get_top_view(self) -> np.ndarray:
        """Get top view (OR of all layers)."""
        if not self.layers:
            return np.zeros((0, 0), dtype=int)
        
        # All layers should have the same shape after constructor normalization
        # But add extra safety check
        top = np.zeros_like(self.layers[0], dtype=int)
        for L in self.layers:
            if L.shape == top.shape:
                top |= (L.astype(bool)).astype(int)
            else:
                # This shouldn't happen after normalization, but handle gracefully
                min_rows = min(L.shape[0], top.shape[0])
                min_cols = min(L.shape[1], top.shape[1])
                top[:min_rows, :min_cols] |= (L[:min_rows, :min_cols].astype(bool)).astype(int)
        return top
    
    
    def normalize(self) -> 'Structure3D':
        """Remove trailing all-zero layers from the top."""
        if not self.layers:
            return Structure3D([])
        
        # Find the highest non-zero layer
        last_non_zero = -1
        for i in range(len(self.layers) - 1, -1, -1):
            if np.any(self.layers[i] != 0):
                last_non_zero = i
                break
        
        # If all layers are zero, return single zero layer to maintain grid shape
        if last_non_zero == -1:
            return Structure3D([self.layers[0] * 0])  # Keep dimensions but all zeros
        
        # Return structure with only layers up to last non-zero
        return Structure3D([layer.tolist() for layer in self.layers[:last_non_zero + 1]])
    
    def get_hash(self) -> str:
        """Get hash for comparison - normalized to ignore trailing zero layers."""
        import hashlib
        
        # Normalize structure before hashing
        normalized = self.normalize()
        
        if not normalized.layers:
            return "empty000"
        
        # Stack layers into 3D array for stable hashing
        arr = np.stack(normalized.layers, axis=0).astype(np.uint8)
        h = hashlib.md5()
        h.update(arr.tobytes())
        h.update(np.array(arr.shape, dtype=np.int64).tobytes())
        return h.hexdigest()[:8]


class Benchmark3D:
    """3D structure discovery benchmark with sampling."""
    
    def __init__(self, dataset_path: str):
        """Load complete dataset from file."""
        with open(dataset_path, 'r') as f:
            self.complete_dataset = json.load(f)
        
        # Load metadata and observation sets
        self.metadata = self.complete_dataset.get('metadata', {})
        self.all_observation_sets = self.complete_dataset.get('observation_sets', [])
        
        print(f"Loaded dataset with {len(self.all_observation_sets)} observation sets")
    
    def sample_observation_sets(self, n_samples: int, observation_type: str = "top", 
                                seed: Optional[int] = None) -> List[Dict]:
        """Sample n observation sets from the complete dataset.
        
        Args:
            n_samples: Number of observation sets to sample
            observation_type: Type of observations to sample (only "top" is supported)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Since we only have top view observations now, use all available sets
        filtered_sets = self.all_observation_sets
        
        n_available = len(filtered_sets)
        n_to_sample = min(n_samples, n_available)
        
        if n_to_sample < n_samples:
            print(f"Warning: Requested {n_samples} samples of type '{observation_type}' but only {n_available} available")
        
        if n_available == 0:
            print(f"Error: No observation sets of type '{observation_type}' found in dataset")
            return []
        
        print(f"Sampling {n_to_sample} observation sets of type '{observation_type}' from {n_available} available")
        
        sampled = random.sample(filtered_sets, n_to_sample)
        return sampled
    
    def _parse_observation(self, obs_data):
        """Parse observation data from either format."""
        if isinstance(obs_data, str):
            # Compact format: string of 0s and 1s
            # Determine grid size from string length
            length = len(obs_data)
            grid_size = int(length ** 0.5)
            observation = []
            for i in range(grid_size):
                row = [int(obs_data[i * grid_size + j]) for j in range(grid_size)]
                observation.append(row)
            return np.array(observation)
        elif isinstance(obs_data, list):
            # Old format: list of lists
            return np.array(obs_data)
        else:
            # Dictionary format from old observation sets
            return np.array(obs_data.get('observation', []))
    
    def create_prompt(self, observations, prior_structures=None) -> str:
        """Create prompt for LLM from observations.
        
        Args:
            observations: The observation data
            prior_structures: List of previously generated structures to avoid repetition
        """
        # Determine grid size from observations
        if isinstance(observations, str):
            grid_size = int(len(observations) ** 0.5)
        elif isinstance(observations, list) and observations:
            if isinstance(observations[0], dict):
                obs_data = observations[0].get('observation', [])
                grid_size = len(obs_data) if obs_data else 3
            else:
                obs_data = self._parse_observation(observations[0])
                grid_size = obs_data.shape[0] if hasattr(obs_data, 'shape') else 3
        else:
            obs_data = self._parse_observation(observations)
            grid_size = obs_data.shape[0] if hasattr(obs_data, 'shape') else 3
        
        # Get metadata for max height if available
        max_height = self.metadata.get('max_height', 3)
        
        prompt = f"""You are given observations of a 3D structure made of unit blocks on a {grid_size}x{grid_size} grid.
        Each observation shows a view of the structure from a specific angle.
        The maximum height of the structure is {max_height} layers.

        Observations (Top View - shows 1 if ANY layer has a block at that position):
        """
        # Handle both formats
        if isinstance(observations, str):
            # Compact format: single string
            observation = self._parse_observation(observations)
            prompt += f"\nTop view:\n"
            for row in observation:
                prompt += " ".join(str(cell) for cell in row) + "\n"
        elif isinstance(observations, list):
            # Old format: list of observation dicts
            for i, obs in enumerate(observations):
                if isinstance(obs, dict):
                    observation = np.array(obs['observation'])
                else:
                    observation = self._parse_observation(obs)
                prompt += f"\nTop view:\n"
                for row in observation:
                    prompt += " ".join(str(cell) for cell in row) + "\n"
        else:
            # Single observation dict
            observation = self._parse_observation(observations)
            prompt += f"\nTop view:\n"
            for row in observation:
                prompt += " ".join(str(cell) for cell in row) + "\n"
        
        # Add history of prior structures if provided
        if prior_structures and len(prior_structures) > 0:
            prompt += "\n\nPrior 3D structure generated (do not repeat if avoidable):\n"
            for idx, struct in enumerate(prior_structures, 1):
                prompt += f"\nAttempt {idx}:\n"
                prompt += struct.to_string() + "\n"
        
        prompt += f"""
        
        Task: Infer the complete 3D structure that could produce these observations.

        Structure specifications:
        - Grid size: {grid_size}x{grid_size}
        - Maximum height: {max_height} layers
        - The structure consists of layers stacked from bottom to top, where each layer is a {grid_size}x{grid_size} grid with 0 (empty) or 1 (block).
        
        Important constraints:
        1. Layer 1 is the BOTTOM layer (ground level) - it should contain blocks, not be all zeros
        2. Blocks must be supported from below (a block at height h requires a block at height h-1 in the same position)
        3. Do not add unnecessary empty layers at the bottom or top
        4. Layers are numbered from bottom (Layer 1) to top (Layer N)
        5. Do not exceed the maximum height of {max_height} layers
        """
        
        if prior_structures:
            prompt += f"""\n        6. Generate a DIFFERENT valid structure from the {len(prior_structures)} prior attempts shown above
        7. Two structures are considered equivalent if they have the same blocks in the same positions across all layers
        """
        
        prompt += f"""

        Provide your answer as a 3D structure with each layer specified.

        Output format:
        Structure:
        Layer 1:  (bottom layer - should contain at least one block)
        [row 1 values separated by spaces]
        [row 2 values separated by spaces]
        ...
        Layer 2:
        [row 1 values separated by spaces]
        [row 2 values separated by spaces]
        ...
        (continue for all layers needed, up to maximum {max_height} layers)
        
        Example of CORRECT output format for a 3x3 grid:
        Structure:
        Layer 1:
        1 0 1
        0 0 0
        1 0 1
        Layer 2:
        1 0 0
        0 0 0
        0 0 1
        
        Note: This represents a 3x3x2 structure where:
        - Layer 1 (bottom) has blocks at corners: (0,0), (0,2), (2,0), (2,2)
        - Layer 2 (top) has blocks at (0,0) and (2,2), both supported by blocks in Layer 1 below
        - Every '1' in Layer 2 has a '1' directly below it in Layer 1 (physical support requirement)
        - Uses spaces between digits (not commas or other separators)
        
        INCORRECT formats to avoid:
        1. Don't start with empty bottom layer: Layer 1 should contain blocks
        2. Don't use commas: "1,0,1" - use spaces instead: "1 0 1"
        3. Don't add blocks without support below them (physics violation)
        4. Don't exceed {max_height} layers in height
        """
        return prompt
    
    def validate_structure_matches_observations(self, structure: Structure3D, observations) -> bool:
        """Check if a structure matches the given observations."""
        # Handle both formats
        if isinstance(observations, str):
            # Compact format
            observed = self._parse_observation(observations)
            generated = structure.get_top_view()
            if generated.shape != observed.shape:
                return False
            if not np.array_equal(generated, observed):
                return False
        elif isinstance(observations, list):
            # Old format: list of observations
            for obs in observations:
                if isinstance(obs, dict):
                    view_type = obs.get('view_type', 'top')
                    observed = np.array(obs['observation'])
                else:
                    observed = self._parse_observation(obs)
                    view_type = 'top'
                
                if view_type == 'top':
                    generated = structure.get_top_view()
                else:
                    return False
                
                # Check if views match
                if generated.shape != observed.shape:
                    return False
                if not np.array_equal(generated, observed):
                    return False
        else:
            # Single observation
            observed = self._parse_observation(observations)
            generated = structure.get_top_view()
            if generated.shape != observed.shape:
                return False
            if not np.array_equal(generated, observed):
                return False
        
        # Also validate physical support (blocks must have support from below)
        for z in range(1, len(structure.layers)):
            current = structure.layers[z]
            below = structure.layers[z-1]
            # Every block in current layer must have at least one block below it
            for r in range(current.shape[0]):
                for c in range(current.shape[1]):
                    if current[r, c] == 1 and below[r, c] != 1:
                        return False
        
        return True
    
    def _classify_error(self, error_message: str) -> str:
        """Classify the type of error from the error message."""
        if "Expecting value" in error_message:
            match = re.search(r'line (\d+) column (\d+)', error_message)
            if match:
                return f"json_parse_error (line {match.group(1)}, col {match.group(2)})"
            return "json_parse_error"
        elif "Rate limit" in error_message.lower() or "rate_limit" in error_message.lower():
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
            match = re.search(r'\b(\d{3})\b', error_message)
            if match:
                return f"http_error_{match.group(1)}"
            return "unknown_error"
    
    def parse_llm_response(self, response: str) -> Optional[Structure3D]:
        """Parse LLM response to extract 3D structure, handling reasoning text."""
        if not isinstance(response, str):
            return None
        
        try:
            lines = response.strip().split('\n')
            
            # Find where structure specification starts
            # Look for "Structure:" as the main indicator
            structure_start = -1
            for i, line in enumerate(lines):
                if 'Structure:' in line:
                    structure_start = i
                    break
            
            # Fallback: look for "Layer 1:" if no "Structure:" found
            if structure_start == -1:
                for i, line in enumerate(lines):
                    # More strict check - must start with "Layer 1:"
                    if line.strip().startswith('Layer 1:'):
                        structure_start = i
                        break
            
            if structure_start == -1:
                return None
            
            # Parse layers
            layers = []
            current_layer = []
            in_layer = False
            layer_num = 0
            
            for line in lines[structure_start:]:
                line = line.strip()
                
                # Skip empty lines and obvious reasoning/explanation lines
                if not line or line.startswith('Note:') or line.startswith('Reasoning:') or line.startswith('Explanation:'):
                    continue
                
                # Check for layer header (e.g., "Layer 1:", "Layer 2:")
                if line.startswith('Layer '):
                    # Extract layer number to ensure proper ordering
                    match = re.match(r'Layer\s+(\d+)', line)
                    if match:
                        new_layer_num = int(match.group(1))
                        # Only process if it's the expected next layer
                        if new_layer_num == layer_num + 1 or (new_layer_num == 1 and layer_num == 0):
                            if current_layer:
                                layers.append(current_layer)
                                current_layer = []
                            in_layer = True
                            layer_num = new_layer_num
                        else:
                            # Stop if we encounter an out-of-order layer
                            break
                elif in_layer and line:
                    # Stop parsing if we hit reasoning text after structure
                    reasoning_indicators = ['therefore', 'because', 'since', 'thus', 'hence', 
                                          'reasoning:', 'explanation:', 'note:', 'this means',
                                          'this structure', 'the structure', 'i chose', 'i created']
                    if any(indicator in line.lower() for indicator in reasoning_indicators):
                        # We've likely hit reasoning text, stop parsing
                        break
                    
                    # Try to parse row - handle multiple formats
                    try:
                        # Skip lines that are clearly not data rows
                        if len(line) > 100:  # Likely explanation text
                            continue
                        
                        # Handle different separator formats
                        if ',' in line:
                            # Comma-separated: "1,0,1,0,0,0,0,0,0" or "1, 0, 1, 0"
                            parts = [x.strip() for x in line.split(',')]
                        elif '|' in line:
                            # Pipe-separated: "1|0|1|0|0|0|0|0|0"
                            parts = [x.strip() for x in line.split('|')]
                        elif any(c in line for c in ['[', ']']):
                            # List format: "[1 0 1 0 0 0 0 0 0]" or "[1, 0, 1, 0]"
                            line = line.strip('[]')
                            if ',' in line:
                                parts = [x.strip() for x in line.split(',')]
                            else:
                                parts = line.split()
                        else:
                            # Space-separated (original format): "1 0 1 0 0 0 0 0 0"
                            parts = line.split()
                        
                        # Convert to integers - must all be 0 or 1
                        row = []
                        for x in parts:
                            if x.strip():
                                val = int(x)
                                if val not in [0, 1]:
                                    # Not a binary value, skip this line
                                    break
                                row.append(val)
                        
                        if row and len(row) > 0:  # Valid non-empty row
                            current_layer.append(row)
                    except:
                        # If parsing fails, skip this line
                        pass
            
            # Don't forget last layer
            if current_layer:
                layers.append(current_layer)
            
            if layers:
                return Structure3D(layers)
            
        except Exception as e:
            print(f"Error parsing response: {e}")
        
        return None
    
    def evaluate_single_observation_set(
        self,
        llm: LLMInterface,
        observation_set: Dict,
        n_queries: int = 10,
        verbose: bool = True,
        max_retries: int = 3
    ) -> Dict:
        """Evaluate LLM on a single observation set with enhanced tracking."""
        
        # Get observations and ground truth structures
        observations = observation_set.get('observation', observation_set.get('observations', []))
        ground_truth_structures = observation_set.get('ground_truth_structures', [])
        
        # Get ground truth hashes for checking recovery
        gt_hashes = set()
        for gt in ground_truth_structures:
            # Parse layers from compact string format
            if 'layers' in gt and isinstance(gt['layers'][0], str):
                # Compact string format
                layers = []
                grid_size = int(len(gt['layers'][0]) ** 0.5)
                for layer_str in gt['layers']:
                    layer = []
                    for i in range(grid_size):
                        row = [int(layer_str[i * grid_size + j]) for j in range(grid_size)]
                        layer.append(row)
                    layers.append(layer)
                struct = Structure3D(layers)
                # Normalize ground truth structure (get_hash already normalizes internally)
                gt_hashes.add(struct.get_hash())
            else:
                # Old nested list format
                struct = Structure3D(gt['layers'])
                # Normalize ground truth structure (get_hash already normalizes internally)
                gt_hashes.add(struct.get_hash())
        
        # Track results
        all_hypotheses = []
        valid_hypotheses = []
        unique_hashes = set()
        unique_structures = []
        parse_success_count = 0
        prior_structures = []  # Track structures to pass as history
        
        # Token and cost tracking
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Error tracking
        errors = []
        error_counts = {}
        
        for i in range(n_queries):
            # Pass prior unique structures as history to avoid repetition
            prompt = self.create_prompt(observations, prior_structures=unique_structures)
            
            # Try to get a valid response with retries
            structure = None
            query_error = None
            
            for attempt in range(max_retries):
                try:
                    # Use query_with_usage if available
                    if hasattr(llm, 'query_with_usage'):
                        result = llm.query_with_usage(prompt)
                        response = result['response']
                        
                        # Track usage
                        usage = result.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0)
                        total_completion_tokens += usage.get('completion_tokens', 0)
                        total_tokens += usage.get('total_tokens', 0)
                        total_cost += result.get('cost', 0.0)
                    else:
                        response = llm.query(prompt)
                    
                    # Check if response is an error
                    if response and response.startswith("Error querying"):
                        query_error = {
                            'query_index': i,
                            'attempt': attempt + 1,
                            'error_message': response,
                            'error_type': self._classify_error(response)
                        }
                        error_type = query_error['error_type']
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                        continue
                    
                    # Parse response
                    structure = self.parse_llm_response(response)
                    if structure:
                        # Normalize the structure to remove trailing zero layers
                        structure = structure.normalize()
                        parse_success_count += 1
                        break
                
                except Exception as e:
                    query_error = {
                        'query_index': i,
                        'attempt': attempt + 1,
                        'error_message': str(e),
                        'error_type': self._classify_error(str(e))
                    }
                    if verbose:
                        print(f"  ⚠ Exception on query {i + 1}: {str(e)[:100]}")
            
            # Record error if all attempts failed
            if not structure and query_error:
                errors.append(query_error)
            
            if structure:
                all_hypotheses.append(structure)
                
                # Check uniqueness for ALL structures (not just valid ones)
                s_hash = structure.get_hash()
                if s_hash not in unique_hashes:
                    unique_hashes.add(s_hash)
                    unique_structures.append(structure)
                    # Update prior_structures for next query (limit to last 10 for context efficiency)
                    prior_structures = unique_structures[-10:] if len(unique_structures) > 10 else unique_structures.copy()
                
                # Check if it's valid (matches observations)
                if self.validate_structure_matches_observations(structure, observations):
                    valid_hypotheses.append(structure)
        
        # Calculate metrics
        parse_success_rate = parse_success_count / n_queries if n_queries > 0 else 0
        valid_rate = len(valid_hypotheses) / n_queries if n_queries > 0 else 0
        novelty_rate = len(unique_structures) / n_queries if n_queries > 0 else 0
        
        # Check recovery (among valid structures)
        recovered_gts = set()
        for struct in valid_hypotheses:
            s_hash = struct.get_hash()
            if s_hash in gt_hashes:
                recovered_gts.add(s_hash)
        
        recovery_rate = len(recovered_gts) / len(gt_hashes) if gt_hashes else 0
        
        # Get observation set ID
        obs_id = observation_set.get('observation_id', observation_set.get('observation_set_id', 'unknown'))
        n_obs = 1 if isinstance(observations, str) else len(observations)
        
        return {
            'observation_set_id': obs_id,
            'n_observations': n_obs,
            'n_ground_truths': len(ground_truth_structures),
            'n_queries': n_queries,
            'n_valid': len(valid_hypotheses),
            'n_unique': len(unique_structures),
            'n_recovered_gts': len(recovered_gts),
            'parse_success_count': parse_success_count,
            'parse_success_rate': parse_success_rate,
            'valid_rate': valid_rate,
            'novelty_rate': novelty_rate,
            'recovery_rate': recovery_rate,
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'cost': total_cost,
            'errors': errors,
            'error_summary': {
                'total_errors': len(errors),
                'error_types': error_counts
            },
            'all_hypotheses': [h.to_string() for h in all_hypotheses],
            'unique_structures': [s.to_string() for s in unique_structures]
        }
    
    def run_benchmark(
        self,
        llm: LLMInterface,
        n_samples: int = 10,
        n_queries_per_sample: Optional[int] = None,
        query_multiplier: float = 2.0,
        observation_type: str = "top",
        seed: Optional[int] = None,
        verbose: bool = True,
        checkpoint_dir: str = "checkpoints",
        run_id: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict:
        """Run enhanced benchmark with comprehensive tracking.
        
        Args:
            llm: LLM interface to use
            n_samples: Number of observation sets to sample
            n_queries_per_sample: Fixed number of queries per observation set (if None, uses query_multiplier)
            query_multiplier: Multiplier for n_gt to determine queries (default 2.0)
            observation_type: Type of observations to use
            seed: Random seed
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints
            run_id: Run identifier for checkpoint
            max_retries: Maximum retries per query
        
        Returns:
            Dictionary with complete benchmark results including statistics, token usage, and costs
        """
        
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID and checkpoint file
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_llm_name = llm.get_name().replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        checkpoint_file = checkpoint_path / f"checkpoint_3d_{safe_llm_name}_{run_id}.json"
        
        print(f"\nRunning Enhanced 3D Structure Benchmark")
        print(f"LLM: {llm.get_name()}")
        print(f"Sampling {n_samples} observation sets")
        if n_queries_per_sample is not None:
            print(f"Queries per sample: {n_queries_per_sample} (fixed)")
        else:
            print(f"Queries per sample: {query_multiplier}x number of ground truths (adaptive)")
        print(f"Max retries: {max_retries}")
        print(f"Checkpoint file: {checkpoint_file}")
        print("-" * 50)
        
        # Sample observation sets
        sampled_sets = self.sample_observation_sets(n_samples, observation_type, seed)
        
        # Initialize results tracking
        all_results = []
        valid_rates = []
        novelty_rates = []
        recovery_rates = []
        parse_success_rates = []
        
        # Token and cost tracking
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Error tracking
        all_errors = []
        total_error_counts = {}
        
        # Load existing checkpoint if it exists
        start_idx = 0
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    all_results = checkpoint_data.get('results', [])
                    start_idx = len(all_results)
                    
                    # Restore token/cost data
                    if 'total_token_usage' in checkpoint_data:
                        total_prompt_tokens = checkpoint_data['total_token_usage'].get('prompt_tokens', 0)
                        total_completion_tokens = checkpoint_data['total_token_usage'].get('completion_tokens', 0)
                        total_tokens = checkpoint_data['total_token_usage'].get('total_tokens', 0)
                    if 'total_cost' in checkpoint_data:
                        total_cost = checkpoint_data['total_cost']
                    
                    # Restore error data
                    all_errors = checkpoint_data.get('all_errors', [])
                    total_error_counts = checkpoint_data.get('total_error_counts', {})
                    
                    print(f"Resuming from checkpoint: {start_idx}/{n_samples} completed")
                    
                    # Recalculate rates from checkpoint
                    for result in all_results:
                        valid_rates.append(result['valid_rate'])
                        novelty_rates.append(result['novelty_rate'])
                        recovery_rates.append(result['recovery_rate'])
                        parse_success_rates.append(result.get('parse_success_rate', 1.0))
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
                print("Starting from beginning...")
        
        # Process each sampled observation set
        for idx in range(start_idx, len(sampled_sets)):
            obs_set = sampled_sets[idx]
            
            if verbose:
                print(f"\nSample {idx + 1}/{n_samples}")
                # Get observation set info
                obs_id = obs_set.get('observation_id', obs_set.get('observation_set_id', 'unknown'))
                n_obs = 1 if 'observation' in obs_set and isinstance(obs_set['observation'], str) else obs_set.get('n_observations', 1)
                n_gts = obs_set.get('n_compatible_structures', 0)
                
                print(f"  ID: {obs_id}")
                print(f"  Observations: {n_obs}")
                print(f"  Compatible structures: {n_gts}")
            
            try:
                # Determine number of queries
                if n_queries_per_sample is not None:
                    n_queries = n_queries_per_sample
                else:
                    n_gt = obs_set.get('n_compatible_structures', 1)
                    n_queries = max(1, int(n_gt * query_multiplier))
                    if verbose:
                        print(f"  Using {n_queries} queries ({query_multiplier}x {n_gt} ground truths)")
                
                # Evaluate
                result = self.evaluate_single_observation_set(
                    llm, obs_set, n_queries, verbose=False, max_retries=max_retries
                )
                
                all_results.append(result)
            
                valid_rates.append(result['valid_rate'])
                novelty_rates.append(result['novelty_rate'])
                recovery_rates.append(result['recovery_rate'])
                parse_success_rates.append(result.get('parse_success_rate', 1.0))
                
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
                    print(f"  Parse success rate: {result.get('parse_success_rate', 1.0):.2%}")
                    print(f"  Valid rate: {result['valid_rate']:.2%}")
                    print(f"  Novelty rate: {result['novelty_rate']:.2%}")
                    print(f"  Recovery rate: {result['recovery_rate']:.2%}")
                    if result.get('cost', 0) > 0:
                        print(f"  Cost: ${result['cost']:.6f}")
                
                # Save checkpoint
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
                    'all_errors': all_errors,
                    'total_error_counts': total_error_counts
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
            'metadata': self.metadata,
            'statistics': {
                'parse_success_rate': {
                    **calculate_stats(parse_success_rates),
                    'p_value': calculate_p_value(parse_success_rates)
                },
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
        
        # Print comprehensive summary
        print("\n" + "=" * 60)
        print("ENHANCED 3D BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(f"Samples evaluated: {len(all_results)}/{n_samples}")
        
        for metric_name, metric_key in [('Parse Success Rate', 'parse_success_rate'),
                                        ('Valid Rate', 'valid_rate'), 
                                        ('Novelty Rate', 'novelty_rate'), 
                                        ('Recovery Rate', 'recovery_rate')]:
            stats_dict = final_results['statistics'][metric_key]
            print(f"\n{metric_name}:")
            print(f"  Mean ± Std: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Variance: {stats_dict['var']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            if stats_dict['p_value'] is not None:
                print(f"  p-value: {stats_dict['p_value']:.4f}")
        
        print(f"\nToken Usage:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Prompt tokens: {total_prompt_tokens:,}")
        print(f"  Completion tokens: {total_completion_tokens:,}")
        print(f"  Avg tokens/sample: {final_results['token_usage']['avg_tokens_per_sample']:.1f}")
        print(f"  Avg tokens/query: {final_results['token_usage']['avg_tokens_per_query']:.1f}")
        
        print(f"\nCost:")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Avg cost/sample: ${final_results['cost']['avg_cost_per_sample']:.4f}")
        print(f"  Avg cost/query: ${final_results['cost']['avg_cost_per_query']:.6f}")
        
        if all_errors:
            print(f"\nErrors:")
            print(f"  Total errors: {len(all_errors)}")
            print(f"  Error rate: {final_results['error_summary']['error_rate']:.2%}")
            if total_error_counts:
                print(f"  Error types:")
                for error_type, count in sorted(total_error_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {error_type}: {count}")
        
        print("=" * 60)
        
        return final_results


def setup_llm(llm_type: str, **kwargs) -> LLMInterface:
    """Set up LLM interface based on type."""
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


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run enhanced 3D structure discovery benchmark with comprehensive tracking\n\n"
                    "Features:\n"
                    "- Token usage and cost tracking\n"
                    "- Checkpoint mechanism for resuming\n"
                    "- Enhanced error handling\n"
                    "- Statistical analysis of results\n"
                    "- Adaptive query count based on ground truths",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Path to complete 3D dataset JSON file")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of observation sets to sample")
    parser.add_argument("--n-queries", type=int, default=None, help="Fixed number of queries per observation set (if not set, uses adaptive)")
    parser.add_argument("--query-multiplier", type=float, default=1.0, help="Multiplier for adaptive queries (n_queries = n_gt * multiplier)")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries per query")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--observation-type", type=str, default="top", choices=["top"], help="Type of observation sets to use (only top view supported)")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    
    args = parser.parse_args()
    
    # Handle verbose/quiet flags
    if args.quiet:
        args.verbose = False
    
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
    checkpoint_dir = args.checkpoint_dir or config.get('benchmark', {}).get('checkpoint_dir', 'checkpoints')
    verbose = args.verbose and config.get('benchmark', {}).get('verbose', True)
    run_id = config.get('benchmark', {}).get('run_id', None)
    
    # Generate output filename if not specified
    if args.output is None:
        dataset_name = Path(args.dataset).stem
        model_name = Path(model).stem if model else llm_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/{dataset_name}_{model_name}_{timestamp}.json"
    
    # Initialize benchmark
    benchmark = Benchmark3D(args.dataset)
    
    # Print configuration
    print("\n" + "=" * 60)
    print("ENHANCED 3D STRUCTURE DISCOVERY BENCHMARK")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"LLM Type: {llm_type}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Samples: {args.n_samples}")
    
    if args.n_queries is not None:
        print(f"Queries per sample: {args.n_queries} (fixed)")
    else:
        print(f"Queries per sample: {args.query_multiplier}x ground truths (adaptive)")
    
    print(f"Max retries: {args.max_retries}")
    print(f"Observation type: {args.observation_type}")
    print(f"Seed: {args.seed}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Output: {args.output}")
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
        observation_type=args.observation_type,
        seed=args.seed,
        verbose=verbose,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id,
        max_retries=args.max_retries
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal results saved to: {args.output}")


if __name__ == "__main__":
    main()