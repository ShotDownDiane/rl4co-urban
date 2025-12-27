"""
ML4CO-Kit Dataset Reader for RL4CO

This module provides dataset readers for loading ML4CO-Kit test datasets
into RL4CO environments. It supports reading from wrapper txt files.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from torch.utils.data import Dataset


class ML4CODataset(Dataset):
    """
    Base class for loading ML4CO-Kit datasets.
    
    Args:
        file_path: Path to the ML4CO-Kit dataset file (txt or pkl)
        device: Device to load tensors to
        normalize: Whether to normalize coordinates (for TSP/CVRP)
    """
    
    def __init__(
        self,
        file_path: str,
        device: str = "cpu",
        normalize: bool = False
    ):
        self.file_path = Path(file_path)
        self.device = device
        self.normalize = normalize
        self.data = []
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Load data based on file extension
        if self.file_path.suffix == ".txt":
            self._load_from_txt()
        elif self.file_path.suffix == ".pkl":
            self._load_from_pkl()
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
    
    def _load_from_txt(self):
        """Load data from txt file. To be implemented by subclasses."""
        raise NotImplementedError
    
    def _load_from_pkl(self):
        """Load data from pkl file."""
        import pickle
        with open(self.file_path, "rb") as f:
            task_list = pickle.load(f)
        self._process_task_list(task_list)
    
    def _process_task_list(self, task_list):
        """Process task list from pkl file. To be implemented by subclasses."""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class TSPDataset(ML4CODataset):
    """
    Dataset for loading TSP instances from ML4CO-Kit.
    
    Returns:
        dict with keys:
            - 'locs': [num_nodes, 2] coordinates
            - 'tour': [num_nodes] optimal tour (optional)
    """
    
    def _load_from_txt(self):
        """
        Load TSP data from txt file.
        Format: <coords> output <tour>
        Example: 0.1 0.2 0.3 0.4 ... output 1 3 2 4 ...
        """
        with open(self.file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split by " output "
                parts = line.split(" output ")
                if len(parts) != 2:
                    raise ValueError(f"Invalid line format: {line[:100]}...")
                
                # Parse coordinates
                coords_str = parts[0].strip().split()
                coords = np.array([float(x) for x in coords_str], dtype=np.float32)
                coords = coords.reshape(-1, 2)
                
                # Parse tour (1-indexed in file, convert to 0-indexed)
                tour_str = parts[1].strip().split()
                tour = np.array([int(x) - 1 for x in tour_str], dtype=np.int64)
                
                # Convert to tensors
                locs = torch.from_numpy(coords).to(self.device)
                tour_tensor = torch.from_numpy(tour).to(self.device)
                
                self.data.append({
                    'locs': locs,
                    'tour': tour_tensor
                })
    
    def _process_task_list(self, task_list):
        """Process TSP task list from pkl file."""
        for task in task_list:
            locs = torch.from_numpy(task.points).to(self.device)
            
            data_dict = {'locs': locs}
            if task.sol is not None:
                tour = torch.from_numpy(task.sol).to(self.device)
                data_dict['tour'] = tour
            
            self.data.append(data_dict)


class CVRPDataset(ML4CODataset):
    """
    Dataset for loading CVRP instances from ML4CO-Kit.
    
    Returns:
        dict with keys:
            - 'depot': [2] depot coordinates
            - 'locs': [num_nodes, 2] customer coordinates
            - 'demand': [num_nodes] customer demands
            - 'capacity': scalar vehicle capacity
            - 'tour': list of node indices (variable length, not suitable for batching)
    
    Note:
        Tours have variable lengths due to different numbers of routes.
        Use custom collate function for batching or access instances individually.
    """
    
    def _load_from_txt(self):
        """
        Load CVRP data from txt file.
        Format: depots <x> <y> points <coords> demands <demands> capacity <cap> output <tour>
        """
        with open(self.file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse depot
                    parts = line.split("depots ")[1]
                    depot_points = parts.split(" points ")
                    depot_str = depot_points[0].strip().split()
                    depot = np.array([float(depot_str[0]), float(depot_str[1])], dtype=np.float32)
                    
                    # Parse points
                    demands_split = depot_points[1].split(" demands ")
                    points_str = demands_split[0].strip().split()
                    points = np.array([float(x) for x in points_str], dtype=np.float32)
                    points = points.reshape(-1, 2)
                    
                    # Parse demands
                    capacity_split = demands_split[1].split(" capacity ")
                    demands_str = capacity_split[0].strip().split()
                    demands = np.array([float(x) for x in demands_str], dtype=np.float32)
                    
                    # Parse capacity
                    output_split = capacity_split[1].split(" output ")
                    capacity = float(output_split[0].strip())
                    
                    # Parse tour
                    tour_str = output_split[1].strip().split()
                    tour = np.array([int(x) for x in tour_str], dtype=np.int64)
                    
                    # Convert to tensors
                    depot_tensor = torch.from_numpy(depot).to(self.device)
                    locs = torch.from_numpy(points).to(self.device)
                    demand_tensor = torch.from_numpy(demands).to(self.device)
                    capacity_tensor = torch.tensor(capacity, dtype=torch.float32).to(self.device)
                    tour_tensor = torch.from_numpy(tour).to(self.device)
                    
                    self.data.append({
                        'depot': depot_tensor,
                        'locs': locs,
                        'demand': demand_tensor,
                        'capacity': capacity_tensor,
                        'tour': tour_tensor
                    })
                    
                except Exception as e:
                    print(f"Error parsing line: {line[:100]}...")
                    print(f"Error: {e}")
                    continue
    
    def _process_task_list(self, task_list):
        """Process CVRP task list from pkl file."""
        for task in task_list:
            depot = torch.from_numpy(task.depots).to(self.device)
            locs = torch.from_numpy(task.points).to(self.device)
            demand = torch.from_numpy(task.demands).to(self.device)
            capacity = torch.tensor(task.capacity, dtype=torch.float32).to(self.device)
            
            data_dict = {
                'depot': depot,
                'locs': locs,
                'demand': demand,
                'capacity': capacity
            }
            
            if task.sol is not None:
                tour = torch.from_numpy(task.sol).to(self.device)
                data_dict['tour'] = tour
            
            self.data.append(data_dict)


def cvrp_collate_fn(batch):
    """
    Custom collate function for CVRP datasets with variable-length tours.
    
    Args:
        batch: List of dictionaries from CVRPDataset
    
    Returns:
        Dictionary with batched tensors (tours excluded due to variable length)
    """
    # Stack fixed-size tensors
    depots = torch.stack([item['depot'] for item in batch])
    locs = torch.stack([item['locs'] for item in batch])
    demands = torch.stack([item['demand'] for item in batch])
    capacities = torch.stack([item['capacity'] for item in batch])
    
    # Keep tours as list (variable length)
    tours = [item['tour'] for item in batch]
    
    return {
        'depot': depots,
        'locs': locs,
        'demand': demands,
        'capacity': capacities,
        'tours': tours  # Note: plural, as it's a list
    }


def load_ml4co_dataset(
    problem: str,
    file_path: str,
    device: str = "cpu",
    **kwargs
) -> ML4CODataset:
    """
    Load ML4CO-Kit dataset for a specific problem type.
    
    Args:
        problem: Problem type ('tsp', 'cvrp', etc.)
        file_path: Path to the dataset file
        device: Device to load tensors to
        **kwargs: Additional arguments passed to dataset constructor
    
    Returns:
        ML4CODataset instance
    
    Example:
        >>> dataset = load_ml4co_dataset(
        ...     'tsp',
        ...     '/path/to/tsp50_uniform_16ins.txt'
        ... )
        >>> print(f"Loaded {len(dataset)} instances")
        >>> instance = dataset[0]
        >>> print(f"Locations shape: {instance['locs'].shape}")
        >>> print(f"Tour shape: {instance['tour'].shape}")
    """
    problem = problem.lower()
    
    dataset_classes = {
        'tsp': TSPDataset,
        'cvrp': CVRPDataset,
    }
    
    if problem not in dataset_classes:
        raise ValueError(
            f"Unsupported problem type: {problem}. "
            f"Supported types: {list(dataset_classes.keys())}"
        )
    
    return dataset_classes[problem](file_path, device=device, **kwargs)


def evaluate_on_ml4co_dataset(
    env,
    dataset: ML4CODataset,
    policy=None,
    decode_type: str = "greedy",
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Evaluate a policy or solver on an ML4CO-Kit dataset.
    
    Args:
        env: RL4CO environment
        dataset: ML4CO dataset
        policy: Policy to evaluate (if None, use random)
        decode_type: Decoding strategy
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with evaluation results
    """
    from torch.utils.data import DataLoader
    
    results = {
        'costs': [],
        'gaps': [],  # Gap to optimal solution if available
        'num_instances': len(dataset)
    }
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for batch in dataloader:
        # Create initial state from batch
        # This depends on the environment's reset method
        # For now, just store the data
        results['costs'].extend([0.0] * len(batch['locs']))
    
    return results
