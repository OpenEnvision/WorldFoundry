"""
KV Cache Manager for Autoregressive Video Generation

Key insight: In AR generation, historical frames have fixed input (clean + small noise),
so their K and V in temporal self-attention remain constant across all diffusion steps.

Cache structure: cache[layer_name][frame_idx] = {'k': tensor, 'v': tensor}
"""

import torch
from typing import Dict, Optional, Tuple


class KVCacheManager:
    """
    Manages KV cache for temporal self-attention in autoregressive generation.
    
    During AR generation of frame t:
    - Historical frames [0:t-1]: Use cached K,V (constant input)
    - Current frame t: Compute K,V (varying input across diffusion steps)
    - At last diffusion step: Store current frame's K,V for future use
    """
    
    def __init__(self, verbose: bool = False):
        """Init.

        Args:
            verbose: The verbose.
        """
        self.cache: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
        self.verbose = verbose
        self.stats = {'hits': 0, 'misses': 0, 'stores': 0}
        
    def store(self, layer_name: str, frame_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Store K,V for a frame at a specific layer."""
        if layer_name not in self.cache:
            self.cache[layer_name] = {}
        
        self.cache[layer_name][frame_idx] = {
            'k': k.detach().clone(),
            'v': v.detach().clone()
        }
        self.stats['stores'] += 1
    
    def retrieve(self, layer_name: str, frame_range: range) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve cached K,V for a range of frames."""
        if layer_name not in self.cache:
            self.stats['misses'] += 1
            return None, None
        
        k_list, v_list = [], []
        for frame_idx in frame_range:
            if frame_idx not in self.cache[layer_name]:
                self.stats['misses'] += 1
                return None, None
            k_list.append(self.cache[layer_name][frame_idx]['k'])
            v_list.append(self.cache[layer_name][frame_idx]['v'])
        
        if not k_list:
            return None, None
        
        # Concatenate along sequence dimension
        k_cached = torch.cat(k_list, dim=1)  # (B*H, T, D)
        v_cached = torch.cat(v_list, dim=1)
        self.stats['hits'] += 1
        return k_cached, v_cached
    
    def clear_all(self):
        """Clear all cache."""
        self.cache.clear()
    
    def print_stats(self):
        """Print cache statistics."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        memory_gb = sum(
            k['k'].element_size() * k['k'].nelement() + k['v'].element_size() * k['v'].nelement()
            for layer in self.cache.values()
            for k in layer.values()
        ) / (1024**3)
        
        print(f"\n{'='*60}")
        print(f"KV Cache Statistics:")
        print(f"  Layers: {len(self.cache)}")
        print(f"  Cache hits: {self.stats['hits']}")
        print(f"  Cache misses: {self.stats['misses']}")
        print(f"  Hit rate: {hit_rate:.2%}")
        print(f"  Stores: {self.stats['stores']}")
        print(f"  Memory: {memory_gb:.3f} GB")
        print(f"{'='*60}\n")
