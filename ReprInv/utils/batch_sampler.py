import torch
from torch.utils.data.sampler import BatchSampler
import math

class PureBatchSampler(BatchSampler):
    def __init__(self, sampler, dataset, batch_size, conditions):
        super().__init__(sampler, batch_size, False)
        self.dataset = dataset
        self.conditions = conditions
        
    def _get_indices_attr(self, condition):
        norm_cond = condition.lower()
        if norm_cond == "adek":
            return "ade20k_indices"
        return f"{norm_cond}_indices"

    def __iter__(self):
        condition_indices = {}
        for condition in self.conditions:
            attr_name = self._get_indices_attr(condition)
            if not hasattr(self.dataset, attr_name):
                 raise AttributeError(f"Dataset missing attribute {attr_name} for condition {condition}")
            
            indices = getattr(self.dataset, attr_name)
            shuffled = indices[torch.randperm(len(indices))]
            condition_indices[condition] = shuffled

        condition_batches = {c: [] for c in self.conditions}
        
        for condition in self.conditions:
            indices = condition_indices[condition]
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size:
                    condition_batches[condition].append(batch.tolist())
        
        max_len = max([len(b) for b in condition_batches.values()]) if condition_batches else 0
        
        for i in range(max_len):
            for condition in self.conditions:
                batches = condition_batches[condition]
                if i < len(batches):
                    yield batches[i]

    def __len__(self):
        total_batches = 0
        for condition in self.conditions:
            attr_name = self._get_indices_attr(condition)
            if hasattr(self.dataset, attr_name):
                indices = getattr(self.dataset, attr_name)
                total_batches += len(indices) // self.batch_size
        return total_batches
