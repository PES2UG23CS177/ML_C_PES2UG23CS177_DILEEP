# lab.py
import torch
import math

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    """
    target_col = tensor[:, -1]  # last column is target
    values, counts = torch.unique(target_col, return_counts=True)
    probabilities = counts.float() / counts.sum()
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return float(entropy)


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.
    """
    total_rows = tensor.shape[0]
    attribute_col = tensor[:, attribute]
    values, counts = torch.unique(attribute_col, return_counts=True)

    avg_info = 0.0
    for v, cnt in zip(values, counts):
        subset = tensor[attribute_col == v]
        subset_entropy = get_entropy_of_dataset(subset)
        weight = cnt.item() / total_rows
        avg_info += weight * subset_entropy

    return float(avg_info)


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)
    """
    total_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    info_gain = total_entropy - avg_info
    return round(float(info_gain), 4)


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.

    Returns a tuple with:
    1. Dictionary mapping attribute indices to their information gains
    2. Index of the attribute with highest information gain
    
    Example: ({0: 0.123, 1: 0.768, 2: 1.23}, 2)
    """
    num_attributes = tensor.shape[1] - 1  # exclude target column
    info_gains = {}

    for attr in range(num_attributes):
        info_gains[attr] = get_information_gain(tensor, attr)

    best_attr = max(info_gains, key=info_gains.get)
    return info_gains, best_attr