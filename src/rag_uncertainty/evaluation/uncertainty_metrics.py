
import torch

def compute_entropy(logits:tuple) -> torch.Tensor | None:
    """Compute the entropy for each token in the logits.
    
    Args:
        logits (tuple): A tuple containing the logits for each token.
        
    Returns:
        torch.Tensor | None: A tensor containing the entropy values for each token, or None if the input is empty.
    """
    if not logits:
            return None
        
    entropies = []
    
    for step_logits in logits:
        log_probs = torch.log_softmax(step_logits.float(), dim=-1)
        
        probs = log_probs.exp()
        
        entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)
        entropies.append(entropy)
        
    return torch.stack(entropies)

def compute_mean_token_entropy(entropy: torch.Tensor) -> float | None:
    """Compute the mean entropy across all tokens.
    
    Args:
        entropy (torch.Tensor): A tensor containing the entropy values for each token.
        
    Returns:
        float | None: The mean entropy value across all tokens, or None if the input is empty.
    """
    if entropy is None or get_number_of_tokens(entropy) == 0:
        return None
    return entropy.mean().item()

def compute_max_token_entropy(entropy: torch.Tensor) -> float | None:
    """Compute the maximum entropy across all tokens.
    
    Args:
        entropy (torch.Tensor): A tensor containing the entropy values for each token.
        
    Returns:
        float | None: The maximum entropy value across all tokens, or None if the input is empty.
    """
    if entropy is None or get_number_of_tokens(entropy) == 0:
        return None
    
    return entropy.max().item()

def compute_p90_entropy(entropy: torch.Tensor) -> float | None:
    """Compute the 90th percentile entropy across all tokens.
    
    Args:
        entropy (torch.Tensor): A tensor containing the entropy values for each token.
        
    Returns:
        float | None: The 90th percentile entropy value across all tokens, or None if the input is empty.
    """
    if entropy is None or get_number_of_tokens(entropy) == 0:
        return None
    p90_entropy = torch.quantile(entropy, 0.9)
    
    return p90_entropy.item()

def get_number_of_tokens(entropy: torch.Tensor) -> int | None:
    """Get the number of tokens based on the entropy tensor.
    
    Args:
        entropy (torch.Tensor): A tensor containing the entropy values for each token.
        
    Returns:
        int | None: The number of tokens, or None if the input is empty.
    """
    if entropy is None:
        return 0
    return entropy.numel()


UNCERTAINTY_METRICS = {
    "mean_token_entropy": compute_mean_token_entropy,
    "max_token_entropy": compute_max_token_entropy,
    "p90_entropy": compute_p90_entropy,
    "num_tokens": get_number_of_tokens
}

def compute_uncertainty_metrics(outputs, metrics : dict = UNCERTAINTY_METRICS):
    """Compute uncertainty metrics based on the model outputs and specified metrics.
    
    Args:
        outputs: The model outputs containing logits for each token.
        metrics (dict, optional): A dictionary of metric functions to compute. Defaults to UNCERTAINTY_METRICS.
        
    Returns:
        dict: A dictionary containing the computed uncertainty metrics.
    """
    entropy = compute_entropy(outputs.logits)
    result = {name: fn(entropy) for name, fn in metrics.items()}
    
    return result