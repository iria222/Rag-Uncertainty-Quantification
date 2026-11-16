
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

def compute_mean_token_entropy(entropy: torch.Tensor, number_of_tokens: int) -> float | None:
    """Compute the mean entropy across all tokens.
    
    Args:
        entropy (torch.Tensor): A tensor containing the entropy values for each token.
        number_of_tokens (int): The total number of tokens.
        
    Returns:
        float | None: The mean entropy value across all tokens, or None if the input is empty.
    """
    if entropy is None or number_of_tokens == 0:
        return None
    return entropy.mean().item()

def compute_max_token_entropy(entropy: torch.Tensor, number_of_tokens: int) -> float | None:
    """Compute the maximum entropy across all tokens.
    
    Args:
        entropy (torch.Tensor): A tensor containing the entropy values for each token.
        number_of_tokens (int): The total number of tokens.
        
    Returns:
        float | None: The maximum entropy value across all tokens, or None if the input is empty.
    """
    if entropy is None or number_of_tokens == 0:
        return None
    
    return entropy.max().item()

def compute_p90_entropy(entropy: torch.Tensor, number_of_tokens: int) -> float | None:
    """Compute the 90th percentile entropy across all tokens.
    
    Args:
        entropy (torch.Tensor): A tensor containing the entropy values for each token.
        number_of_tokens (int): The total number of tokens.
        
    Returns:
        float | None: The 90th percentile entropy value across all tokens, or None if the input is empty.
    """
    if entropy is None or number_of_tokens == 0:
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