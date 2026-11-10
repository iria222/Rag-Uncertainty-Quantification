
import torch

def compute_mean_token_entropy(logits: tuple) -> float | None:
    """Compute mean token entropy over the generated sequence

    Args:
        logits (tuple): Tuple containing the vocabulary logits for each generation step

    Returns:
        float | None: Mean token entropy. None if no tokens were generated
    """
    if not logits:
        return None
    
    entropies = []
    
    for step_logits in logits:
        log_probs = torch.log_softmax(step_logits.float(), dim=-1)
        
        probs = log_probs.exp()
        
        entropy = -(probs * log_probs).sum(dim=-1)
        entropies.append(entropy)
        
    mean_entropy = torch.stack(entropies).mean()
    
    return mean_entropy.item()