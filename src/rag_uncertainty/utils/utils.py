import gc

import torch

from src.rag_uncertainty.config import DEBUG_MODE, DEVICE, ALLOWED_QUANTIZATIONS


def free_model(model):
    """Free up memory used by the model.
    
    Args:
        model: The model to be freed.
    """
    if model is not None:
        if hasattr(model, "to"):
            try:
                model.to("cpu")
            except Exception:
                pass
        del model
        
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    debug_print("Memory freed.")
    
def debug_print(message:str):
    """Prints messages only if debug mode is activate

    Args:
        message (str): debug message
    """
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")
        
def validate_quantization(quantization: str | None) -> str |None:
    if quantization not in ALLOWED_QUANTIZATIONS:
        allowed = ", ".join(sorted(map(repr, ALLOWED_QUANTIZATIONS)))
        raise ValueError(
            f"Quantization value not valid: {quantization}."
            f"The allowed values are: {allowed}"
        )
    return quantization


