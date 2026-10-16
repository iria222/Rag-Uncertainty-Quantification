import gc

import torch

from config import DEBUG_MODE, DEVICE


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