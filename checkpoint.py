"""Checkpoint manager"""

import json
import os
import tempfile

def load_checkpoint(output_path: str) -> list:
    """Load previous results if output file already exists

    Args:
        output_path (str): path of the output json file

    Returns:
        list: list with the already processed results
    """
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"Checkpoint file can't be read: {e}")
    return[]

def get_processed_ids(results: list) -> set:
    """Get a set of the IDs of the already processed queries

    Args:
        results (list): current results

    Returns:
        set: IDs already completed
    """
    
    return {item["id"] for item in results if "id" in item}

def save_checkpoint(ouptut_path: str, results: list):
    """Save results

    Args:
        ouput_path (str): ouput path for json file
        results (list): updated results lists to save
    """
    output_dir = os.path.dirname(os.path.abspath(ouptut_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=output_dir, delete=False, encoding="utf-8") as tf:
            json.dump(results, tf, ensure_ascii=False, indent=4)
            temp_name = tf.name
        os.replace(temp_name, ouptut_path)
    except Exception as e:
        print(f"Error when saving checkpoint: {e}")
        if temp_name and os.path.exists(temp_name):
            os.remove(temp_name)