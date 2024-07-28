import json
import random
import datasets
from tqdm import tqdm


def run_alpaca_eval(moa_system, alpaca_moa_path, alpaca_reference_path, num_examples=None):
    # Load AlpacaEval 2.0 dataset
    eval_set = datasets.load_dataset(
        "tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline", trust_remote_code=True
    )["eval"]
    
    # Select random subset
    subset = random.sample(list(eval_set), num_examples) if num_examples else eval_set
    
    alpaca_moa_results = []
    alpaca_reference_results = []
    
    for item in tqdm(subset):
        output = moa_system.run(user_text=item["instruction"])
        
        alpaca_moa_results.append({
            "instruction": item["instruction"],
            "output": output,
            "generator": "MoA-System"
        })
        
        alpaca_reference_results.append({
            "instruction": item["instruction"],
            "output": item["output"],
            "generator": item["generator"]
        })
        
    with open(alpaca_moa_path, "w") as f:
        json.dump(alpaca_moa_results, f, indent=2)
        
    with open(alpaca_reference_path, "w") as f:
        json.dump(alpaca_reference_results, f, indent=2)
