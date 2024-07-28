from benchmark_runners import run_alpaca_eval
from moa_system_pass_all_layers import MoASystem


def main():
    moa_system = MoASystem()
    moa_outputs_path = "moa_outputs.json"
    reference_outputs_path = "reference_outputs.json"
    
    run_alpaca_eval(moa_system, moa_outputs_path, reference_outputs_path, num_examples=50)
    
    # alpaca_eval --model_outputs moa_outputs.json --reference_outputs reference_outputs.json --output_path leaderboard


if __name__ == '__main__':
    main()
