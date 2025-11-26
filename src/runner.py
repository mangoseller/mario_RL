import argparse
from models import ConvolutionalSmall, ImpalaLike, TransPala
from config import (
    IMPALA_TRAIN_CONFIG, 
    IMPALA_TEST_CONFIG, 
    IMPALA_TUNE_CONFIG,
    TRANSPALA_TRAIN_CONFIG,
    TRANSPALA_TEST_CONFIG,
    TRANSPALA_TUNE_CONFIG,
    CONV_TRAIN_CONFIG, 
    CONV_TEST_CONFIG, 
    CONV_TUNE_CONFIG
)


def run_training():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'finetune', 'resume', 'curriculum'], default='test')
    parser.add_argument('--model', type=str, default=None, 
                       help='Model to use: ConvolutionalSmall, ImpalaLike, or TransPala')
    parser.add_argument('--checkpoint', type=str, default='finetune.pt', 
                       help='Checkpoint path for fine-tuning or resuming')
    parser.add_argument('--num_eval_episodes', type=int, default=9,
                       help='Number of episodes to run during evaluation')
    parser.add_argument('--curriculum', action='store_true',
                       help='Enable curriculum learning (alternative to --mode curriculum)')
    args = parser.parse_args()
    
    model_map = {
        'ConvolutionalSmall': ConvolutionalSmall,
        'ImpalaLike': ImpalaLike,
        'TransPala': TransPala,
    }
    
    if args.model is None:
        while True:
            model_choice = input("Select model (ConvolutionalSmall/ImpalaLike/TransPala/exit): ").strip()
            if model_choice.lower() == 'exit':
                print("Exiting program.")
                return
            elif model_choice in model_map:
                model = model_map[model_choice]
                break
            else:
                print(f"Unrecognized model '{model_choice}'. Valid options: {list(model_map.keys())}")
    else:
        if args.model not in model_map:
            print(f"Unrecognized model '{args.model}'. Valid options: {list(model_map.keys())}")
            return
        model = model_map[args.model]
    
    # Determine effective mode (--curriculum flag overrides mode to curriculum)
    effective_mode = 'curriculum' if args.curriculum else args.mode
    
    config_map = {
        # ImpalaLike mappings
        (ImpalaLike, 'train'): IMPALA_TRAIN_CONFIG,
        (ImpalaLike, 'test'): IMPALA_TEST_CONFIG,
        (ImpalaLike, 'finetune'): IMPALA_TUNE_CONFIG,
        (ImpalaLike, 'resume'): IMPALA_TRAIN_CONFIG,
        (ImpalaLike, 'curriculum'): IMPALA_TRAIN_CONFIG, # Uses standard train config
        
        # TransPala mappings
        (TransPala, 'train'): TRANSPALA_TRAIN_CONFIG,
        (TransPala, 'test'): TRANSPALA_TEST_CONFIG,
        (TransPala, 'finetune'): TRANSPALA_TUNE_CONFIG,
        (TransPala, 'resume'): TRANSPALA_TRAIN_CONFIG,
        (TransPala, 'curriculum'): TRANSPALA_TRAIN_CONFIG, # Uses standard train config
        
        # ConvolutionalSmall mappings
        (ConvolutionalSmall, 'train'): CONV_TRAIN_CONFIG,
        (ConvolutionalSmall, 'test'): CONV_TEST_CONFIG,
        (ConvolutionalSmall, 'finetune'): CONV_TUNE_CONFIG,
        (ConvolutionalSmall, 'resume'): CONV_TRAIN_CONFIG,
        (ConvolutionalSmall, 'curriculum'): CONV_TRAIN_CONFIG, # Uses standard train config
    }
    
    config = config_map[(model, effective_mode)]
    
    # If curriculum mode but config doesn't have curriculum enabled, enable it
    if effective_mode == 'curriculum' and not getattr(config, 'use_curriculum', False):
        # Create a copy with curriculum enabled
        from dataclasses import replace
        # Note: Ensure your TrainingConfig dataclass has a 'use_curriculum' field, 
        # or this will raise a TypeError. If it's missing, you may need to add it to config.py
        try:
            config = replace(config, use_curriculum=True)
        except TypeError:
            print("Warning: 'use_curriculum' field not found in TrainingConfig. Proceeding without explicit flag.")

    from train import train, finetune, resume
    
    if effective_mode == 'finetune':
        finetune(model, args.checkpoint, config, args.num_eval_episodes)
    elif effective_mode == 'resume':
        resume(model, args.checkpoint, config, args.num_eval_episodes)
    else:
        # train and curriculum modes both use train()
        train(model, config, args.num_eval_episodes)
