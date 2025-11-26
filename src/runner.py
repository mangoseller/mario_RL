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


def select_curriculum() -> int:
    """Prompt user to select a curriculum option."""
    from curriculum import CURRICULUM_OPTIONS, CurriculumState
    
    print("\n" + "="*60)
    print("CURRICULUM OPTIONS")
    print("="*60)
    
    for option_num, schedule in CURRICULUM_OPTIONS.items():
        print(f"\nCurriculum {option_num}:")
        temp_state = CurriculumState(schedule=schedule)
        for phase_idx in range(len(schedule)):
            end_pct = int(schedule[phase_idx][0] * 100)
            start_pct = int(schedule[phase_idx - 1][0] * 100) if phase_idx > 0 else 0
            print(f"  {start_pct}-{end_pct}%: {temp_state.get_description(phase_idx).replace(f'Phase {phase_idx}: ', '')}")
        
        # Show trained levels
        trained = temp_state.get_all_trained_levels()
        print(f"  Trained levels: {', '.join(sorted(trained))}")
    
    print("\n" + "="*60)
    
    while True:
        choice = input(f"Select curriculum ({', '.join(map(str, CURRICULUM_OPTIONS.keys()))}): ").strip()
        try:
            choice_int = int(choice)
            if choice_int in CURRICULUM_OPTIONS:
                return choice_int
            else:
                print(f"Invalid choice. Please select from: {list(CURRICULUM_OPTIONS.keys())}")
        except ValueError:
            print(f"Invalid input. Please enter a number: {list(CURRICULUM_OPTIONS.keys())}")


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
    parser.add_argument('--curriculum_option', type=int, default=None,
                       help='Curriculum option (1 or 2). If not provided, will prompt.')
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
    
    # Handle curriculum selection
    curriculum_option = None
    if effective_mode == 'curriculum':
        if args.curriculum_option is not None:
            from curriculum import CURRICULUM_OPTIONS
            if args.curriculum_option in CURRICULUM_OPTIONS:
                curriculum_option = args.curriculum_option
            else:
                print(f"Invalid curriculum option: {args.curriculum_option}")
                print(f"Valid options: {list(CURRICULUM_OPTIONS.keys())}")
                return
        else:
            curriculum_option = select_curriculum()
    
    config_map = {
        # ImpalaLike mappings
        (ImpalaLike, 'train'): IMPALA_TRAIN_CONFIG,
        (ImpalaLike, 'test'): IMPALA_TEST_CONFIG,
        (ImpalaLike, 'finetune'): IMPALA_TUNE_CONFIG,
        (ImpalaLike, 'resume'): IMPALA_TRAIN_CONFIG,
        (ImpalaLike, 'curriculum'): IMPALA_TRAIN_CONFIG,
        
        # TransPala mappings
        (TransPala, 'train'): TRANSPALA_TRAIN_CONFIG,
        (TransPala, 'test'): TRANSPALA_TEST_CONFIG,
        (TransPala, 'finetune'): TRANSPALA_TUNE_CONFIG,
        (TransPala, 'resume'): TRANSPALA_TRAIN_CONFIG,
        (TransPala, 'curriculum'): TRANSPALA_TRAIN_CONFIG,
        
        # ConvolutionalSmall mappings
        (ConvolutionalSmall, 'train'): CONV_TRAIN_CONFIG,
        (ConvolutionalSmall, 'test'): CONV_TEST_CONFIG,
        (ConvolutionalSmall, 'finetune'): CONV_TUNE_CONFIG,
        (ConvolutionalSmall, 'resume'): CONV_TRAIN_CONFIG,
        (ConvolutionalSmall, 'curriculum'): CONV_TRAIN_CONFIG,
    }
    
    config = config_map[(model, effective_mode)]
    
    # Enable curriculum if requested
    if effective_mode == 'curriculum' and not getattr(config, 'use_curriculum', False):
        from dataclasses import replace
        config = replace(config, use_curriculum=True)

    from train import train, finetune, resume
    
    if effective_mode == 'finetune':
        finetune(model, args.checkpoint, config, args.num_eval_episodes)
    elif effective_mode == 'resume':
        resume(model, args.checkpoint, config, args.num_eval_episodes)
    else:
        # train and curriculum modes both use train()
        train(model, config, args.num_eval_episodes, curriculum_option=curriculum_option)
