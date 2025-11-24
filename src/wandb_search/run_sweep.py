import wandb
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import TrainingConfig
from models import ConvolutionalSmall, ImpalaLike
from train import training_loop
from wandb_search.sweep_config import CONV_SWEEP, IMPALA_SWEEP


IMPALA_SWEEP_CONFIG = TrainingConfig(
    architecture='ImpalaLike',
    lr_schedule='cosine',  
    learning_rate=2e-4,   
    min_lr=1e-6,
    epochs=4,          
    clip_eps=0.2,
    c1=0.5,             
    c2=0.01,             
    gamma=0.99,           
    num_envs=8,            
    steps_per_env=512,
    num_training_steps=800_000,  
    checkpoint_freq=1_000_000,   
    eval_freq=100_000,           
    show_progress=True,
    USE_WANDB=True
)
CONV_SWEEP_CONFIG = TrainingConfig(
    architecture='ConvolutionalSmall',
    lr_schedule='cosine',
    learning_rate=2e-5,    
    min_lr=1e-6,
    epochs=8,             
    clip_eps=0.2,
    c1=0.5,                
    c2=0.01,               
    gamma=0.99,           
    num_envs=8,
    steps_per_env=512,
    num_training_steps=800_000,
    checkpoint_freq=1_000_000,
    eval_freq=100_000,
    show_progress=True,
    USE_WANDB=True
)

MODEL_CLASS = None
BASE_CONFIG = None


def train_sweep(config=None):
    with wandb.init(config=config):
        updated_config = TrainingConfig.from_wandb(BASE_CONFIG)
        agent = MODEL_CLASS()
        training_loop(agent, updated_config)


if __name__ == "__main__":
    while True:
        model_choice = input("Select model for sweep (ConvolutionalSmall/ImpalaLike/exit): ").strip()
        if model_choice.lower() == 'exit':
            print("Exiting program.")
            sys.exit(0)
        elif model_choice == 'ConvolutionalSmall':
            sweep_config = CONV_SWEEP
            BASE_CONFIG = CONV_SWEEP_CONFIG
            MODEL_CLASS = ConvolutionalSmall
            break
        elif model_choice == 'ImpalaLike':
            sweep_config = IMPALA_SWEEP
            BASE_CONFIG = IMPALA_SWEEP_CONFIG
            MODEL_CLASS = ImpalaLike
            break
        else:
            print(f"Unrecognized model '{model_choice}'. Please choose ConvolutionalSmall or ImpalaLike.")
    
    sweep_id = wandb.sweep(
        sweep_config,
        project="marioRL"
    )
    print(f"Sweep initialized with ID: {sweep_id} for model: {model_choice}")
    
    # Start the agent to run the search
    wandb.agent(
        sweep_id,
        function=train_sweep,
        count=10
    )
