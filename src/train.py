import warnings

warnings.filterwarnings('ignore', message=".*Overwriting existing videos.*", category=UserWarning)
warnings.filterwarnings('ignore', message="Can't initialize NVML", category=UserWarning, module="torch.cuda")
warnings.filterwarnings('ignore', message="Conversion of an array with ndim > 0 to a scalar is deprecated", 
                       category=DeprecationWarning, module="torchrl.envs.libs.gym")
warnings.filterwarnings('ignore', message=".*Overwriting existing videos.*", 
                    category=UserWarning, module="gymnasium.wrappers.rendering")

import torch as t 
import numpy as np
import argparse
from tqdm import tqdm
from models import ConvolutionalSmall, ImpalaLike
from evals import run_evaluation
from runner import run_training
from utils import (
    init_training, 
    init_tracking, 
    update_episode_tracking,
    log_training_metrics, 
    save_checkpoint, 
    handle_env_resets,
    get_torch_compatible_actions,
    get_entropy,
    readable_timestamp,
    load_checkpoint,
    setup_from_checkpoint
)
from environment import make_curriculum_env

from config import (
    IMPALA_TRAIN_CONFIG,
    IMPALA_TEST_CONFIG,
    IMPALA_TUNE_CONFIG,
    CONV_TRAIN_CONFIG,
    CONV_TEST_CONFIG,
    CONV_TUNE_CONFIG
) 
from curriculum import (
    get_curriculum_phase,
    get_phase_distribution,
    should_change_phase,
    get_phase_description
)

def transition_curriculum_phase(env, config, new_phase, device):
    """Handle transition to a new curriculum phase.
    
    Closes the current environment and creates a new one with the updated
    level distribution for the new phase.
    
    Args:
        env: Current environment to close
        config: Training configuration
        new_phase: Index of the new curriculum phase
        device: Device for tensor operations
    
    Returns:
        Tuple of (new_env, environment_tensordict, initial_state)
    """
    # Close the old environment (required by stable-retro)
    env.close()
    
    # Get new level distribution for this phase
    level_distribution = get_phase_distribution(new_phase, config.num_envs)
    
    # Print the transition
    print(f"\n{'='*60}")
    print(f"CURRICULUM TRANSITION: {get_phase_description(new_phase)}")
    print(f"Level distribution: {level_distribution}")
    print(f"{'='*60}\n")
    
    # Create new environment with updated distribution
    new_env = make_curriculum_env(config.num_envs, level_distribution)
    
    # Reset and get initial state
    environment = new_env.reset()
    state = environment['pixels']
    if config.num_envs == 1 and state.dim() == 3:
        state = state.unsqueeze(0)
    
    return new_env, environment, state

def init_curriculum_training(agent, config, device):
    """Initialize training with curriculum learning.
    
    Starts with phase 0 distribution instead of default levels.
    """
    from ppo import PPO
    from buffer import RolloutBuffer
    
    policy = PPO(agent, config, device)
    buffer = RolloutBuffer(config.buffer_size // config.num_envs, config.num_envs, device)
    
    # Get initial phase distribution
    initial_phase = 0
    level_distribution = get_phase_distribution(initial_phase, config.num_envs)
    
    print(f"\n{'='*60}")
    print(f"CURRICULUM LEARNING ENABLED")
    print(f"Starting with: {get_phase_description(initial_phase)}")
    print(f"Level distribution: {level_distribution}")
    print(f"{'='*60}\n")
    
    env = make_curriculum_env(config.num_envs, level_distribution)
    environment = env.reset()
    state = environment['pixels']
    if config.num_envs == 1 and state.dim() == 3:
        state = state.unsqueeze(0)
    
    return policy, buffer, env, environment, state


def training_loop(agent, config, num_eval_episodes=5, checkpoint_path=None, resume=False):

    run = config.setup_wandb()
    device = "cuda" if t.cuda.is_available() else "cpu"
    agent = agent.to(device)
    
    # Use curriculum initialization if enabled
    if config.use_curriculum:
        policy, buffer, env, environment, state = init_curriculum_training(agent, config, device)
        current_phase = 0
    else:
        policy, buffer, env, environment, state = init_training(agent, config, device)
        current_phase = None  # Not using curriculum
    
    if checkpoint_path is not None:
        start_step, tracking = setup_from_checkpoint(checkpoint_path, agent, policy, config, device, resume)
        # Update curriculum phase if resuming with curriculum
        if config.use_curriculum:
            current_phase = get_curriculum_phase(start_step, config.num_training_steps)
            if current_phase != 0:
                # Need to transition to correct phase for resumed training
                env, environment, state = transition_curriculum_phase(
                    env, config, current_phase, device
                )
    else:
        tracking = init_tracking(config)
        start_step = 0
        
    param = 0
    for name, p in agent.named_parameters():
        param += p.numel()
        print(name, p.numel())
    print(f"Total Params: {param}")
    if device == "cuda":
        print(f"Training {config.architecture} on GPU")
    else:
        print(f"Training {config.architecture} on CPU")
    
    pbar = tqdm(range(start_step, config.num_training_steps), disable=not config.show_progress)
    
    for step in pbar:
        # Check for curriculum phase transition
        if config.use_curriculum and should_change_phase(step, config.num_training_steps, current_phase):
            new_phase = get_curriculum_phase(step, config.num_training_steps)
            env, environment, state = transition_curriculum_phase(
                env, config, new_phase, device
            )
            current_phase = new_phase
            # Clear buffer since environment changed
            buffer.clear()
        
        policy.c2 = get_entropy(step, total_steps=config.num_training_steps, max_entropy=config.c2) 
        
        actions, log_probs, values = policy.action_selection(state)
        environment["action"] = get_torch_compatible_actions(actions)
        environment = env.step(environment)
        next_state = environment["next"]["pixels"]

        if config.num_envs == 1 and next_state.dim() == 3:
            next_state = next_state.unsqueeze(0)

        rewards = environment["next"]["reward"]
        dones = environment["next"]["done"] # Isn't Dones Terminated and Truncated? 
        trunc = environment["next"].get("truncated", t.zeros_like(dones))
        terminated = dones | trunc
        
        if config.num_envs == 1:
            if rewards.dim() == 0:
                rewards = rewards.unsqueeze(0)
            if dones.dim() == 0:
                dones = dones.unsqueeze(0)


        buffer.store(
            state.cpu().numpy(),
            rewards.squeeze().cpu().numpy(),
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
            dones.squeeze().cpu().numpy(),
        )
        
        tracking['total_env_steps'] += config.num_envs
        update_episode_tracking(tracking, config, rewards, terminated)

        if config.show_progress and len(tracking['completed_rewards']) > 0:
            postfix = {
                'episodes': tracking['episode_num'],
                'mean_reward': f"{np.mean(tracking['completed_rewards']):.2f}",
                'updates': tracking['num_updates'],
                'lr': f"{policy.get_current_lr():.2e}",
                'c2': f"{policy.c2:.4f}",
            }
            if config.use_curriculum:
                postfix['phase'] = current_phase
            pbar.set_postfix(postfix)
    
        if step - tracking['last_eval_step']  >= config.eval_freq:
            run_evaluation(agent.__class__, policy, tracking, config, run, step, num_eval_episodes)

        state, environment = handle_env_resets(env, environment, next_state, terminated, config.num_envs)      
          
        if buffer.idx == buffer.capacity:
            mean_loss = policy.update(buffer, config, next_state=state)
            tracking['num_updates'] += 1
            
            log_training_metrics(tracking, mean_loss, policy, config, step)
            tracking['completed_rewards'].clear()
            tracking['completed_lengths'].clear()
            
            buffer.clear()
            
            if step - tracking['last_checkpoint'] >= config.checkpoint_freq:
                save_checkpoint(agent, policy, tracking, config, run, step)
                print(f"Model checkpoint saved at step {step}")
    else:
        env.close()
    # Perform final evaluation and store last weights
    run_evaluation(agent.__class__, policy, tracking, config, run, step, num_eval_episodes)
    save_checkpoint(agent, policy, tracking, config, run, step)

    if config.USE_WANDB:
        import wandb
        wandb.finish()
    else:
        print("Training completed successfully")
    
    return agent

def train(model, config, num_eval_episodes=9):
    agent = model()
    return training_loop(agent, config, num_eval_episodes)


def finetune(model, checkpoint_path, config, num_eval_episodes=9):
    agent = model()
    return training_loop(agent, config, num_eval_episodes, checkpoint_path=checkpoint_path)

def resume(model, checkpoint_path, config, num_eval_episodes=9):
    agent = model()
    return training_loop(agent, config, num_eval_episodes, checkpoint_path=checkpoint_path, resume=True)

if __name__ == "__main__":
    run_training()


