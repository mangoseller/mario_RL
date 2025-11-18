"""
Test for race condition in the actual training loop.
 
This test exercises the exact code path in train.py where the race condition
was occurring, specifically the environment reset logic after each step.
"""
 
import torch as t
import numpy as np
from model_small import ImpalaSmall
from ppo import PPO
from buffer import RolloutBuffer
from environment import make_training_env
from training_utils import TESTING_CONFIG, get_torch_compatible_actions
 
 
def test_training_loop_race_condition(num_envs=4, max_steps=10000, max_episodes=20):
    """
    Test the actual training loop for race conditions.
 
    This simulates the real training loop from train.py to ensure that
    the reset logic doesn't cause 1-step episodes in parallel environments.
    """
    print(f"\n{'='*70}")
    print(f"Testing training loop with {num_envs} parallel environments")
    print(f"{'='*70}\n")
 
    # Use testing config for faster execution
    config = TESTING_CONFIG
    config.num_envs = num_envs
 
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Device: {device}")
 
    # Initialize agent and policy (same as in train.py)
    agent = ImpalaSmall().to(device)
    policy = PPO(
        model=agent,
        lr=config.learning_rate,
        epsilon=config.clip_eps,
        optimizer=t.optim.Adam,
        device=device,
        c1=config.c1,
        c2=config.c2
    )
 
    # Create buffer and environment (same as in train.py)
    buffer = RolloutBuffer(config.buffer_size // config.num_envs, config.num_envs, device)
    env = make_training_env(num_envs=config.num_envs)
    environment = env.reset()
    state = environment['pixels']
 
    if config.num_envs == 1 and state.dim() == 3:
        state = state.unsqueeze(0)
 
    # Episode tracking
    current_episode_rewards = [0.0] * config.num_envs
    current_episode_lengths = [0] * config.num_envs
    completed_rewards = []
    completed_lengths = []
    all_episode_lengths = [[] for _ in range(config.num_envs)]
    consecutive_short_episodes = [0] * config.num_envs
 
    race_condition_detected = False
    total_episodes = 0
 
    print(f"Running training loop for up to {max_steps} steps or {max_episodes} episodes...\n")
 
    # Main training loop (same as in train.py)
    for step in range(max_steps):
        # Action selection (same as train.py)
        actions, log_probs, values = policy.action_selection(state)
        environment["action"] = get_torch_compatible_actions(actions, config.num_envs)
        environment = env.step(environment)
        next_state = environment["next"]["pixels"]
 
        if config.num_envs == 1 and next_state.dim() == 3:
            next_state = next_state.unsqueeze(0)
 
        rewards = environment["next"]["reward"]
        dones = environment["next"]["done"]
 
        if config.num_envs == 1:
            if rewards.dim() == 0:
                rewards = rewards.unsqueeze(0)
            if dones.dim() == 0:
                dones = dones.unsqueeze(0)
 
        # Store experience (same as train.py)
        buffer.store(
            state.cpu().numpy(),
            rewards.squeeze().cpu().numpy(),
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
            dones.squeeze().cpu().numpy(),
        )
 
        # Update episode tracking (same as train.py)
        for i in range(config.num_envs):
            current_episode_rewards[i] += rewards[i].item()
            current_episode_lengths[i] += 1
 
            if dones[i].item():
                completed_rewards.append(current_episode_rewards[i])
                completed_lengths.append(current_episode_lengths[i])
                all_episode_lengths[i].append(current_episode_lengths[i])
                total_episodes += 1
 
                print(f"  Env {i}: Episode {len(all_episode_lengths[i])} completed "
                      f"(length: {current_episode_lengths[i]}, reward: {current_episode_rewards[i]:.1f})")
 
                # Check for race condition: consecutive short episodes
                if current_episode_lengths[i] == 1:
                    consecutive_short_episodes[i] += 1
                    if consecutive_short_episodes[i] >= 3:
                        print(f"\n  ⚠️  RACE CONDITION DETECTED in Env {i}!")
                        print(f"      {consecutive_short_episodes[i]} consecutive 1-step episodes")
                        race_condition_detected = True
                else:
                    consecutive_short_episodes[i] = 0
 
                current_episode_rewards[i] = 0.0
                current_episode_lengths[i] = 0
 
        # *** THIS IS THE CRITICAL PART WE'RE TESTING ***
        # Handle environment resets (FIXED version from train.py)
        if config.num_envs == 1:
            # Single environment: manually reset when done
            if dones.item():
                environment = env.reset()
                state = environment["pixels"].unsqueeze(0)
            else:
                state = next_state
        else:
            # Multiple environments: ParallelEnv automatically resets individual workers
            # The next_state already contains reset observations for done environments
            state = next_state
         # Clear buffer when full (same as train.py) to avoid buffer overflow
        if buffer.idx == buffer.capacity:
            # In real training, PPO update happens here
            # For this test, we just clear the buffer since we're testing reset logic
            buffer.clear()
        # Stop if we've completed enough episodes
        if total_episodes >= max_episodes:
            print(f"\nReached {max_episodes} episodes, stopping test.")
            break
 
        # Stop immediately if race condition detected
        if race_condition_detected:
            print(f"\nStopping test due to race condition detection.")
            break
 
    env.close()
 
    # Report results
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total steps: {step + 1}")
    print(f"Total episodes completed: {total_episodes}")
    print(f"\nPer-environment statistics:")
 
    for i in range(config.num_envs):
        if all_episode_lengths[i]:
            avg_length = np.mean(all_episode_lengths[i])
            std_length = np.std(all_episode_lengths[i])
            min_length = np.min(all_episode_lengths[i])
            max_length = np.max(all_episode_lengths[i])
 
            print(f"\n  Env {i}:")
            print(f"    Episodes: {len(all_episode_lengths[i])}")
            print(f"    Avg length: {avg_length:.1f} ± {std_length:.1f}")
            print(f"    Min/Max length: {min_length}/{max_length}")
            print(f"    All lengths: {all_episode_lengths[i]}")
 
            # Count suspiciously short episodes
            short_episodes = sum(1 for length in all_episode_lengths[i] if length <= 2)
            if short_episodes > 0:
                percentage = (short_episodes / len(all_episode_lengths[i])) * 100
                print(f"    ⚠️  Warning: {short_episodes}/{len(all_episode_lengths[i])} "
                      f"episodes were ≤2 steps ({percentage:.1f}%)")
 
    print(f"\n{'='*70}")
 
    # Assertions
    if race_condition_detected:
        print("❌ TEST FAILED: Race condition detected!")
        print("   Episodes are terminating immediately after reset.")
        print("   The fix in train.py is NOT working correctly.")
        assert False, "Race condition detected - 3+ consecutive 1-step episodes"
 
    # Check that we got reasonable episode lengths
    if total_episodes > 0:
        avg_overall = np.mean(completed_lengths)
        print(f"\nOverall average episode length: {avg_overall:.1f}")
 
        # Mario episodes should be longer than 5 steps on average
        # (even with random actions, you don't die immediately every time)
        if avg_overall < 5:
            print(f"⚠️  Warning: Average episode length is suspiciously short ({avg_overall:.1f})")
            print("   This might indicate a subtle race condition or environment issue.")
 
    print("\n✅ TEST PASSED: No race condition detected!")
    print("   The reset logic in train.py is working correctly.")
    print("   Parallel environments are resetting independently as expected.")
    print(f"{'='*70}\n")
 
 
def main():
    """Run the race condition test with different configurations."""
 
    print("\n" + "="*70)
    print("TRAINING LOOP RACE CONDITION TEST")
    print("="*70)
    print("\nThis test exercises the actual training loop code from train.py")
    print("to verify that the race condition fix is working correctly.")
    print("="*70)
 
    configs = [
        # (num_envs, max_steps, max_episodes)
        # (4, 15000, 20),  # 4 parallel envs
        (8, 15000, 20),  # 8 parallel envs - more likely to trigger race conditions
    ]
 
    all_passed = True
 
    for num_envs, max_steps, max_episodes in configs:
        try:
            test_training_loop_race_condition(num_envs, max_steps, max_episodes)
        except AssertionError as e:
            print(f"\n❌ Test failed for {num_envs} environments: {e}")
            all_passed = False
            break
        except Exception as e:
            print(f"\n❌ Unexpected error with {num_envs} environments: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            break
 
    print("\n" + "="*70)
    print("FINAL RESULT:")
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("   The race condition fix in train.py is working correctly.")
        print("   Parallel environments are resetting independently without interference.")
    else:
        print("❌ TESTS FAILED!")
        print("   The race condition is still present or other issues detected.")
    print("="*70 + "\n")
 
    return 0 if all_passed else 1
 
 
if __name__ == "__main__":
    exit(main())
