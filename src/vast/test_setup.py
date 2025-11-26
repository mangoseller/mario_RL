#!/usr/bin/env python
"""
Test script to verify everything is set up correctly before training.
Run this FIRST to avoid wasting GPU time on setup issues.

Usage: python test_setup.py
"""

import sys
import os

def print_status(name, success, details=""):
    status = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{color}[{status}]{reset} {name}")
    if details:
        print(f"    {details}")
    return success

def test_imports():
    """Test all required imports"""
    print("\n" + "="*50)
    print("TESTING IMPORTS")
    print("="*50)
    
    all_passed = True
    
    imports = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("retro", "stable-retro"),
        ("torchrl", "TorchRL"),
        ("gymnasium", "Gymnasium"),
        ("einops", "einops"),
        ("tqdm", "tqdm"),
        ("wandb", "Weights & Biases"),
        ("torchvision", "TorchVision"),
    ]
    
    for module, name in imports:
        try:
            __import__(module)
            all_passed &= print_status(f"{name} ({module})", True)
        except ImportError as e:
            all_passed &= print_status(f"{name} ({module})", False, str(e))
    
    return all_passed

def test_cuda():
    """Test CUDA availability"""
    print("\n" + "="*50)
    print("TESTING CUDA")
    print("="*50)
    
    import torch
    
    cuda_available = torch.cuda.is_available()
    print_status("CUDA available", cuda_available)
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_status(f"GPU: {device_name}", True, f"{memory:.1f} GB VRAM")
        
        # Test tensor operation on GPU
        try:
            x = torch.randn(100, 100, device='cuda')
            y = x @ x.T
            del x, y
            torch.cuda.empty_cache()
            print_status("GPU tensor operations", True)
        except Exception as e:
            print_status("GPU tensor operations", False, str(e))
            return False
    else:
        print("    WARNING: Training will be very slow on CPU!")
    
    return True

def test_retro_rom():
    """Test that the ROM is properly installed"""
    print("\n" + "="*50)
    print("TESTING RETRO + ROM")
    print("="*50)
    
    import retro
    
    # Check data path
    data_path = retro.data.path()
    print_status("Retro data path", True, data_path)
    
    # Check if SuperMarioWorld-Snes exists
    game_path = os.path.join(data_path, "stable", "SuperMarioWorld-Snes")
    game_exists = os.path.exists(game_path)
    print_status("Game folder exists", game_exists, game_path)
    
    if not game_exists:
        return False
    
    # Check required files
    required_files = ["rom.sfc", "data.json"]
    optional_files = ["scenario.json", "metadata.json"]
    
    for f in required_files:
        exists = os.path.exists(os.path.join(game_path, f))
        if not print_status(f"Required: {f}", exists):
            return False
    
    for f in optional_files:
        exists = os.path.exists(os.path.join(game_path, f))
        print_status(f"Optional: {f}", exists)
    
    # Try to create environment
    try:
        env = retro.make('SuperMarioWorld-Snes', state='YoshiIsland2')
        obs = env.reset()
        print_status("Create environment (YoshiIsland2)", True, f"Obs shape: {obs[0].shape}")
        
        # Take a few steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
        print_status("Environment stepping", True)
        
        env.close()
    except Exception as e:
        print_status("Create environment", False, str(e))
        return False
    
    # Test other states used in training
    test_states = ['YoshiIsland1', 'DonutPlains1', 'DonutPlains3', 'DonutPlains4', 'DonutPlains5']
    for state in test_states:
        try:
            env = retro.make('SuperMarioWorld-Snes', state=state)
            env.reset()
            env.close()
            print_status(f"State: {state}", True)
        except Exception as e:
            print_status(f"State: {state}", False, str(e))
    
    return True

def test_display():
    """Test that display/Xvfb is working"""
    print("\n" + "="*50)
    print("TESTING DISPLAY (Xvfb)")
    print("="*50)
    
    display = os.environ.get('DISPLAY', None)
    has_display = display is not None
    print_status("DISPLAY env var set", has_display, f"DISPLAY={display}")
    
    if not has_display:
        print("    WARNING: No DISPLAY set. Video recording may fail.")
        print("    Run: Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99")
    
    return True  # Non-fatal

def test_wandb():
    """Test WandB configuration"""
    print("\n" + "="*50)
    print("TESTING WANDB")
    print("="*50)
    
    api_key = os.environ.get('WANDB_API_KEY', None)
    has_key = api_key is not None and len(api_key) > 10
    
    if has_key:
        print_status("WANDB_API_KEY set", True, f"Key: {api_key[:8]}...")
        
        # Try to verify the key
        try:
            import wandb
            wandb.login(key=api_key)
            print_status("WandB login", True)
        except Exception as e:
            print_status("WandB login", False, str(e))
    else:
        print_status("WANDB_API_KEY set", False, 
                    "Set with: export WANDB_API_KEY='your-key-here'")
        print("    Training will work but won't log to WandB")
    
    return True  # Non-fatal

def test_project_imports():
    """Test that project code imports correctly"""
    print("\n" + "="*50)
    print("TESTING PROJECT CODE")
    print("="*50)
    
    # Add src to path
    sys.path.insert(0, '/workspace/src')
    sys.path.insert(0, 'src')  # For local testing
    
    modules = [
        ("models", "Models (ConvolutionalSmall, ImpalaLike, ImpalaLarge)"),
        ("ppo", "PPO"),
        ("buffer", "RolloutBuffer"),
        ("config", "TrainingConfig"),
        ("rewards", "Reward functions"),
        ("wrappers", "Environment wrappers"),
        ("environment", "Environment creation"),
        ("utils", "Utilities"),
    ]
    
    all_passed = True
    for module, name in modules:
        try:
            __import__(module)
            all_passed &= print_status(name, True)
        except Exception as e:
            all_passed &= print_status(name, False, str(e))
    
    return all_passed

def test_full_environment_pipeline():
    """Test the complete environment pipeline as used in training"""
    print("\n" + "="*50)
    print("TESTING FULL TRAINING PIPELINE")
    print("="*50)
    
    try:
        sys.path.insert(0, '/workspace/src')
        sys.path.insert(0, 'src')
        
        from environment import make_training_env
        from models import ImpalaLarge, ImpalaLike, ConvolutionalSmall
        from ppo import PPO
        from config import IMPALA_LARGE_TEST_CONFIG
        import torch as t
        
        # Test single-env creation
        print("Creating single training environment...")
        env = make_training_env(num_envs=1)
        environment = env.reset()
        state = environment['pixels']
        print_status("Single env creation", True, f"State shape: {state.shape}")
        env.close()
        
        # Test model creation
        print("\nTesting model creation...")
        device = "cuda" if t.cuda.is_available() else "cpu"
        
        for model_cls in [ConvolutionalSmall, ImpalaLike, ImpalaLarge]:
            model = model_cls().to(device)
            param_count = sum(p.numel() for p in model.parameters())
            
            # Test forward pass
            dummy_input = t.randn(1, 4, 84, 84, device=device)
            policy_out, value_out = model(dummy_input)
            
            print_status(f"{model_cls.__name__}", True, 
                        f"{param_count:,} params, policy: {policy_out.shape}, value: {value_out.shape}")
            del model
        
        if device == "cuda":
            t.cuda.empty_cache()
        
        # Test PPO creation
        print("\nTesting PPO agent...")
        model = ImpalaLarge().to(device)
        config = IMPALA_LARGE_TEST_CONFIG
        ppo = PPO(model, config, device)
        print_status("PPO agent creation", True)
        
        # Test action selection
        dummy_state = t.randn(1, 4, 84, 84, device=device)
        actions, log_probs, values = ppo.action_selection(dummy_state)
        print_status("Action selection", True, 
                    f"Action: {actions.item()}, Log prob: {log_probs.item():.3f}, Value: {values.item():.3f}")
        
        del model, ppo
        if device == "cuda":
            t.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        import traceback
        print_status("Full pipeline test", False, str(e))
        traceback.print_exc()
        return False

def test_video_recording():
    """Test that video recording works"""
    print("\n" + "="*50)
    print("TESTING VIDEO RECORDING")
    print("="*50)
    
    try:
        import retro
        from gymnasium.wrappers import RecordVideo
        import tempfile
        import os
        
        # Create temp directory for video
        with tempfile.TemporaryDirectory() as tmpdir:
            env = retro.make('SuperMarioWorld-Snes', state='YoshiIsland2', render_mode='rgb_array')
            env = RecordVideo(env, video_folder=tmpdir, episode_trigger=lambda x: True, name_prefix="test")
            
            obs = env.reset()
            for _ in range(60):  # ~1 second of gameplay
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                if done:
                    break
            
            env.close()
            
            # Check if video was created
            video_files = [f for f in os.listdir(tmpdir) if f.endswith('.mp4')]
            if video_files:
                video_path = os.path.join(tmpdir, video_files[0])
                video_size = os.path.getsize(video_path)
                print_status("Video recording", True, f"Created {video_files[0]} ({video_size} bytes)")
                return True
            else:
                print_status("Video recording", False, "No video file created")
                return False
                
    except Exception as e:
        import traceback
        print_status("Video recording", False, str(e))
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("  MARIORL SETUP VERIFICATION")
    print("  Run this BEFORE training to catch issues early!")
    print("="*60)
    
    results = {}
    
    results['imports'] = test_imports()
    results['cuda'] = test_cuda()
    results['retro'] = test_retro_rom()
    results['display'] = test_display()
    results['wandb'] = test_wandb()
    results['project'] = test_project_imports()
    results['pipeline'] = test_full_environment_pipeline()
    results['video'] = test_video_recording()
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    
    critical_tests = ['imports', 'retro', 'project', 'pipeline']
    critical_passed = all(results.get(t, False) for t in critical_tests)
    
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        critical = " (CRITICAL)" if name in critical_tests else ""
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"  {color}{status}{reset} - {name}{critical}")
    
    print()
    if critical_passed:
        print("\033[92m" + "="*60)
        print("  ✓ ALL CRITICAL TESTS PASSED - READY TO TRAIN!")
        print("="*60 + "\033[0m")
        return 0
    else:
        print("\033[91m" + "="*60)
        print("  ✗ CRITICAL TESTS FAILED - FIX ISSUES BEFORE TRAINING")
        print("="*60 + "\033[0m")
        return 1

if __name__ == "__main__":
    sys.exit(main())
