import gymnasium as gym
import numpy as np
from rewards import REWARD_CONFIG
# h/wrappers.py

class DilatedFrameStack(gym.Wrapper):
    def __init__(self, env, k=4, dilation=8):
        super().__init__(env)
        self.k = k
        self.dilation = dilation
        self.buffer_size = (k - 1) * dilation + 1
        self.frames = []
        
        # Update observation space
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(shp[0] * k, shp[1], shp[2]), 
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill buffer with duplicates of the first frame
        self.frames = [obs] * self.buffer_size
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        if len(self.frames) > self.buffer_size:
            self.frames.pop(0)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        # Gather frames: [t, t-dilation, t-2*dilation, ...]
        # Since we append new frames to the end, the 'current' frame is at index -1.
        selected_frames = [self.frames[-(1 + i * self.dilation)] for i in range(self.k)]
        return np.concatenate(selected_frames, axis=0)

class Discretizer(gym.ActionWrapper):
# Wrap an env to use COMBOS as its discrete action space
    def __init__(self, env, combos):
        super().__init__(env)
        buttons = env.unwrapped.buttons 
        self._decode_discrete_action = []
        for c in combos:
            arr = np.array([False] * env.action_space.n) # All positions except for the button we want to press should be False
            for button in c:
                arr[buttons.index(button)] = True # Set the button position in the array to True
            self._decode_discrete_action.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action)) 
    
    def action(self, action):
        return self._decode_discrete_action[action].copy() # Convert integer action into expected boolean arr of button presses


class FrameSkipAndTermination(gym.Wrapper):
    """Frame skip wrapper that handles termination on death and level completion"""
    def __init__(self, env, skip=2):
        super().__init__(env)
        self.skip = skip
        self.prev_lives = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_lives = info.get('lives', None)
        return obs, info
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for i in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            
            # Check for level completion
            level_complete = info.get('level_complete', 80)
            if level_complete != 80:
                terminated = True
                break
            
            # Check for life loss (death)
            current_lives = info.get('lives', None)
            if self.prev_lives is not None and current_lives is not None:
                if current_lives < self.prev_lives:
                    terminated = True
                    break
            
            self.prev_lives = current_lives
        
        return obs, total_reward, terminated, truncated, info


class MaxStepWrapper(gym.Wrapper):

    def __init__(self, env, max_steps=85000):
        super().__init__(env)
        self.max_steps = max_steps
        self.step_count = 0
        self.penalty = REWARD_CONFIG['max_steps_penalty']
    
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        if self.step_count >= self.max_steps:
            truncated = True
            reward += self.penalty
        
        return obs, reward, terminated, truncated, info
