import retro
import numpy as np
import argparse
import time
import os
from pynput import keyboard

# Arrow keys + space (jump) + A (run)
ACTION_MAP = {
    frozenset(): 0,                                    # nothing
    frozenset(['right']): 1,                           # RIGHT
    frozenset(['right', 'a']): 2,                      # RIGHT + run
    frozenset(['right', 'space']): 3,                  # RIGHT + jump
    frozenset(['right', 'a', 'space']): 4,             # RIGHT + run + jump
    frozenset(['left']): 5,                            # LEFT
    frozenset(['left', 'a']): 6,                       # LEFT + run
    frozenset(['left', 'space']): 7,                   # LEFT + jump
    frozenset(['left', 'a', 'space']): 8,              # LEFT + run + jump
    frozenset(['space']): 9,                           # jump
    frozenset(['down']): 11,                           # DOWN (pipes)
    frozenset(['up']): 12,                             # UP
}

pressed = set()

def on_press(key):
    if key == keyboard.Key.up:
        pressed.add('up')
    elif key == keyboard.Key.down:
        pressed.add('down')
    elif key == keyboard.Key.left:
        pressed.add('left')
    elif key == keyboard.Key.right:
        pressed.add('right')
    elif key == keyboard.Key.space:
        pressed.add('space')
    elif hasattr(key, 'char') and key.char:
        pressed.add(key.char)

def on_release(key):
    if key == keyboard.Key.up:
        pressed.discard('up')
    elif key == keyboard.Key.down:
        pressed.discard('down')
    elif key == keyboard.Key.left:
        pressed.discard('left')
    elif key == keyboard.Key.right:
        pressed.discard('right')
    elif key == keyboard.Key.space:
        pressed.discard('space')
    elif hasattr(key, 'char') and key.char:
        pressed.discard(key.char)
    if key == keyboard.Key.esc:
        return False

def get_action():
    return ACTION_MAP.get(frozenset(pressed), 0)

def record(level, save_path, skip=2):
    from environment import MARIO_ACTIONS
    from wrappers import Discretizer, FrameSkipAndTermination, MaxStepWrapper
    from rewards import ComposedRewardWrapper
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    env = retro.make('SuperMarioWorld-Snes', state=level, render_mode='human')
    env = Discretizer(env, MARIO_ACTIONS)
    env = ComposedRewardWrapper(env)
    env = FrameSkipAndTermination(env, skip=skip)
    env = MaxStepWrapper(env, max_steps=10000)
    
    import torch as t
    from torchvision.transforms import functional as TF
    
    def process_state(obs):
        obs = t.from_numpy(obs).permute(2, 0, 1).float() / 255.0
        obs = TF.rgb_to_grayscale(obs)
        obs = TF.resize(obs, [84, 84])
        return obs
    
    states, actions = [], []
    frame_stack = []
    
    obs, info = env.reset()
    
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    print("Recording... Arrow keys move, SPACE jump, A run. ESC to stop.")
    
    done = False
    while not done and listener.running:
        action = get_action()
        
        processed = process_state(obs)
        frame_stack.append(processed)
        if len(frame_stack) > 4:
            frame_stack.pop(0)
        
        if len(frame_stack) == 4:
            state = t.cat(frame_stack, dim=0)
            states.append(state.numpy())
            actions.append(action)
        
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        
        time.sleep(0.03)
    
    listener.stop()
    env.close()
    
    if states:
        np.savez(save_path, states=np.stack(states), actions=np.array(actions))
        print(f"Saved {len(states)} frames to {save_path}")
    else:
        print("No frames recorded")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', required=True)
    parser.add_argument('--save', required=True)
    args = parser.parse_args()
    record(args.level, args.save)
