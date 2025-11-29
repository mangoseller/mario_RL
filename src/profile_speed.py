import time
import retro
import numpy as np
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from wrappers import Discretizer, FrameSkipAndTermination, MaxStepWrapper, DilatedFrameStack
from rewards import ComposedRewardWrapper
from environment import MARIO_ACTIONS

def time_env(env, name, steps=500):
    env.reset()
    action = env.action_space.sample()
    t0 = time.perf_counter()
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            env.reset()
    elapsed = time.perf_counter() - t0
    print(f"{name:30s}: {steps/elapsed:6.0f} steps/sec")
    return env

# Build up wrapper stack one at a time
env = retro.make('SuperMarioWorld-Snes', state='YoshiIsland2', render_mode='rgb_array')
env = time_env(env, "1. Raw Retro")

env = Discretizer(env, MARIO_ACTIONS)
env = time_env(env, "2. + Discretizer")

env = ComposedRewardWrapper(env)
env = time_env(env, "3. + ComposedReward")

env = FrameSkipAndTermination(env, skip=4)
env = time_env(env, "4. + FrameSkip(4)")

env = MaxStepWrapper(env, max_steps=8000)
env = time_env(env, "5. + MaxStep")

env = ResizeObservation(env, (84, 84))
env = time_env(env, "6. + Resize")

env = GrayscaleObservation(env)
env = time_env(env, "7. + Grayscale")

env = DilatedFrameStack(env, k=4, dilation=4)
env = time_env(env, "8. + DilatedFrameStack")

env.close()
