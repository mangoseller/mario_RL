from torchrl.envs import TransformedEnv, GymWrapper
from torchrl.envs.transforms import (
    ToTensorImage,
    Resize,
    GrayScale,
    CatFrames,
    StepCounter,
    RewardSum, 
    Compose,
    FrameSkipTransform,
)
import retro 
training=False

base_env = GymWrapper(
    retro.make(
        'SuperMarioWorld-Snes', 
        use_restricted_actions=retro.Actions.DISCRETE, 
        render_mode='rgb_array' if training else 'human'
    )      
)

env = TransformedEnv(base_env, Compose(*[
    FrameSkipTransform(frame_skip = 4),
    ToTensorImage(), # Convert stable-retro return values to PyTorch Tensors
    Resize(84, 84), # Can also do 96x96, 128x128
    GrayScale(),
    CatFrames(N=4, dim=-3), # dim -3 stacks frames over the channel dimension (does this make sense with gray frames?)
    StepCounter(),
    RewardSum(),
  ]))


