
from torchrl.envs import TransformedEnv, GymWrapper, ParallelEnv 
from torchrl.envs.transforms import (
    ToTensorImage,
    Resize,
    GrayScale,
    CatFrames,
    StepCounter,
    RewardSum, 
    Compose,
)
import retro
from gymnasium.wrappers import RecordVideo
from wrappers import Discretizer, FrameSkipAndTermination, MaxStepWrapper
from rewards import ComposedRewardWrapper
from torchvision.transforms import InterpolationMode
from curriculum import compute_level_distribution, uniform_distribution


# Default level configuration for non-curriculum training
DEFAULT_LEVELS = {
    'YoshiIsland2': 0.5,
    'YoshiIsland3': 0.5,
}

MARIO_ACTIONS = [
    [],                   # Do nothing
    ['RIGHT'],            # Walk right
    ['RIGHT', 'Y'],       # Run right  
    ['RIGHT', 'B'],       # Jump right
    ['RIGHT', 'Y', 'B'],  # Run + jump right
    ['LEFT'],             # Walk left
    ['LEFT', 'Y'],        # Run left
    ['LEFT', 'B'],        # Jump left
    ['LEFT', 'Y', 'B'],   # Run + jump left
    ['B'],                # Jump in place
    ['X'],                # Attack
    ['DOWN'],             # Duck/enter pipe
    ['UP'],               # Look up/climb
    ['A'],                # Spin Jump
]


def _wrap_env(env, skip=2, record=False, record_dir=None):
    """Apply all wrappers to a raw retro environment."""
    wrapped_env = Discretizer(env, MARIO_ACTIONS)
    wrapped_env = ComposedRewardWrapper(wrapped_env)
    wrapped_env = FrameSkipAndTermination(wrapped_env, skip=skip)
    wrapped_env = MaxStepWrapper(wrapped_env, max_steps=8000)
    
    if record:
        from utils import readable_timestamp
        wrapped_env = RecordVideo(
            wrapped_env,
            video_folder=record_dir,
            episode_trigger=lambda x: True,
            name_prefix=f"eval_{readable_timestamp()}"
        )
    
    wrapped_env = GymWrapper(wrapped_env)

    return TransformedEnv(wrapped_env, Compose([
        ToTensorImage(),
        Resize(84, 84, interpolation=InterpolationMode.NEAREST),
        GrayScale(),
        CatFrames(N=4, dim=-3),
        StepCounter(),
        RewardSum(),
    ]))


def make_env(
    num_envs: int = 1,
    level_weights: dict = None,
    level_distribution: list = None,
    render_human: bool = False,
    frame_skip: int = 2,
    record: bool = False,
    record_dir: str = None,
    specs: dict = None,  # [FIX] Added specs argument
):
    """
    Create training environment(s) with flexible level configuration.
    
    Args:
        num_envs: Number of parallel environments (1 for single env)
        level_weights: Dict mapping level names to weights.
        level_distribution: Explicit list of level names for each env.
        render_human: If True, render to screen (only works with num_envs=1)
        frame_skip: Number of frames to skip per action
        record: If True, record videos
        record_dir: Directory for recorded videos
        specs: Optional dict containing env specs (observation_spec, etc.) 
               to avoid creating a dummy env in ParallelEnv.
    """
    # Determine level distribution
    if level_distribution is not None:
        if len(level_distribution) != num_envs:
            raise ValueError(
                f"level_distribution length ({len(level_distribution)}) "
                f"must match num_envs ({num_envs})"
            )
        dist = level_distribution
    elif level_weights is not None:
        dist = compute_level_distribution(num_envs, level_weights)
    else:
        dist = compute_level_distribution(num_envs, DEFAULT_LEVELS)
    
    # Determine render mode
    if num_envs == 1:
        render_mode = 'human' if render_human else 'rgb_array'
    else:
        if render_human:
            raise ValueError("render_human=True only supported with num_envs=1")
        render_mode = 'rgb_array'
    
    # Create environment(s)
    if num_envs == 1:
        raw_env = retro.make(
            'SuperMarioWorld-Snes',
            state=dist[0],
            render_mode=render_mode,
        )
        return _wrap_env(raw_env, skip=frame_skip, record=record, record_dir=record_dir)
    else:
        def create_env(level):
            raw_env = retro.make(
                'SuperMarioWorld-Snes',
                state=level,
                render_mode='rgb_array',
            )
            return _wrap_env(raw_env, skip=frame_skip)
        
        env_kwargs = {
            'num_workers': num_envs,
            'create_env_fn': create_env,
            'create_env_kwargs': [{'level': level} for level in dist],
        }
        
        if specs:
            env_kwargs.update(specs)
            
        return ParallelEnv(**env_kwargs)


def make_eval_env(level: str, record_dir: str = None):
    """
    Create a single environment configured for evaluation.
    """
    raw_env = retro.make(
        'SuperMarioWorld-Snes',
        state=level,
        render_mode='rgb_array',
    )
    return _wrap_env(
        raw_env,
        record=record_dir is not None,
        record_dir=record_dir,
    )


# Legacy aliases for backwards compatibility
def make_training_env(num_envs=1, **level_kwargs):
    if level_kwargs:
        level_weights = {v: 1.0 for v in level_kwargs.values()}
    else:
        level_weights = None
    
    return make_env(
        num_envs=num_envs,
        level_weights=level_weights,
        render_human=(num_envs == 1),
    )


def make_curriculum_env(num_envs: int, level_distribution: list):
    return make_env(
        num_envs=num_envs,
        level_distribution=level_distribution,
        render_human=(num_envs == 1),
    )


prepare_env = _wrap_env
