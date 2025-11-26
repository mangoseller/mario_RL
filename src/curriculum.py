"""
Curriculum learning schedule for progressive training difficulty.

Schedule:
- Phase 0 (0-20%): 100% level1
- Phase 1 (20-40%): 70% level1, 30% level2  
- Phase 2 (40-60%): 40% level1, 40% level2, 20% level3
- Phase 3 (60-100%): 10% level1, 20% level2, 30% level3, 40% level4
"""

CURRICULUM_PHASES = [
    {
        'end_progress': 0.2,
        'distribution': {'level1': 1.0}
    },
    {
        'end_progress': 0.4,
        'distribution': {'level1': 0.7, 'level2': 0.3}
    },
    {
        'end_progress': 0.6,
        'distribution': {'level1': 0.4, 'level2': 0.4, 'level3': 0.2}
    },
    {
        'end_progress': 1.0,
        'distribution': {'level1': 0.1, 'level2': 0.2, 'level3': 0.3, 'level4': 0.4}
    },
]

CURRICULUM_LEVELS = {
    'level1': 'YoshiIsland2',
    'level2': 'YoshiIsland3',
    'level3': 'DonutPlains1',
    'level4': 'DonutPlains4',
}


def get_curriculum_phase(step, total_steps):
    """Get the current curriculum phase index based on training progress."""
    if total_steps == 0:
        return 0
    progress = step / total_steps
    for i, phase in enumerate(CURRICULUM_PHASES):
        if progress < phase['end_progress']:
            return i
    return len(CURRICULUM_PHASES) - 1


def get_phase_distribution(phase_idx, num_envs):
    """Convert phase distribution percentages to actual env counts and return level list."""
    phase = CURRICULUM_PHASES[phase_idx]
    distribution = phase['distribution']
    
    # Calculate counts for each level
    level_counts = {}
    remaining = num_envs
    levels = list(distribution.keys())
    
    # Distribute envs proportionally
    for i, level in enumerate(levels[:-1]):
        count = int(round(distribution[level] * num_envs))
        level_counts[level] = min(count, remaining)
        remaining -= level_counts[level]
    
    # Last level gets remaining envs to ensure we hit num_envs exactly
    level_counts[levels[-1]] = remaining
    
    # Build the distribution list
    result = []
    for level_key, count in level_counts.items():
        level_name = CURRICULUM_LEVELS[level_key]
        result.extend([level_name] * count)
    
    return result


def should_change_phase(step, total_steps, current_phase):
    """Check if training should transition to a new phase."""
    new_phase = get_curriculum_phase(step, total_steps)
    return new_phase != current_phase


def get_phase_description(phase_idx):
    """Get a human-readable description of the phase distribution."""
    phase = CURRICULUM_PHASES[phase_idx]
    parts = []
    for level_key, pct in phase['distribution'].items():
        level_name = CURRICULUM_LEVELS[level_key]
        parts.append(f"{int(pct*100)}% {level_name}")
    return f"Phase {phase_idx}: " + ", ".join(parts)
