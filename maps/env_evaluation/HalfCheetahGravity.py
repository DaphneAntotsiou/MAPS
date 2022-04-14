def success(env, reward):
    ENV_REWARDS = {
        'HalfCheetah-v2': 4315,
        'HalfCheetahGravityHalf-v0': 12731,
        'HalfCheetahGravityOneAndHalf-v0': 5910,
        "HalfCheetahGravityOneAndQuarter-v0": 7892,
        "HalfCheetahGravityThreeQuarters-v0": 7386
    }
    return max(float(reward / ENV_REWARDS[env]), 0) if env in ENV_REWARDS else None
