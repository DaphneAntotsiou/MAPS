def success(env, reward):
    ENV_REWARDS = {
        'HalfCheetah-v2': 4315,
        'HalfCheetahSmallTorso-v0': 10546,
        'HalfCheetahSmallHead-v0': 11465,
        "HalfCheetahSmallFoot-v0": 3632,  # 1816,
        "HalfCheetahSmallLeg-v0": 7229,
        "HalfCheetahSmallThigh-v0": 10869,
        "HalfCheetahBigTorso-v0": 7784,
        "HalfCheetahBigFoot-v0": 11668,
        "HalfCheetahBigLeg-v0": 7943,
        "HalfCheetahBigHead-v0": 8465,
        "HalfCheetahBigThigh-v0": 11379,
    }
    return max(float(reward / ENV_REWARDS[env]), 0) if env in ENV_REWARDS else None
