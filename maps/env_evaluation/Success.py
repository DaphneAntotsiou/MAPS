def success(env, reward):
    if env.find("CheetahBig") != -1 or env.find("CheetahSmall") != -1:
        from maps.env_evaluation.HalfCheetahModified import success
    elif 'Cheetah' in env or 'CheetahGravity' in env:
        from maps.env_evaluation.HalfCheetahGravity import success
    return success(env, reward)
