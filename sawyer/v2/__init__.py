from gym.envs.registration import register

register(
    id='reach-v2',
    entry_point='sawyer.v2.envs:SawyerReachEnvV2',
    timestep_limit=150,
    reward_threshold=5e50,
    nondeterministic=True,
    # kwargs={'random_init': True}
)

register(
    id='push-v2',
    entry_point='sawyer.v2.envs:SawyerPushEnvV2',
    timestep_limit=150,
    reward_threshold=5e50,
    nondeterministic=True,
    # kwargs={
    #         'random_init': True}
)

register(
    id='pick-place-v2',
    entry_point='sawyer.v2.envs:SawyerPickPlaceEnvV2',
    timestep_limit=150,
    reward_threshold=5e50,
    nondeterministic=True,
    # kwargs={
    #         'random_init': True}
)

register(
    id='door-open-v2',
    entry_point='sawyer.v2.envs:SawyerDoorEnvV2',
    timestep_limit=150,
    reward_threshold=5e50,
    nondeterministic=True,
    # kwargs={'random_init': True}
)

register(
    id='drawer-close-v2',
    entry_point='sawyer.v2.envs:SawyerDrawerCloseEnvV2',
    timestep_limit=150,
    reward_threshold=5e50,
    nondeterministic=True,
    # kwargs={'random_init': True}
)

register(
    id='drawer-open-v2',
    entry_point='sawyer.v2.envs:SawyerDrawerOpenEnvV2',
    timestep_limit=150,
    reward_threshold=5e50,
    nondeterministic=True,
    # kwargs={'random_init': True}
)

register(
    id='button-press-topdown-v2',
    entry_point='sawyer.v2.envs:SawyerButtonPressTopdownEnvV2',
    timestep_limit=150,
    reward_threshold=5e50,
    nondeterministic=True,
    # kwargs={'random_init': True}
)

register(
    id='peg-insert-side-v2',
    entry_point='sawyer.v2.envs:SawyerPegInsertionSideEnvV2',
    timestep_limit=150,
    reward_threshold=5e50,
    nondeterministic=True,
    # kwargs={'random_init': True}
)

register(
    id='window-open-v2',
    entry_point='sawyer.v2.envs:SawyerWindowOpenEnvV2',
    timestep_limit=150,
    reward_threshold=5e50,
    nondeterministic=True,
    # kwargs={'random_init': True}
)

register(
    id='window-close-v2',
    entry_point='sawyer.v2.envs:SawyerWindowCloseEnvV2',
    timestep_limit=150,
    reward_threshold=5e50,
    nondeterministic=True,
    # kwargs={'random_init': True}
)