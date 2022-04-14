from setuptools import setup

setup(name='maps',
      version='0.0.1',
      packages=['maps', 'sawyer'],
      install_requires=['mujoco-py==1.50.1.68', 'gym==0.12.0', 'baselines', 'gym_extensions', 'metaworld']
      )
