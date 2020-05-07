from setuptools import setup, find_packages


setup(
    name="drl-collab-compet",
    version="0.1.0",
    packages=find_packages(),

    install_requires=[
        "numpy",
        "pandas",
        "plotnine>=0.2.0",
        "click>=7.0.0",
    ],
    author="Darius Aliulis",
    author_email="darius.aliulis@gmail.com",
    description="Deep Reinforcement Learning for Collaboration and Competition",
    url="https://github.com/daraliu/drl-collab-compet",
    entry_points='''
        [console_scripts]
        drl-cc=drl_cc.cli:cli
    '''
)
