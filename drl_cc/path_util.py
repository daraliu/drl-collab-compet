import pathlib

from drl_cc import config as cfg


def mk_path_weights_actor_local(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_WEIGHTS_ACTOR)


def mk_path_weights_actor_target(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_WEIGHTS_ACTOR_TARGET)


def mk_path_weights_critic_local(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_WEIGHTS_CRITIC)


def mk_path_weights_critic_target(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_WEIGHTS_CRITIC_TARGET)


def mk_path_scores(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_SCORES)


def mk_path_metadata(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_METADATA)
