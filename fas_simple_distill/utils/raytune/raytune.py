import ray
import yaml
from addict import Dict as Adict
from ray import tune
from ray.tune import ExperimentAnalysis, schedulers
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from iglovikov_helper_functions.config_parsing.utils import object_from_dict

from ray.tune.trial import Trial


def trial_name_creator_v2(run_name, *_):
    identifier = run_name
    identifier.replace("/", "_")
    return identifier


def trial_name_creator(run_name, trial: Trial):
    env_name = 0
    if "env" in trial.config:
        env_name = trial.config["env"]
        if isinstance(env_name, type):
            env_name = env_name.__name__

    identifier = f"{run_name}__{env_name}__{trial.trial_id}"
    identifier.replace("/", "_")
    return identifier


def parse_tuning_config(config_path: str):
    with open(config_path) as f:
        tuning_config = Adict(yaml.load(f, Loader=yaml.SafeLoader))
    tuning_config.freeze()

    points_to_evaluate = tuning_config.points_to_evaluate
    search_space = {k: object_from_dict(v) for k, v in tuning_config.search_space.items()}

    algo = object_from_dict(tuning_config.algo, points_to_evaluate=points_to_evaluate)
    try:
        algo = ConcurrencyLimiter(algo, max_concurrent=tuning_config.max_concurrent)
    except RuntimeError:
        pass

    scheduler_config = tuning_config.scheduler
    scheduler = None
    if scheduler_config:
        scheduler = object_from_dict(tuning_config.scheduler)

    ret = {
        "search_space": search_space,
        "algo": algo,
        "scheduler": scheduler,
        "tuning_config": tuning_config,
    }

    return ret
