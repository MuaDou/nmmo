import os
import subprocess
from copy import deepcopy
from threading import Thread
import json

from ruamel.yaml import YAML

from fixture import py_path, this_dir, pve_stage1_setup  # noqa

yaml = YAML(typ="safe")

INITIAL_ENV = deepcopy(os.environ)
INITIAL_ENV["LOGURU_LEVEL"] = "INFO"
AICROWD_CONFIG_FILE = os.path.join(os.path.dirname(__file__),
                                   "../aicrowd.yaml")
AICROWD_CONFIG = yaml.load(open(AICROWD_CONFIG_FILE).read())


def load_run_config(run_name: str):
    run_cfg = None

    for cfg in AICROWD_CONFIG["evaluation"]["runs"]:
        if cfg["name"] == run_name:
            run_cfg = cfg
            break

    assert run_cfg, "No such run found, please check evaluation.runs in aicrowd.yaml"
    return run_cfg


def load_env_vars(stage: str):
    env = deepcopy(INITIAL_ENV)
    run_cfg = load_run_config(stage)
    env.update(run_cfg["env"])
    env.update(run_cfg.get("gym_server", {}).get("env", {}))
    return env


def execute_runs():
    for cfg in AICROWD_CONFIG["evaluation"]["runs"]:
        # setup env
        env = deepcopy(INITIAL_ENV)
        env.update(cfg.get("env", {}))
        env.update(cfg.get("gym_server", {}).get("env", {}))
        os.environ = env

        # start server
        from evaluator import AIcrowdEvaluator, Constants

        evaluator = AIcrowdEvaluator()
        server_thread = Thread(target=evaluator.serve)
        server_thread.start()

        # start participant's code
        os.environ["AICROWD_SUBMISSION_ID"] = "TheRandomNoob"
        os.environ["AICROWD_SUBMISSION_PATH"] = os.path.join(
            this_dir, "submissions", "random")
        p = subprocess.Popen(
            f"python {py_path}",
            env=dict(os.environ),
            shell=True,
        )
        p.wait()

        exist = False
        for filename in os.listdir(Constants.SHARED_DIR):
            if filename.startswith("metrics-"):
                exist = True
                break
        assert exist

        p.kill()


def execute_scoring():
    os.environ = deepcopy(INITIAL_ENV)
    os.environ.update(AICROWD_CONFIG["evaluation"]["scoring"].get("env", {}))

    from evaluator import AIcrowdEvaluator

    evaluator = AIcrowdEvaluator()
    scores = evaluator.evaluate()

    # make sure scores are JSON serializable
    print(json.loads(json.dumps(scores)))


def test_evaluator():
    execute_runs()
    execute_scoring()
