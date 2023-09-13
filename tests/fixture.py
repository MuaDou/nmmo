import os
import pytest
import shutil

this_dir = os.path.dirname(os.path.abspath(__file__))

py_path = py_path = os.path.join(os.path.dirname(this_dir), "data",
                                 "run_team.py")


@pytest.fixture(scope="package", autouse=True)
def setup():
    shared_dir = os.path.join(this_dir, "shared")
    shutil.rmtree(shared_dir, ignore_errors=True)
    os.makedirs(shared_dir, exist_ok=True)

    kvs = {
        "AICROWD_REMOTE_SERVER_HOST": "127.0.0.1",
        "AICROWD_REMOTE_SERVER_PORT": "5000",
        "AICROWD_SHARED_DIR": shared_dir,
        "AICROWD_API_TOKEN": "1111111111",
        "META": "{}",
        "NMMO_MODE": "",
        "NMMO_MAX_EPISODES": "1",
        "NMMO_MAX_STEPS_PER_EPISODE": "40",
        "NMMO_LOCAL": "1",
        "NMMO_SAVE_REPLAY": "1",
    }
    for k, v in kvs.items():
        if k not in os.environ:
            os.environ[k] = v


@pytest.fixture(scope="package", autouse=True)
def pve_stage1_setup():
    kvs = {
        "AICROWD_SUBMISSION_ID": "Melee",
        "AICROWD_SUBMISSION_PATH": os.path.join(this_dir, "submissions",
                                                "melee"),
        "NMMO_MODE": "PVE_STAGE1"
    }

    for k, v in kvs.items():
        os.environ[k] = v


@pytest.fixture(scope="package", autouse=True)
def pvp_setup():
    kvs = {
        "NMMO_MODE": "PVP",
        "AICROWD_GRADER_IDS": "3778",
        "AICROWD_API_TOKEN": "2bd1cc575998585bf42cec5c7f76f0f5",
        "NMMO_NUM_PARALLEL_MATCH": "2",
        "NMMO_NUM_ROUNDS": "1",
        "NMMO_MIN_SCORE": "0.01",
        # 2022-11-01 23:59:59 Los_Angeles / 2022-11-02 14:59:59 Shanghai
        "NMMO_DEADLINE": "1667372399",
        "NMMO_MAX_STEPS_PER_EPISODE": "20",
        "NMMO_UPLOAD_REPLAY": "0",
        "NMMO_PVP_AGGREGATE_DRY_RUN": "1",
        # "NMMO_SPECIFIC_SUBMISSIONS": "205360,205308,205323,205343,203451,204709,205351,204437,205317,205379,200862,203443,205222,204419,199331,198822"
    }

    for k, v in kvs.items():
        os.environ[k] = v
