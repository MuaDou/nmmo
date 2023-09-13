import os
import json
import subprocess

from fixture import py_path, this_dir, pve_stage1_setup  # noqa


def test_aicrowd_evaluator(pve_stage1_setup):
    from evaluator import AIcrowdEvaluator, Constants
    try:
        for _ in range(1):
            os.environ["AICROWD_SUBMISSION_ID"] = "Random"
            os.environ["AICROWD_SUBMISSION_PATH"] = os.path.join(
                this_dir, "submissions", "random")
            p = subprocess.Popen(
                f"python {py_path}",
                env=dict(os.environ),
                shell=True,
            )

            evaluator = AIcrowdEvaluator()
            evaluator.serve()
            p.wait()

            exist = False
            for filename in os.listdir(Constants.SHARED_DIR):
                if filename.startswith("result-"):
                    exist = True
                    break
            assert exist

        evaluator = AIcrowdEvaluator()
        print(evaluator.evaluate())

        exist = False
        for filename in os.listdir(Constants.SHARED_DIR):
            if filename == "result.json":
                exist = True
                break
        assert exist

        with open(os.path.join(Constants.SHARED_DIR, "result.json"),
                  "r") as fp:
            print(json.load(fp))
    except:
        p.kill()
        raise


def test_aicrowd_evaluator_pve_end_to_end(pve_stage1_setup):
    from evaluator import AIcrowdEvaluator, Constants, Mode
    try:
        for mode in [
                Mode.PVE_STAGE1, Mode.PVE_STAGE2, Mode.PVE_STAGE3,
                Mode.PVE_BONUS
        ]:
            os.environ["AICROWD_SUBMISSION_ID"] = "MyCombat"
            os.environ["AICROWD_SUBMISSION_PATH"] = os.path.join(
                this_dir, "submissions", "combat")
            os.environ["NMMO_MODE"] = str(mode)

            # need to hard code here, since Constants is already imported
            Constants.MODE = mode

            p = subprocess.Popen(
                f"python {py_path}",
                env=dict(os.environ),
                shell=True,
            )

            evaluator = AIcrowdEvaluator()
            evaluator.serve()
            p.wait()

            exist = False
            for filename in os.listdir(f"{Constants.SHARED_DIR}/results"):
                if filename.startswith(f"result-{mode}-"):
                    exist = True
                    break
            assert exist

            exist = False
            for filename in os.listdir(f"{Constants.SHARED_DIR}/replays"):
                print(filename, f"replay-{mode}-")
                if filename.startswith(f"replay-{mode}"):
                    exist = True
                    break
            assert exist

        Constants.MODE = Mode.PVE_AGGREGATE
        evaluator = AIcrowdEvaluator()
        print(evaluator.evaluate())

        exist = False
        for filename in os.listdir(Constants.SHARED_DIR):
            if filename == "result.json":
                exist = True
                break
        assert exist

        with open(os.path.join(Constants.SHARED_DIR, "result.json"),
                  "r") as fp:
            print(json.load(fp))
    except:
        p.kill()
        raise


def test_aicrowd_evaluator_pve_quick_return(pve_stage1_setup):
    from evaluator import AIcrowdEvaluator, Constants, Mode, PVEEvaluator
    try:
        for mode in [
                Mode.PVE_STAGE1, Mode.PVE_STAGE2, Mode.PVE_STAGE3,
                Mode.PVE_BONUS
        ]:
            os.environ["AICROWD_SUBMISSION_ID"] = "TheRandomNoob"
            os.environ["AICROWD_SUBMISSION_PATH"] = os.path.join(
                this_dir, "submissions", "random")
            os.environ["NMMO_MODE"] = str(mode)

            # need to hard code here, since Constants is already imported
            Constants.MODE = mode

            p = subprocess.Popen(
                f"python {py_path}",
                env=dict(os.environ),
                shell=True,
            )

            PVEEvaluator.pass_values[Mode.PVE_STAGE1] = 1.0
            evaluator = AIcrowdEvaluator()
            evaluator.serve()
            p.wait()

        Constants.MODE = Mode.PVE_AGGREGATE
        evaluator = AIcrowdEvaluator()
        print(evaluator.evaluate())

        exist = False
        for filename in os.listdir(Constants.SHARED_DIR):
            if filename == "result.json":
                exist = True
                break
        assert exist

        with open(os.path.join(Constants.SHARED_DIR, "result.json"),
                  "r") as fp:
            print(json.load(fp))
    except:
        p.kill()
        raise


# if __name__ == "__main__":
#     pve_stage1_setup()
#     test_aicrowd_evaluator()
