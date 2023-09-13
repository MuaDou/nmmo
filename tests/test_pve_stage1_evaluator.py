import os
import json
import subprocess

from fixture import py_path, this_dir, pve_stage1_setup  # noqa


def test_pve_stage1_evaluator(pve_stage1_setup):
    from evaluator import AIcrowdEvaluator, Constants, Mode
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
            for filename in os.listdir(f"{Constants.SHARED_DIR}/results"):
                if filename.startswith("result-"):
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