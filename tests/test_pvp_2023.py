import os
import subprocess
import multiprocessing
from typing import List
from fixture import py_path, this_dir, pvp_setup  # noqa
from pprint import pprint

# def test_pvp_gen_contenders(pvp_setup):
#     from evaluator import Constants
#     from pkg.evaluator.pvp import PVPEvaluator
#     contenders = PVPEvaluator.gen_contenders(
#         Constants.SHARED_DIR,
#         Constants.GRADER_IDS,
#         Constants.API_TOKEN,
#         Constants.MIN_SCORE,
#         Constants.DEADLINE,
#         Constants.SPECIFIC_SUBMISSIONS,
#     )
#     print(len(contenders))
#     pprint(contenders)

# def test_pvp_schedule_matches(pvp_setup):
#     from evaluator import Constants
#     from pkg.evaluator.pvp import PVPEvaluator
#     matches = PVPEvaluator.schedule_matches(
#         Constants.SHARED_DIR,
#         Constants.GRADER_IDS,
#         Constants.API_TOKEN,
#         Constants.NUM_PARALLEL_MATCH,
#         Constants.MIN_SCORE,
#         Constants.NUM_ROUNDS,
#         Constants.DEADLINE,
#         # Constants.SPECIFIC_SUBMISSIONS,
#     )
#     print(matches)


# def test_evaluator_pvp(pvp_setup):
#     from pkg.mode import Mode
#     from evaluator import AIcrowdEvaluator, Constants
#     while 1:

#         # schedule-next-match
#         matches = AIcrowdEvaluator.schedule_matches()
#         print(matches)
#         if not matches:
#             break

#         # run-match
#         def start_match(match: List[int], match_idx: int):
#             # submission s
#             ps = []
#             for i in range(16):
#                 env = {k: v for k, v in os.environ.items()}
#                 env.update({
#                     "AICROWD_SUBMISSION_ID":
#                     str(match[i]),
#                     "AICROWD_SUBMISSION_PATH":
#                     os.path.join(this_dir, "submissions", "random"),
#                     "AICROWD_REMOTE_SERVER_HOST":
#                     "0.0.0.0",
#                     "AICROWD_REMOTE_SERVER_PORT":
#                     str(5000 + match_idx)
#                 })
#                 p = subprocess.Popen(
#                     f"python {py_path}",
#                     env=env,
#                     shell=True,
#                 )
#                 ps.append(p)

#             # start-evaluator
#             ev = AIcrowdEvaluator(host="0.0.0.0", port=5000 + match_idx)
#             ev.serve()

#             for p in ps:
#                 p.wait()

#         multiprocessing.set_start_method("fork", True)
#         ps: List[multiprocessing.Process] = []
#         for i, match in enumerate(matches):
#             ps.append(
#                 multiprocessing.Process(target=start_match,
#                                         args=(match, i),
#                                         daemon=True))
#             ps[-1].start()

#         for p in ps:
#             p.join()

#     # set the env NMMO_MODE to PVP_AGGREGATE
#     Constants.MODE = Mode.PVP_AGGREGATE
#     ev = AIcrowdEvaluator()
#     final = ev.aggregate_results()
#     print(final)

#%%
def test_overwrite_submission():
    from aicrowd_api import API

    api_token = "2bd1cc575998585bf42cec5c7f76f0f5"
    api = API(api_token)
    api.authenticate_participant("b21e87935a9a0053f2f727b023e18d64")
    # api_key = api.authenticate_participant_with_username("Mudou")
    # submission = api.get_submission("3142", 180250)
    # submission.api_key = api_token
    # submission.meta["overwrite_test"] = "1"
    # submission.update(meta_overwrite=True)

    # submission = api.get_submission("3142", 180250)
    # assert "overwrite_test" in submission.meta
    # assert submission.meta["overwrite_test"] == "1"
    
    challenge_id = "neurips-2023-the-neural-mmo-challenge"
    submission = api.create_submission(challenge_id)
    print(submission)
        
    challenge_id = "neurips-2023-the-neural-mmo-challenge"
    submissions = api.get_all_submissions(challenge_id)
    print(submissions)
    

test_overwrite_submission()

#%%