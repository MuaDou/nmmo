import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import lz4.block
import pickle
import lzma
import json
import requests
from loguru import logger
from aicrowd_api import API, AIcrowdSubmission
import multiprocessing as mp

from pkg import cos

AICROWD_API_TOKEN = os.environ["AICROWD_API_TOKEN"]
N = 30

api = API(AICROWD_API_TOKEN)
grader_id = "3380"


def deal_submission(submission_id, api_token):
    replay_dir = f"replays/{submission_id}"
    os.system(f"mkdir -p {replay_dir}")

    subm: AIcrowdSubmission = api.get_submission(grader_id, submission_id)

    # pve replays
    if "Replay" not in subm.meta:
        logger.warning(f"{submission_id} do not have pve replay")
        return
    zippath = f"{replay_dir}/pve.zip"

    with open(zippath, "wb") as fp:
        content = requests.get(subm.meta["Replay"], stream=True).content
        fp.write(content)
    os.system(f"unzip {zippath} -d {replay_dir}")
    olds = glob.glob(f"{replay_dir}/*.replay")
    for old in olds:
        logger.info(f"{old} replay to lzma")
        with open(old, "rb") as fp:
            data = fp.read()
        data = lz4.block.decompress(data)
        data = pickle.loads(data)
        data = json.dumps(data).encode("utf-8")
        data = lzma.compress(data, format=lzma.FORMAT_ALONE)
        new = os.path.join(os.path.dirname(old), "replay-" + os.path.basename(old).replace(".replay", ".lzma"))
        with open(new, "wb") as fp:
            fp.write(data)

    logger.info("upload pve replays")
    upload_ret = cos.upload_pve_replays(submission_id, replay_dir)

    # overwrite
    subm.api_key = api_token
    subm.message = subm.raw_response["grading_message"]
    if "zip" in upload_ret:
        subm.meta["Replay"] = upload_ret["zip"]
    else:
        logger.warning(f"no zip for {submission_id}")
    for mode in upload_ret["modes"]:
        subm.meta[f"Replay-{mode}"] = upload_ret["modes"].get(mode, "none")

    logger.info("upload meta")
    subm.update(meta_overwrite=True)


q = mp.Queue()
submission_ids = api.get_all_submissions(grader_id)
for i in submission_ids:
    q.put(i)

def do(q: mp.Queue):
    while 1:
        try:
            submission_id = q.get_nowait()
            logger.info(f"deal submission {submission_id}")
            try:
                deal_submission(submission_id, AICROWD_API_TOKEN)
            except:
                logger.exception(f"{submission_id} failed")
        except:
            break

ps = []
for _ in range(N):
    p = mp.Process(target=do, args=(q,), daemon=True)
    p.start()
    ps.append(p)

for p in ps:
    p.join()
