import sys
import os

from torch import R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import lz4.block
import pickle
import random
import lzma
import json
import requests
from loguru import logger
from aicrowd_api import API, AIcrowdSubmission
import multiprocessing as mp

from pkg import cos, util
from nmmo import Replay

AICROWD_API_TOKEN = os.environ["AICROWD_API_TOKEN"]
N = 5

api = API(AICROWD_API_TOKEN)
grader_id = "3380"

players = {"week1": {}, "week2": {}, "week3": {}, "week4": {}, "week5": {}}

def pre_submission(submission_id):
    subm = api.get_submission(grader_id, submission_id).raw_response

    tournaments = []
    for key in subm["meta"]:
        if key.startswith("Replay-PVP-week"):
            tournament = key.split("-")[-1]
            tournaments.append(tournament)
            players[tournament][submission_id] = subm["participant_name"]

    if not tournaments:
        return

    for tournament in tournaments:
        replay_dir = f"replays/pvp/{tournament}"
        os.system(f"mkdir -p {replay_dir}")
        zippath = f"{replay_dir}/pvp.zip"
        if os.path.exists(zippath):
            continue

        with open(zippath, "wb") as fp:
            content = requests.get(subm["meta"][f"Replay-PVP-{tournament}"], stream=True).content
            fp.write(content)
        os.system(f"unzip {zippath} -d {replay_dir}")


submission_ids = api.get_all_submissions(grader_id)
for i in submission_ids:
    pre_submission(i)

def deal_tournament(tournament, players, api_token):
    replay_dir = f"replays/pvp/{tournament}"
    replays = glob.glob(f"{replay_dir}/*.replay")

    # replay to lzma
    for old in replays:
        logger.info(f"{old} replay to lzma")
        with open(old, "rb") as fp:
            data = fp.read()
        data = lz4.block.decompress(data)
        data = pickle.loads(data)
        data = json.dumps(data).encode("utf-8")
        data = lzma.compress(data, format=lzma.FORMAT_ALONE)
        new = os.path.join(
            os.path.dirname(old),
            "replay-" + os.path.basename(old).replace(".replay", ".lzma"))
        with open(new, "wb") as fp:
            fp.write(data)

    # submission to name
    replays = glob.glob(f"{replay_dir}/*.lzma")
    for path in replays:
        logger.info(f"{path} rename")
        replay = Replay.load(path)
        replay = util.submission_id_to_name(replay, players)
        replay.path = path
        replay.save()

    upload_ret = cos.upload_pvp_replays(tournament, replay_dir, list(players.keys()))
    for submission_id in upload_ret["submissions"]:
        subm: AIcrowdSubmission = api.get_submission(grader_id, submission_id)
        subm.api_key = api_token
        subm.message = subm.raw_response["grading_message"]
        subm.meta[f"Replay-PVP-{tournament}"] = upload_ret["submissions"][submission_id]
        logger.info(f"update meta {submission_id}")
        subm.update(meta_overwrite=True)


q = mp.Queue()
for i in players:
    q.put(i)


def do(q: mp.Queue):
    while 1:
        try:
            tournament = q.get_nowait()
            logger.info(f"deal tournament {tournament}")
            try:
                deal_tournament(tournament, players[tournament], AICROWD_API_TOKEN)
            except:
                logger.exception(f"{tournament} failed")
        except:
            break


ps = []
for _ in range(N):
    p = mp.Process(target=do, args=(q, ), daemon=True)
    p.start()
    ps.append(p)

for p in ps:
    p.join()
