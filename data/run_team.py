import os
import pickle
from loguru import logger
from neurips2022nmmo import submission as subm
from neurips2022nmmo import CompetitionConfig
from aicrowd_gym.serializers.base import BaseSerializer
from aicrowd_gym.clients.zmq_oracle_client import ZmqOracleClient


class PickleSerializer(BaseSerializer):

    def __init__(self):
        self.content_type = "application/octet-stream"

    def raw_encode(self, data):
        return pickle.dumps(data)

    def raw_decode(self, data):
        return pickle.loads(data)


class Constants:
    SERVER_HOST = os.getenv("AICROWD_REMOTE_SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("AICROWD_REMOTE_SERVER_PORT", "5000"))
    SUBMISSION_ID = os.getenv("AICROWD_SUBMISSION_ID", "00000")
    SUBMISSION_PATH = os.getenv("AICROWD_SUBMISSION_PATH",
                                "submissions/random")
    assert SUBMISSION_PATH
    MODE: str = os.getenv("NMMO_MODE")


def main():
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    if Constants.MODE in [
            "PVE_STAGE1_VALIDATION", "PVE_STAGE2_VALIDATION",
            "PVE_BONUS_VALIDATION"
    ]:
        logger.info("validation, return")
        return

    subm.check(Constants.SUBMISSION_PATH)
    logger.info("Check submission done")

    team = subm.get_team_from_submission(Constants.SUBMISSION_PATH,
                                         Constants.SUBMISSION_ID,
                                         CompetitionConfig())
    logger.info(f"Get team {team.__class__.__name__} done")

    client = ZmqOracleClient(
        host=Constants.SERVER_HOST,
        port=Constants.SERVER_PORT,
        serializer=PickleSerializer(),
    )
    client.register_agent(agent=team,
                          metadata={
                              "team_id": team.id,
                              "policy_id": Constants.SUBMISSION_ID,
                              "submission_id": Constants.SUBMISSION_ID,
                              "is_user": True,
                          })
    logger.info("Register agent done")

    logger.info("Run team")
    client.run_agent()

    logger.info("Close, bye.")


if __name__ == "__main__":
    main()
