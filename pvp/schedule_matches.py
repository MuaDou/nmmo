import json
from evaluator import AIcrowdEvaluator

matches = AIcrowdEvaluator.schedule_matches()
with open("/tmp/next-matches", "w") as fp:
    fp.write(json.dumps(matches))

