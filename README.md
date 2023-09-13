# NIPS2022 NMMO EVALUATION

## Dev Setup

```bash
$ pip install -r requirements.txt
```

## Run Test

```bash
$ make test
```

The test is currently running PVE stage1, which 15 scripted teams and 1 submission team are involved.

The outputs are located at ``tests/shared``.

* The ``metrics-*.json`` is the output of ``AIcrowdEvaluator.serve()``.
* The ``result.json`` is the output of ``AIcrowdEvaluator.evaluate()``.
