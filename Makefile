ver ?= test

.PHONY : format

format:
	@autoflake --in-place --remove-all-unused-imports -r evaluator.py pkg data tests
	@yapf -i -r evaluator.py pkg data tests

test_pve:
	@PYTHONPATH=$PYTHONPATH:`pwd` pytest -s -v tests/test_pve_stage1_evaluator.py

test_pvp:
	@PYTHONPATH=$PYTHONPATH:`pwd` pytest -s -v tests/test_aicrowd_evaluator_pvp.py

evaluator:
	@docker build -t hkccr.ccs.tencentyun.com/neurips2022nmmo/evaluator:$(ver) -f pvp/Dockerfile.pvp .
	@docker push hkccr.ccs.tencentyun.com/neurips2022nmmo/evaluator:$(ver)

aicrowd_gym:
	@docker build -t hkccr.ccs.tencentyun.com/neurips2022nmmo/aicrowd_gym:latest -f pvp/Dockerfile.aicrowd_gym .
	@docker push hkccr.ccs.tencentyun.com/neurips2022nmmo/aicrowd_gym:latest

