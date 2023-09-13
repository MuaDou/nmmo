REPO ?= neurips2022nmmo/evaluator

TAG ?= $(shell date +%Y%m%d)-$(shell git describe --match=$(git rev-parse --short=8 HEAD) --tags --always --dirty)

.PHONY : image push

image:
	@docker build -t ${REPO}:${TAG} -f Dockerfile .
	@docker tag ${REPO}:${TAG} ${REPO}:latest

push: image
	@docker push ${REPO}:${TAG}
	@docker push ${REPO}:latest
