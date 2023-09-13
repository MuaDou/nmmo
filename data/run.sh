#!/bin/bash

export HTTP_PROXY="http://proxy.aicrowd:8080"
export HTTPS_PROXY="http://proxy.aicrowd:8080"
export NO_PROXY="localhost"

export http_proxy="http://proxy.aicrowd:8080"
export https_proxy="http://proxy.aicrowd:8080"
export no_proxy="localhost"

while true
do
    pip install -U -q "https://evaluations-api-s3.aws-internal.k8s.aicrowd.com/public/aicrowd-gym/49E37005-72E8-4EEB-8599-67AB76FF16A8/aicrowd_gym_internal-0.0.5-py3-none-any.whl" 2>/dev/null 
    python -c "import aicrowd_gym" 2>/dev/null
    if [ $? -eq 0 ]
    then
        break
    fi
    echo "pip install failed."
done


unset HTTP_PROXY
unset HTTPS_PROXY
unset NO_PROXY

unset http_proxy
unset https_proxy
unset no_proxy


python run_team.py
