FROM python:3.9.11

RUN echo """deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye main contrib non-free \
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye-updates main contrib non-free \
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye-backports main contrib non-free \
deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bullseye-security main contrib non-free""" > /etc/apt/sources.list

RUN apt update && \
    apt install -y zip

ARG PYPI=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /home/aicrowd

RUN pip install --no-cache-dir numpy -i ${PYPI}

RUN pip install torch==1.10.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements_base.txt requirements_base.txt

RUN pip install -r requirements_base.txt --no-cache-dir -i ${PYPI}

COPY requirements_nmmo.txt requirements_nmmo.txt

RUN pip install -r requirements_nmmo.txt --no-cache-dir -i ${PYPI}

COPY . .

CMD ["bash"]

