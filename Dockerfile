FROM neurips2022nmmo/evaluator:latest

WORKDIR /home/aicrowd

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

CMD ["bash"]

