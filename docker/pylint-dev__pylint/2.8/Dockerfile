# Generated by generate_dockerfiles.py

FROM aorwall/swe-bench-pylint-dev_pylint:bookworm-slim

RUN git -c advice.detachedHead=false checkout 49a6206c7756307844c1c32c256afdf9836d7bce

RUN conda create -n pylint-dev__pylint__2.8 python=3.9 -y
SHELL ["conda", "run", "-n", "pylint-dev__pylint__2.8", "/bin/bash", "-c"]

COPY docker/pylint-dev__pylint/2.8/requirements.txt .

RUN pip install -r requirements.txt

RUN pip install pytest
RUN pip install coverage cosmic-ray

RUN pip install -e .


RUN git checkout 49a6206c7756307844c1c32c256afdf9836d7bce .

ENV TESTBED_NAME=pylint-dev__pylint__2.8
ENV IMAGE_TYPE=conda

WORKDIR /home/swe-bench

COPY swebench_docker swebench_docker

ENTRYPOINT ["./entrypoint.sh"]
