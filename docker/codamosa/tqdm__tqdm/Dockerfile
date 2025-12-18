FROM aorwall/swe-bench-conda:bookworm-slim

WORKDIR /opt

USER root

RUN git clone https://github.com/tqdm/tqdm.git

WORKDIR /opt/tqdm

RUN git -c advice.detachedHead=false checkout 18d7aa4

RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir pytest pytest-cov coverage

RUN git checkout 18d7aa4 .

ENV TESTBED_NAME=tqdm__tqdm__18d7aa4
ENV IMAGE_TYPE=conda
ENV REPO_DIR=/opt/tqdm

COPY swebench_docker swebench_docker
COPY --chmod=0755 docker/entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]
