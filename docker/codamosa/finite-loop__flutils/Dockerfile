FROM aorwall/swe-bench-conda:bookworm-slim

RUN mkdir -p /testbed

WORKDIR /testbed

RUN git clone https://gitlab.com/finite-loop/flutils.git

WORKDIR /testbed/flutils

RUN git -c advice.detachedHead=false checkout df0f84e1

RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir pytest pytest-cov coverage

RUN git checkout df0f84e1 .

ENV TESTBED_NAME=finite-loop__flutils__df0f84e1
ENV IMAGE_TYPE=conda
ENV REPO_DIR=/testbed/flutils

WORKDIR /testbed
COPY swebench_docker swebench_docker
COPY --chmod=0755 docker/entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]
