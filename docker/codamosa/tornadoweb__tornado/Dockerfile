FROM aorwall/swe-bench-conda:bookworm-slim

RUN mkdir -p /testbed

WORKDIR /testbed

RUN git clone https://github.com/tornadoweb/tornado.git

WORKDIR /testbed/tornado

RUN git -c advice.detachedHead=false checkout 2047e7a

RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir pytest pytest-cov coverage

RUN git checkout 2047e7a .

ENV TESTBED_NAME=tornadoweb__tornado__2047e7a
ENV IMAGE_TYPE=conda
ENV REPO_DIR=/testbed/tornado

WORKDIR /testbed
COPY swebench_docker swebench_docker
COPY --chmod=0755 docker/entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]
