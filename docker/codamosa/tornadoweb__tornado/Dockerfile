FROM aorwall/swe-bench-conda:bookworm-slim

WORKDIR /opt

USER root

RUN git clone https://github.com/tornadoweb/tornado.git

WORKDIR /opt/tornado

RUN git -c advice.detachedHead=false checkout 2047e7a

RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir pytest pytest-cov coverage

RUN git checkout 2047e7a .

ENV TESTBED_NAME=tornadoweb__tornado__2047e7a
ENV IMAGE_TYPE=python
ENV REPO_DIR=/opt/tornado

WORKDIR /opt

COPY swebench_docker swebench_docker
COPY --chmod=0755 docker/entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]
