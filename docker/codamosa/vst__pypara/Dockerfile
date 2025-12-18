FROM aorwall/swe-bench-conda:bookworm-slim

RUN mkdir -p /testbed

WORKDIR /testbed

RUN git clone https://github.com/vst/pypara.git

WORKDIR /testbed/pypara

RUN git -c advice.detachedHead=false checkout 7d705a54

RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir pytest pytest-cov coverage

RUN git checkout 7d705a54 .

ENV TESTBED_NAME=vst__pypara__7d705a54
ENV IMAGE_TYPE=conda
ENV REPO_DIR=/testbed/pypara

WORKDIR /testbed
COPY swebench_docker swebench_docker
COPY --chmod=0755 docker/entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]
