FROM aorwall/swe-bench-conda:bookworm-slim

RUN mkdir -p /testbed

WORKDIR /testbed

RUN git clone https://github.com/ytdl-org/youtube-dl.git

WORKDIR /testbed/youtube-dl

RUN git -c advice.detachedHead=false checkout b224cf3

RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir pytest pytest-cov coverage

RUN git checkout b224cf3 .

ENV TESTBED_NAME=ytdl-org__youtube-dl__b224cf3
ENV IMAGE_TYPE=conda
ENV REPO_DIR=/testbed/youtube-dl

WORKDIR /testbed
COPY swebench_docker swebench_docker
COPY --chmod=0755 docker/entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]
