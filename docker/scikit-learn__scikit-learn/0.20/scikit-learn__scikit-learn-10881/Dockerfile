# Generated by generate_dockerfiles.py

FROM aorwall/swe-bench-scikit-learn_scikit-learn-testbed:0.20

WORKDIR /opt/scikit-learn__scikit-learn

RUN git -c advice.detachedHead=false checkout 4989a9503753a92089f39e154a2bb5d160b5d276

RUN pip install -v --no-build-isolation -e .
RUN pip install coverage cosmic-ray

WORKDIR /opt
