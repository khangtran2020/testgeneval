# Generated by generate_dockerfiles.py

FROM aorwall/swe-bench-scikit-learn_scikit-learn-testbed:0.21

WORKDIR /opt/scikit-learn__scikit-learn

RUN git -c advice.detachedHead=false checkout b7b4d3e2f1a65bcb6d40431d3b61ed1d563c9dab

RUN pip install -v --no-build-isolation -e .
RUN pip install coverage cosmic-ray

WORKDIR /opt
