# Generated by generate_dockerfiles.py

FROM aorwall/swe-bench-scikit-learn_scikit-learn-testbed:0.20

WORKDIR /opt/scikit-learn__scikit-learn

RUN git -c advice.detachedHead=false checkout b0e91e4110942e5b3c4333b1c6b6dfefbd1a6124

RUN pip install -v --no-build-isolation -e .
RUN pip install coverage cosmic-ray

WORKDIR /opt
