# Generated by generate_dockerfiles.py

FROM aorwall/swe-bench-scikit-learn_scikit-learn-testbed:0.21

WORKDIR /opt/scikit-learn__scikit-learn

RUN git -c advice.detachedHead=false checkout f9af18b4e5b9d4b379867d32381296062782dc15

RUN pip install -v --no-build-isolation -e .
RUN pip install coverage cosmic-ray

WORKDIR /opt
