# Generated by generate_dockerfiles.py

FROM aorwall/swe-bench-scikit-learn_scikit-learn-testbed:0.22

WORKDIR /opt/scikit-learn__scikit-learn

RUN git -c advice.detachedHead=false checkout ecea98f9c965fbb25d2dfb1fdc23784364aa5657

RUN pip install -v --no-build-isolation -e .
RUN pip install coverage cosmic-ray

WORKDIR /opt
