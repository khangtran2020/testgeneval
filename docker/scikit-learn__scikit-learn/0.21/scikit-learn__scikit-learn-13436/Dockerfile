# Generated by generate_dockerfiles.py

FROM aorwall/swe-bench-scikit-learn_scikit-learn-testbed:0.21

WORKDIR /opt/scikit-learn__scikit-learn

RUN git -c advice.detachedHead=false checkout 77b73d63d05bc198ba89193582aee93cae1f69a4

RUN pip install -v --no-build-isolation -e .
RUN pip install coverage cosmic-ray

WORKDIR /opt
