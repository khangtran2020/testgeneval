# Generated by generate_dockerfiles.py

FROM aorwall/swe-bench-scikit-learn_scikit-learn-testbed:0.20

WORKDIR /opt/scikit-learn__scikit-learn

RUN git -c advice.detachedHead=false checkout dd69361a0d9c6ccde0d2353b00b86e0e7541a3e3

RUN pip install -v --no-build-isolation -e .
RUN pip install coverage cosmic-ray

WORKDIR /opt
