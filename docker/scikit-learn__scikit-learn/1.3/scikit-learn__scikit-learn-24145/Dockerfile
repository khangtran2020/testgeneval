# Generated by generate_dockerfiles.py

FROM aorwall/swe-bench-scikit-learn_scikit-learn-testbed:1.3

WORKDIR /opt/scikit-learn__scikit-learn

RUN git -c advice.detachedHead=false checkout 55af30d981ea2f72346ff93602f0b3b740cfe8d6

RUN pip install -v --no-use-pep517 --no-build-isolation -e .
RUN pip install coverage cosmic-ray

WORKDIR /opt
