# Generated by generate_dockerfiles.py

FROM aorwall/swe-bench-sphinx-doc_sphinx:bookworm-slim

RUN git -c advice.detachedHead=false checkout 0938c193ea6f56dbb930bfb323602bc4e2b7b9c6

RUN conda create -n sphinx-doc__sphinx__4.4 python=3.9  -y
SHELL ["conda", "run", "-n", "sphinx-doc__sphinx__4.4", "/bin/bash", "-c"]

RUN sed -i 's/pytest/pytest -rA/' tox.ini
RUN sed -i 's/Jinja2>=2.3/Jinja2<3.0/' setup.py
RUN sed -i 's/sphinxcontrib-applehelp/sphinxcontrib-applehelp<=1.0.7/' setup.py
RUN sed -i 's/sphinxcontrib-devhelp/sphinxcontrib-devhelp<=1.0.5/' setup.py
RUN sed -i 's/sphinxcontrib-qthelp/sphinxcontrib-qthelp<=1.0.6/' setup.py
RUN sed -i 's/alabaster>=0.7,<0.8/alabaster>=0.7,<0.7.12/' setup.py
RUN sed -i "s/'packaging',/'packaging', 'markupsafe<=2.0.1',/" setup.py
RUN sed -i 's/sphinxcontrib-htmlhelp>=2.0.0/sphinxcontrib-htmlhelp>=2.0.0,<=2.0.4/' setup.py
RUN sed -i 's/sphinxcontrib-serializinghtml>=1.1.5/sphinxcontrib-serializinghtml>=1.1.5,<=1.1.9/' setup.py

RUN pip install tox
RUN pip install coverage cosmic-ray

RUN pip install -e .[test]


RUN git checkout 0938c193ea6f56dbb930bfb323602bc4e2b7b9c6 .

ENV TESTBED_NAME=sphinx-doc__sphinx__4.4
ENV IMAGE_TYPE=conda

WORKDIR /home/swe-bench

COPY swebench_docker swebench_docker

ENTRYPOINT ["./entrypoint.sh"]
