from ghcr.io/antreasantoniou/tali-collector-dev:0.2.0

SHELL ["conda", "run", "-n", "tali-collector", "/bin/bash", "-c"]

RUN apt update
RUN apt install aptitude -y
RUN aptitude install glances -y
RUN mamba install -c conda-forge flake8 pylint -y
RUN echo yes | pip install itables tabulate nvitop 

ENTRYPOINT ["/bin/bash"]