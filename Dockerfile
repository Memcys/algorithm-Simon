# FROM sagemath/sagemath AS sage

# FROM julia:latest AS yao

FROM sagemathinc/cocalc

LABEL Name=quantum-algorithm-Simon Version=0.1
# RUN apt-get -y update && apt-get install -y fortunes
# CMD ["sh", "-c", "/usr/games/fortune -a | cowsay"]

WORKDIR /code

# add julia packages
# ! substitute the Julia mirror to whichever you prefer
RUN export JULIA_PKG_SERVER=https://mirrors.tuna.tsinghua.edu.cn/julia
RUN julia -e 'using Pkg; Pkg.add(["Yao", "YaoPlots", "BitBasis", "CSV", "DataFrames"])'

# Install pip requirements
COPY quafu/requirements.txt quafu/
# ! substitute the PyPI mirror to whichever you prefer
RUN sage -pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r quafu/requirements.txt

# add sagemath
# ARG SAGE_ROOT=/home/sage/sage
# COPY --from=sage /home/sage/sage /usr/bin/sage
# RUN ln -s /usr/bin/sage /usr/bin/sagemath

# WORKDIR Simon