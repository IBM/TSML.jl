From ubuntu:18.04
LABEL maintainer="paulpalmes@ie.ibm.com"

RUN set -ex

ENV CONDA_JL_VERSION "3" 
ENV PYTHON ""

RUN apt-get update &&  apt-get install -y tzdata 

RUN apt-get update && apt-get -y upgrade && \
	apt-get install -y git r-base r-base-dev r-recommended wget

RUN apt-get clean -y && apt-get autoremove -y && apt-get autoclean -y

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.3-linux-x86_64.tar.gz 

RUN tar xzvf julia-1.0.3-linux-x86_64.tar.gz

RUN /julia-1.0.3/bin/julia -e 'using Pkg;Pkg.add(PackageSpec(url="https://github.com/IBM/TSML.jl",rev="master"));Pkg.build("TSML");using TSML'

RUN /julia-1.0.3/bin/julia -e 'using Pkg;Pkg.add("Conda");using Conda'

RUN /julia-1.0.3/bin/julia -e 'using Pkg;Pkg.add("IJulia");using IJulia'

RUN /root/.julia/conda/3/bin/conda install -y jupyter
RUN /root/.julia/conda/3/bin/conda install -y jupyterlab

CMD  /root/.julia/conda/3/bin/jupyter lab --ip="0.0.0.0" --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser

#  docker build -t tsml . && docker run --rm -p 8888:8888 -p 4040:4040  -it tsml  
