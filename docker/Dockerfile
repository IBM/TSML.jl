From ubuntu:18.04
LABEL maintainer="paulpalmes@ie.ibm.com"

WORKDIR /

RUN set -ex

ENV CONDA_JL_VERSION "3" 
ENV PYTHON ""

RUN apt-get update &&  apt-get install -y --allow-unauthenticated tzdata 

RUN apt-get update && apt-get -y --allow-unauthenticated upgrade && \
	apt-get install -y --allow-unauthenticated git wget 

RUN apt-get clean -y && apt-get autoremove -y && apt-get autoclean -y

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.1-linux-x86_64.tar.gz

RUN tar xzvf julia-1.1.1-linux-x86_64.tar.gz

RUN /julia-1.1.1/bin/julia -e 'using Pkg;Pkg.add(PackageSpec(url="https://github.com/IBM/TSML.jl",rev="docker_packagecompiler"));Pkg.build("TSML");using TSML'

RUN /julia-1.1.1/bin/julia -e 'using Pkg;Pkg.add(["DataFrames","CSV","PackageCompiler"]);using CSV;using DataFrames'

RUN apt-get update && apt-get -y --allow-unauthenticated upgrade && \
	apt-get install -y --allow-unauthenticated build-essential

RUN /julia-1.1.1/bin/julia -e 'using Pkg;Pkg.build("PackageCompiler")'

RUN /julia-1.1.1/bin/julia -e 'using PackageCompiler; compile_package("TSML",force=false)'

ENV PATH="$PATH:/julia-1.1.1/bin"

COPY tsmldockermain.jl /tsmldockermain.jl

RUN chmod +x /tsmldockermain.jl

ENTRYPOINT ["/tsmldockermain.jl"]
