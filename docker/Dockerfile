FROM debian:buster AS build
LABEL maintainer="paulpalmes@ie.ibm.com"

WORKDIR /

RUN set -ex

RUN apt-get update --allow-releaseinfo-change && apt-get -y upgrade && \
	apt-get install -y tzdata git wget build-essential

RUN apt-get clean -y && apt-get autoremove -y && apt-get autoclean -y

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.1-linux-x86_64.tar.gz &&  tar xzvf julia-1.1.1-linux-x86_64.tar.gz
ENV PATH="$PATH:/julia-1.1.1/bin"

RUN git clone https://github.com/JuliaLang/PackageCompiler.jl.git 

COPY main.jl /
RUN /julia-1.1.1/bin/julia -e 'using Pkg; pkg"up; add TSML#docker_packagecompiler PackageCompiler#sd-notomls ArgParse"'
RUN /julia-1.1.1/bin/julia -e 'using TSML, PackageCompiler; sysnew,sysold = compile_incremental(:TSML;install=true); cp(sysnew,sysold,force=true)'
RUN /julia-1.1.1/bin/julia /PackageCompiler.jl/juliac.jl -vae /main.jl
ENTRYPOINT ["/builddir/main"]

#FROM debian:buster
#WORKDIR /
#COPY --from=build /julia-1.1.1 /julia-1.1.1
#COPY --from=build /root/.julia /root/.julia
#ENV PATH="$PATH:/julia-1.1.1/bin"
#COPY tsmldockermain.jl /tsmldockermain.jl
#RUN chmod +x /tsmldockermain.jl
#ENTRYPOINT ["/tsmldockermain.jl"]
