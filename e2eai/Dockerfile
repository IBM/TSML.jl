#From --platform=linux/amd64 julia:latest
# for nerdctl
From julia:latest
MAINTAINER Paulito Palmes "paulpalmes@ie.ibm.com"
RUN apt-get update && apt-get install build-essential -y
RUN mkdir /tsml
RUN cd /tsml
WORKDIR /tsml
COPY Project.toml /tsml/
COPY tsml_sysimage.jl /tsml/
COPY tsml_precompile.jl /tsml/
RUN julia --project -e 'using Pkg;\
       Pkg.instantiate();\
       Pkg.update();\
       Pkg.precompile();\
       include("tsml_sysimage.jl")'
COPY * /tsml/
#ENTRYPOINT ["julia", "--project", "-Jtsml.so", "--", "main_parse.jl"]
