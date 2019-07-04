From debian:buster
LABEL maintainer="paulpalmes@ie.ibm.com"

WORKDIR /

RUN set -ex

RUN apt-get update &&  apt-get install -y  tzdata 

RUN apt-get update && apt-get -y  upgrade && \
	apt-get install -y  git wget build-essential

RUN apt-get clean -y && apt-get autoremove -y && apt-get autoclean -y

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.1-linux-x86_64.tar.gz

RUN tar xzvf julia-1.1.1-linux-x86_64.tar.gz

#RUN /julia-1.1.1/bin/julia -e 'using Pkg;Pkg.add(PackageSpec(url="https://github.com/IBM/TSML.jl",rev="docker_packagecompiler"));using TSML'
#RUN /julia-1.1.1/bin/julia -e 'using Pkg;Pkg.add(PackageSpec(name="TSML",rev="docker_packagecompiler"));using TSML'
#RUN /julia-1.1.1/bin/julia -e 'using Pkg;Pkg.add(["DataFrames","CSV","PackageCompiler"]);using CSV;using DataFrames'
#RUN /julia-1.1.1/bin/julia -e 'using Pkg; pkg"precompile";using DataFrames'
#RUN /julia-1.1.1/bin/julia -e 'using PackageCompiler; compile_incremental(:DataFrames,install=true)'
#RUN /julia-1.1.1/bin/julia -e 'using Pkg;Pkg.add(PackageSpec(url="https://github.com/JuliaLang/PackageCompiler.jl.git",rev="master"));Pkg.build("PackageCompiler")'
#RUN /julia-1.1.1/bin/julia -e 'using Pkg;Pkg.add(PackageSpec(url="https://github.com/JuliaLang/PackageCompiler.jl.git",rev="sd-notomls"));Pkg.build("PackageCompiler")'
#RUN /julia-1.1.1/bin/julia -e 'using Pkg;Pkg.add(PackageSpec(url="https://github.com/JuliaLang/PackageCompiler.jl.git",rev="sd-notomls"))'
#RUN /julia-1.1.1/bin/julia -e 'using Pkg;Pkg.add(PackageSpec(name="PackageCompiler",rev="sd-notomls"))'
#RUN /julia-1.1.1/bin/julia -e 'using TSML; using PackageCompiler; compile_incremental(:TSML,install=true,force=true)'

#RUN /julia-1.1.0/bin/julia -e 'using Pkg; pkg"add TSML#docker_packagecompiler"'
#RUN /julia-1.1.0/bin/julia -e 'using Pkg; pkg"add PackageCompiler#sd-notomls"'
#RUN /julia-1.1.0/bin/julia -e 'using Pkg; pkg"precompile"'
#RUN /julia-1.1.0/bin/julia -e 'using TSML; using PackageCompiler; compile_incremental(:TSML;install=true)'
#
#ENV PATH="$PATH:/julia-1.1.1/bin"
#COPY tsmldockermain.jl /tsmldockermain.jl
#RUN chmod +x /tsmldockermain.jl
#ENTRYPOINT ["/tsmldockermain.jl"]
