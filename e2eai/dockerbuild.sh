sudo buildkitd 
#sudo docker build --no-cache -t tsml:latest .
sudo docker build -t tsml:latest .
sudo docker tag tsml:latest ppalmes/tsml:latest
#sudo docker run -it --rm ppalmes/automlpipeline:latest julia --project main.jl 5 1
sudo docker login registry.docker.com
sudo docker push ppalmes/tsml:latest

sudo docker run --rm -v $PWD:/data  ppalmes/tsml:latest -w 5 -o /data/output.txt iris.csv
julia --project main_parse.jl -c "high" -t "classification"  ../data/diabetes.csv

sudo docker build --no-cache -t amlp:latest -f ./Dockerfile.sysimage .
sudo docker tag amlp:latest ppalmes/amlp:latest
#sudo docker run -it --rm ppalmes/amlp:latest julia --project main.jl 5 1
sudo docker login registry.docker.com
sudo docker push ppalmes/amlp:latest
