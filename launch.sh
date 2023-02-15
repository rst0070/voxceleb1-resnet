sudo docker container rm rst2
sudo docker build -t vox1_resnet .
sudo nvidia-docker run --name rst2 --shm-size=50gb -v /home/yeongsoo/datas/VoxCeleb1:/data -t vox1_resnet:latest