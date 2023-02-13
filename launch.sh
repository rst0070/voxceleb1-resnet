sudo docker container rm rst
sudo docker build -t vox1_rst .
sudo nvidia-docker run --name rst --shm-size=50gb -v /home/yeongsoo/datas/VoxCeleb1:/data -t vox1_rst:latest