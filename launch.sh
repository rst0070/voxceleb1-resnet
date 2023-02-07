sudo docker container rm vox1_rst
sudo docker build -t vox1_rst .
sudo nvidia-docker run --name vox1_rst --shm-size=50gb -v /home/yeongsoo/datas/VoxCeleb1:/data -t vox1_rst:latest