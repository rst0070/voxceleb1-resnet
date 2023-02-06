sudo docker container rm vox1_proj
sudo docker build -t vox1_adam .
sudo nvidia-docker run --name vox1_proj --shm-size=50gb -v /home/yeongsoo/datas/VoxCeleb1:/data -t vox1_adam:latest