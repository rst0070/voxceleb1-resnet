sudo docker container rm wf_embsize
sudo docker build -t vox1_rst .
sudo nvidia-docker run --name wf_embsize --shm-size=50gb -v /home/yeongsoo/datas/VoxCeleb1:/data -t vox1_rst:latest