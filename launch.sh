sudo docker container rm wf_embsize_non_relu
sudo docker build -t vox1_rst .
sudo nvidia-docker run --name wf_embsize_non_relu --shm-size=50gb -v /home/yeongsoo/datas/VoxCeleb1:/data -t vox1_rst:latest