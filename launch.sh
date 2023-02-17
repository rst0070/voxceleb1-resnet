#sudo docker container rm rst_wf_
sudo docker build -t vox1_rst .
sudo nvidia-docker run --name rst_wf_total_log --shm-size=50gb -v /home/yeongsoo/datas/VoxCeleb1:/data -t vox1_rst:latest