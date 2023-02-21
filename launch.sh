#sudo docker container rm wf_concat_reduce_fc
sudo docker build -t wf_concat_reduce_fc .
sudo nvidia-docker run --name wf_concat_reduce_fc --shm-size=50gb -v /home/yeongsoo/datas/VoxCeleb1:/data -t wf_concat_reduce_fc:latest