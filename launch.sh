#sudo docker container rm wf_concat_reduce_fc
sudo docker build -t wf_concat_reduce_fc_without_norm .
sudo nvidia-docker run --name wf_concat_reduce_fc_without_norm --shm-size=50gb -v /home/yeongsoo/datas/VoxCeleb1:/data -t wf_concat_reduce_fc_without_norm:latest