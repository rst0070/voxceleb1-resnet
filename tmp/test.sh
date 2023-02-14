sudo docker container rm rst
sudo docker build -t vox1_rst .
sudo nvidia-docker run --name rst --shm-size=50gb -it -v /home/yeongsoo/datas/VoxCeleb1:/data -v /home/rst/workspace/vox-resnet:/app -t vox1_rst:latest