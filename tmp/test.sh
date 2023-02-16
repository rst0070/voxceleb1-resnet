sudo docker container rm rst1
sudo docker build -t rst_test .
sudo nvidia-docker run --name rst1 --shm-size=50gb -it -v /home/yeongsoo/datas/VoxCeleb1:/data -v /home/rst/workspace/vox-resnet:/app -t rst_test:latest