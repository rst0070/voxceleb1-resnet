FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN apt-get update

RUN pip install torch==1.10.2
RUN pip install torchaudio==0.10.0
RUN pip install torchtext==0.11.2

RUN pip install wandb

RUN pip install sklearn
RUN pip install scipy
RUN pip install pytorch_lightning
RUN pip install omegaconf==2.0.6

RUN pip install pip --upgrade
COPY ./ /app/
WORKDIR /app

CMD ["touch", "/app/result.pth"]
CMD ["python",  "main.py"]