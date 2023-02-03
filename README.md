#


## `dataset.py`  
두가지의 클래스를 정의한다.  
- `TrainDataset`
- `TestDataset`
  
### `TrainDataset`  

__`__getitem__`의 return값__  
`(waveform:torch.Tensor, label:int)`을 return 한다.  
  
- `waveform` - 특정 발성에 대한 `4 * 16000`개 frame의 waveform
    - 이때 waveform의 shape은 `[4 * 16000]`이다.  
- `label` - waveform의 화자를 구분하는 번호(1~1211 중 하나)  

### `TestDataset`  

__`__getitem__`의 return값__  
`(waveforms1:torch.Tensor, waveforms2:torch.Tensor, label:int)`을 return 한다.  
  
- `waveforms1` - 화자1의 특정 발성에 대한 38개의(변동가능) waveform
    - 이것의 shape은 `[38, 4 * 16000]`이다.  

- `waveforms2` - 화자2의 특정 발성에 대한 38개의(변동가능) waveform
    - 이것의 shape은 `[38, 4 * 16000]`이다.

- `label` - 화자1과 화자2가 동일인물인지 나타내는 라벨.
    - `0` - 다른 화자
    - `1` - 같은 화자

## `model.py`
neural network model을 정의한다.  
neural network는 기본적으로 `torch.nn.Module`을 상속받는 클래스의 형태이어야 한다.  
  
__입력__  
특정한 길이의 waveform을 입력으로 받는다.  
  
__입력에 대한 처리__  


__출력__  


## `trainer.py`
`Trainer`라는 클래스를 정의한다.

__클래스 생성시__  
`__init__()`이 호출될때 train에 필요한것들을 인자로 전달받는다. 세부사항은 아래와 같다.  

- model - `model.py`에 정의되어있는 클래스의 인스턴스로 실제 train을 시킬 대상이다.  
- train_dataset - `dataset.py`에 정의되어있는 `TrainDataset` 클래스의 인스턴스로 train data를 `__getitem__`을 통해 넘겨준다.

위의 두가지를 통해 `__init__`내부에서 `dataloader`를 정의하고 model을 학습시킬 준비를 한다.  
  
__train과정__  
train과정은 `train()`이라는 클래스 내부의 함수가 호출되어 진행된다.  
  
- train의 진행단위: 한 epoch  


## `tester.py`
`Tester`라는 클래스를 정의한다. 

__클래스 생성시__  
`__init__()`이 호출될때 test에 필요한것들을 인자로 전달받는다. 세부사항은 아래와 같다.  
  
- model - `model.py`에 정의되어있는 클래스의 인스턴스로 실제 test를 위한 feature를 뽑아내는 도구이다.  
- test_dataset - `dataset.py`에 정의되어있는 `TestDataset` 클래스의 인스턴스로 test data를 `__getitem__`을 통해 넘겨준다.  
  
__test과정__  
test과정은 `test()`라는 클래스 내부의 함수가 호출되어 진행된다.  
  
- test의 진행단위: 한 epoch