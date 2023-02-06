## 정보
- train 화자 수: 1211명
- window length: 400
- hop length: 160
- 가장긴 frame 개수: 2318721

## `main.py`  
프로그램의 시작점. 각 객체들을 초기화 시킨다.
  
## `arguments.py`  
hyper parameter, system arguments등을 정의한다.  
기본적으로 main.py에서 해당 파일을 import하여 초기화시 사용하는것을 예상하고 있으며,   
각 test, train, model등 logic에서 learning rate, weight decay등 필요한것이 있을경우 추가로 참조할것.

## `dataset.py`  
두가지의 클래스를 정의한다.  
- `TrainDataset`
- `TestDataset`
  
### `TrainDataset`  

__`__getitem__`의 return값__  
`(waveform:torch.Tensor, label:int)`을 return 한다.  
  
- `waveform` - 특정 발성에 대한 `4 * 16000`개 frame의 waveform
    - 이때 waveform의 shape은 `[1, 4 * 16000]`이다.  
- `label` - waveform의 화자를 구분하는 번호(1~1211 중 하나)  

### `TestDataset`  

__`__getitem__`의 return값__  
`(waveform1:torch.Tensor, wf_id1:str, waveform2:torch.Tensor, wf_id2:str, label:int)`을 return 한다.  
  
- `waveform1` - 화자1의 특정 발성에 대한 waveform
    - 이것의 shape은 `[1, sec * 16000]`이다. 이때 `sec`은 4이상    
  
- `wf_id1` - `waveform1`에 대한 고유한 id  
  
- `waveform2` - 화자2의 특정 발성에 대한 waveform
    - 이것의 shape은 `[1, sec * 16000]`이다. 이때 `sec`은 4이상  
  
- `wf_id2` - `waveform2`에 대한 고유한 id  
  
- `label` - 화자1과 화자2가 동일인물인지 나타내는 라벨.
    - `0` - 다른 화자
    - `1` - 같은 화자
  

`wf_id1`, `wf_id2`를 통해서 waveform들을 효율적으로 관리할 수 있다. (ex. 같은 id인 waveform에 대해 임베딩을 저장하는 방식)  

## `model.py`
neural network model을 정의한다.  
neural network는 기본적으로 `torch.nn.Module`을 상속받는 클래스의 형태이어야 한다.  
  
__입력__  
특정한 길이의 waveform을 입력으로 받는다.  
  
__입력에 대한 처리__  
입력을 mel spectogram화 하여 신경망을 통과시킨다. 

__출력__  
`is_test`의 bool값에 따라 embedding 혹은 classification값을 출력한다. 

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