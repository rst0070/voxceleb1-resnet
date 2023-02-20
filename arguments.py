import torch

def get_args():
    """
	Returns
		system_args (dict): path, log setting
		experiment_args (dict): hyper-parameters
		args (dict): system_args + experiment_args
    """
    system_args = {
	    # expeirment info
	    'project'       : '',
	    'name'          : 'experiment_001',
	    'tags'          : ['', ''],
	    'description'   : '',

	    # log
        'path_save' : '/result.pth', # 모델을 저장할 위치
	    'path_log'      : '/results',
        'wandb_disabled': False,
        'wandb_key'     : '6ef86c7e660c02088ca226a60f3e1073b3f78876',
        'wandb_project' : 'Voxceleb1 resnet18',
	    'wandb_group'   : '',
        'wandb_name'    : 'waveform - normalzing과 동일한 환경 - embedding의 relu 제거함',
	    'wandb_entity'  : 'irlab_undgrd',
        'wandb_notes'   : '',

        # dataset
        'path_train_label'  :   'labels/train_label.csv',
        'path_train'        :   '/data/train',
        'path_test_label'   :   'labels/trial_label.csv',
        'path_test'         :   '/data/test',

        # processor
        'cpu'           : "cpu",
        'gpu'           : ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
        
        # others
        'num_workers': 4,
	    'usable_gpu': None,
    }

    experiment_args = {
        # experiment
        'epoch'             : 100,
        'batch_size'        : 64,
		'rand_seed'		    : 0, # 이전에도 항상 0으로 돌아갔었다. 이 파일의 값을 참조안했을뿐
        
        # model
		'embedding_size'	: 128,
        'aam_margin'        : 0.15,
        'aam_scale'         : 20,
        'spec_mask_F'       : 100,
        'spec_mask_T'       : 10,

        # data processing
        'test_sample_num'   : 10, # test시 발성에서 몇개의 sample을 뽑아낼것인지
        'num_seg'           : 10,
        'num_train_frames'  : int(3.2 * 16000)-1, # train에서 input 으로 사용할 frame 개수
        'sample_rate'       : 16000, # voxceleb1의 기본 sample rate
        
        # mel config
        'n_fft'             : 512,
        'n_mels'            : 64,
        'win_length'        : int(25*0.001*16000), 
        'hop_length'        : int(10*0.001*16000),
        'f_min'             : int(100),
        'f_max'             : int(8000),
        #'num_test_frames'   : 300,
        
        # learning rate
        'lr'            : 1e-3,
        'lr_min'        : 1e-6,
		'weight_decay'  : 1e-5,
        'T_0'           : 80,
        'T_mult'        : 1,
    }

    return system_args, experiment_args