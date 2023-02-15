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
	    'description'   : '~~기준, 뭐가 바뀌었는지 작성',

	    # log
        'path_save' : '/result.pth', # 모델을 저장할 위치
	    'path_log'      : '/results',
        'wandb_disabled': False,
        'wandb_key'     : '029f70f728310335e4824743783a5ae2ee3bcd21',
        'wandb_project' : 'model change',
	    'wandb_group'   : '',
        'wandb_name'    : 'separate downsampling',
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
		'rand_seed'		    : 1,
        
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