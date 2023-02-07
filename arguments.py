import os
import itertools
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
	    'wandb_group'   : '',
	    'wandb_entity'  : '',

        # dataset
        'path_train_label'  :   'labels/tmp_train_label.csv',
        'path_train'        :   '/data/train',
        'path_test_label'   :   'labels/tmp_trial_label.csv',
        'path_test'         :   '/data/test',

        # processor
        'cpu'           : "cpu",
        'gpu'           : ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
        
        # others
        'num_workers': 0,
	    'usable_gpu': None,
    }

    experiment_args = {
        # experiment
        'epoch'             : 80,
        'batch_size'        : 32,
		'rand_seed'		    : 1,
        
        # model
		'embedding_size'	: 512,
        'aam_margin'        : 0.15,
        'aam_scale'         : 20,
        'spec_mask_F'       : 100,
        'spec_mask_T'       : 10,

        # data processing
        'test_sample_num'   : 10, # test시 발성에서 몇개의 sample을 뽑아낼것인지
        'num_seg'           : 10,
        'num_train_frames'  : 3 * 16000, # train에서 input 으로 사용할 frame 개수
        #'num_test_frames'   : 300,
        
        # learning rate
        'lr'            : 1e-4,
        'lr_min'        : 1e-6,
		'weight_decay'  : 1e-5,
        'T_0'           : 80,
        'T_mult'        : 1,
    }

    return system_args, experiment_args