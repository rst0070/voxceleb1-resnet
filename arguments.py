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
	    'path_log'      : '/results',
	    'wandb_group'   : '',
	    'wandb_entity'  : '',

        # dataset
        'path_train_label'  :   'labels/train_label.csv',
        'path_train'    : '/data/VoxCeleb2_TimeStretch/train',
        'path_test_label'  :   'labels/train_label.csv',
        'path_test'     : '/data/VoxCeleb1',
	    'path_trials'  	: '/data/VoxCeleb1/trials',
	    'path_musan'  	: '/data/musan',
        'path_rir'      : '/data/RIRS_NOISES/simulated_rirs',

        # processor
        'cpu'           : "cpu",
        'gpu'           : ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
        
        # others
        'num_workers': 4,
	    'usable_gpu': None,
    }

    experiment_args = {
        # huggingface model
        'model_name_or_path'    : 'facebook/wav2vec2-large-xlsr-53',
        
        # experiment
        'epoch'             : 20,
        'batch_size'        : 128,
		'rand_seed'		    : 1,
        
        # model
        'C'                 : 1024,
        'num_hidden_layers' : 4,
        'n_class'           : 5994 * 3,
		'embedding_size'	: 192,
        'aam_margin'        : 0.15,
        'aam_scale'         : 20,
        'spec_mask_F'       : 100,
        'spec_mask_T'       : 10,

        # data processing
        'sample_num'        : 5,
        'num_seg'           : 5,
        'num_train_frames'  : 4 * 16000, # train에서 input 으로 사용할 frame 개수
        'num_test_frames'   : 300,
        
        # learning rate
        'lr'            : 1e-4,
        'lr_min'        : 1e-6,
		'weight_decay'  : 0,
        'T_0'           : 80,
        'T_mult'        : 1,
    }

    return system_args, experiment_args