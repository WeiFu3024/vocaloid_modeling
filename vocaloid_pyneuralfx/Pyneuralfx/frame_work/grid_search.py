import utils
import torch
import numpy as np
from utils import traverse_dir
import os
import sys
sys.path.append('')
sys.path.append('..')
from tqdm.contrib import tzip
import json
from pyneuralfx.loss_func.loss_func import * 
from pyneuralfx.eval_metrics.eval_metrics import * 
import librosa
from tqdm import tqdm
from shutil import copyfile

import solver
from dataset import SnapShot_AudioDataset



# ============================================================ #
# Functions
# ============================================================ #
def collate_fn(batch):
	wav_x_s = []
	wav_y_s = []
	#cond_s = []

	for idx in range(len(batch)):
		wav_x, wav_y = batch[idx]
		wav_x_s.append(wav_x[None, ...])
		wav_y_s.append(wav_y[None, ...])
		#cond_s.append([cond])

	x_final = np.concatenate(wav_x_s, axis=0)
	y_final = np.concatenate(wav_y_s, axis=0)
	#c_final = np.concatenate(cond_s, axis=0)

	return torch.from_numpy(x_final), torch.from_numpy(y_final), None #, torch.from_numpy(c_final)


def inference(path_savedir, exp_dir_val):
	global model
	print(' >>>>> inference')

	# load model 
	model = utils.load_model(
				exp_dir_val,
				model,
				device=args.device, 
				name='best_params.pt')
	
	# data
	valid_set = SnapShot_AudioDataset(
		input_data_path = args.data.test_x_path, 
		target_data_path = args.data.test_y_path, 
		sr=args.data.sampling_rate,
		win_len=None,
		pre_room=PRE_ROOM,)

	loader_valid = torch.utils.data.DataLoader(
		valid_set,
		batch_size=args.inference.batch_size,
		shuffle=False,
		num_workers=2,
		pin_memory=True,
		collate_fn=collate_fn
	)

	# validate
	path_outdir = os.path.join(exp_dir_val, path_savedir) 
	solver.validate(  
		args, 
		model, 
		loader_valid,
		loss_func_val, 
		path_save=path_outdir)
	
	amount, amount_train = model.compute_num_of_params()
	print(' > params amount: {:,d} | trainable: {:,d}'.format(amount, amount_train))


def train(args):
	global model

	if args.load_dir:
		print(' >>>>> fine-tuning')
		model = utils.load_model(
				LOAD_DIR,
				model,
				device=args.device, 
				name='best_params.pt')
	else:
		print(' >>>>> training')

	# datasets
	
	train_set = SnapShot_AudioDataset(
		input_data_path = args.data.train_x_path, 
		target_data_path = args.data.train_y_path, 
		sr=args.data.sampling_rate,
		win_len=args.data.buffer_size,
		pre_room=PRE_ROOM,)
	
	loader_train = torch.utils.data.DataLoader(
		train_set,
		batch_size=args.train.batch_size,
		shuffle=True,
		num_workers=2,
		pin_memory=True,
		collate_fn=collate_fn
	)
	
	print('> train dataset ready ...........')

	
	valid_set = SnapShot_AudioDataset(
		input_data_path = args.data.valid_x_path, 
		target_data_path = args.data.valid_y_path, 
		sr=args.data.sampling_rate,
		win_len=args.data.buffer_size,
		pre_room=PRE_ROOM,)
	
	
	loader_valid = torch.utils.data.DataLoader(
		valid_set,
		batch_size=args.train.batch_size,
		shuffle=False,
		num_workers=2,
		pin_memory=True,
		collate_fn=collate_fn,
	)
	

	print('> valid dataset ready ...........')
	os.makedirs(args['env']['expdir'], exist_ok=True)
	
	copyfile(__file__, os.path.join(args['env']['expdir'], os.path.basename(__file__)))
	copyfile(cmd['config'], os.path.join(args['env']['expdir'], os.path.basename(cmd['config'])))
	# training
	
	solver.train(
		args, 
		model, 
		loss_funcs, 
		optimizer,
		scheduler,
		loader_train, 
		valid_set=loader_valid,
		is_jit=args.env.is_jit)


def validation_loss(exp_names, args):
	SR = 44100

	REPORTED_METRICS = {
		'HybridLoss': HybridLoss(),
		'MRSTFTLoss': MRSTFTLoss(
			scales=[2048]
		),
		'ESRLoss': ESRLoss(),
		'Transient': TransientPreservation_v2(SR),
		'LUFS': LUFS(sr=SR),
		'CrestFactor': CrestFactor(),
		'RMSEnergy': RMSEnergy(),
		'SpectralCentroid': SpectralCentroid(),
	}

	losses = []

	for exp_name in exp_names:
		print('> exp name: ', exp_name)

		for t in ['valid_gen']:

			path_json = os.path.join('exp',  exp_name, t)

			path_gt = os.path.join('exp',  exp_name, t, 'anno')
			path_pred = os.path.join('exp',  exp_name, t, 'pred')


			filelist_gt = traverse_dir(path_gt, is_pure=True, is_sort=True)
			filelist_pred = traverse_dir(path_pred, is_pure=True, is_sort=True)
			assert len(filelist_gt) == len(filelist_pred)
			print('> total audio clips: ', len(filelist_pred))

			loss_record = {}
			loss_record.clear()

			# initial loss record 
			for k in REPORTED_METRICS:
				loss_record[k] = []
		
			for (gt_fn, pred_fn) in tzip(filelist_gt, filelist_pred):
			
				wav_gt, sr_gt = librosa.load(os.path.join(path_gt, gt_fn), sr=None, mono=True)
				wav_pred, sr_pred = librosa.load(os.path.join(path_pred, pred_fn), sr=None, mono=True)
			
				assert gt_fn == pred_fn
				assert sr_gt == sr_pred
				assert sr_gt == SR 

				# to torch 
				wav_gt = torch.from_numpy(wav_gt).unsqueeze(0).unsqueeze(0).float()
				wav_pred = torch.from_numpy(wav_pred).unsqueeze(0).unsqueeze(0).float()
			
				for k in REPORTED_METRICS:
					metri = REPORTED_METRICS[k]
					score = metri(wav_pred, wav_gt)
					loss_record[k].append(score.item())
			
		
			print( '##################################################')
			print(f'#####    evaluation report   {t, exp_name}  #########')
			print( '##################################################')
			final_reports = {}
			final_reports.clear()
			for r in loss_record:
				loss_value = np.mean(loss_record[r])
				print(f'> metric {r}: ', loss_value)
				final_reports[r] = loss_value
			print('#########################################')
			losses.append(final_reports['HybridLoss'])

		file_names = os.path.basename(exp_name)
	
		with open(os.path.join(path_json, f'{file_names}_metric.json'), 'w') as f:
			json.dump(final_reports, f, indent = 4)
		final_reports.clear()

		#dump hyperparameters to the result folder
		with open(os.path.join('exp', exp_name, 'hyperparam.json'), 'w') as f:
			json.dump({
				'causal': args['model']['causal'],
				'dilation_growth': args['model']['dilation_growth'],
				'n_channel': args['model']['n_channel'],
				'n_blocks': args['model']['n_blocks'],
				'kernel_size': args['model']['kernel_size'],
				'epochs': args['train']['epochs'],
				'lr': args['train']['lr'],
				'batch_size': args['train']['batch_size']
				},f,indent = 4)

	losses = np.array(losses)
	print('Mean Loss: ', np.mean(losses))
	return np.mean(losses)


if __name__ == '__main__':
	print("==========================")

	cmd = {
		# 'config': './configs/cnn/gcn/snapshot_gcn.yml'
		'config': './configs/cnn/tcn/snapshot_tcn.yml'
	}

	# Change here every time you run a different experiment:
	#						||
	#					   \||/
	#						\/
	uuid = 're1'

	# parameter space
	causal = [False]
	dilation_growth = [2]       # change later
	n_channel = [32]
	epochs = [20]
	lr = [0.0005]
	batch_size = [40]
	kernel_size = [7, 9]
	n_blocks = [8, 10, 12]

	args = utils.load_config(cmd['config'])

	# Compute total iterations for tqdm
	total_iterations = (
		len(causal) * len(dilation_growth) * len(n_channel) * 
		len(epochs) * len(lr) * len(batch_size) * len(n_blocks) * len(kernel_size)
	)


	best_loss = float('inf')
	best_params = None

	experiment_count = 0
	
	with tqdm(total=total_iterations, desc="Grid Search Progress") as pbar:
		for c in causal:
			for dg in dilation_growth:
				for nc in n_channel:
					for e in epochs:
						for l in lr:
							for b in batch_size:
								for nb in n_blocks:
									for ks in kernel_size:
										args['model']['causal'] = c
										args['model']['dilation_growth'] = dg
										args['model']['n_channel'] = nc
										args['model']['n_blocks'] = nb
										args['model']['kernel_size'] = ks
										args['train']['epochs'] = e
										args['train']['lr'] = l
										args['train']['batch_size'] = b
										args['env']['expdir'] = os.path.join('exp', f'vocaloid/snapshot_tcn_{uuid}_{experiment_count}')

										print('=======================')
										print('train_x_path:', args['data']['train_x_path'])
										print('train_y_path:', args['data']['train_y_path'])
										print(f'> causal: {c}')
										print(f'> dilation_growth: {dg}')
										print(f'> n_channel: {nc}')
										print(f'> epochs: {e}')
										print(f'> lr: {l}')
										print(f'> batch_size: {b}')
										print(f'> kernel_size: {ks}')
										print(f'> n_blocks: {nb}')
										print('=======================')

										# loss functions
										loss_func_tra = utils.setup_loss_funcs(args) 
										loss_func_val = utils.setup_loss_funcs(args) 
										loss_funcs = [loss_func_tra, loss_func_val]

										# device 
										device = 'cuda' if torch.cuda.is_available() else 'cpu'
										if device == 'cuda':
											torch.cuda.set_device(args.env.gpu_id)
										args['device'] = device

										# model 
										model = utils.setup_models(args)

										# expdir
										LOAD_DIR = args.env.load_dir
										print('EXP DIR: ', args.env.expdir)

										PRE_ROOM = model.compute_receptive_field()[0] - 1
										args['model']['pre_room'] = PRE_ROOM

										# optimizer
										optimizer = torch.optim.Adam(
											filter(lambda p: p.requires_grad, model.parameters()), args.train.lr
										)
										scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
											optimizer, mode='min', factor=0.5, patience=args.train.lr_patience, verbose=True
										)

										# to device
										model.to(args.device)
										for func in loss_funcs:
											func.to(args.device)

										utils.check_configs(args)
										train(args)
										inference('valid_gen', args.env.expdir)

										# separate different experiment's data to different folder
										en = [f'vocaloid/snapshot_tcn_{uuid}_{experiment_count}']
										loss = validation_loss(en, args)
										
										if loss < best_loss:
											best_loss = loss
											best_params = [c, dg, nc, e, l, b, ks, nb]
											print(f'Best Loss: {best_loss}')
											print(f'Best Params: {best_params}')
										print('=======================')
										
										# Update tqdm progress bar
										pbar.update(1)

										#update counter
										experiment_count += 1

	print('Best Loss: ', best_loss)
	print('Best Params: ', best_params)