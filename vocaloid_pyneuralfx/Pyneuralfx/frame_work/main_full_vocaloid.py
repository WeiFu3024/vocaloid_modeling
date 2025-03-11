import os 
import sys 
sys.path.append('')
sys.path.append('..')
import torch
import numpy as np 

from shutil import copyfile

import solver 
import utils
from dataset_vocaloid import Full_Modeling_Vocaloid_AudioDataset, Inference_Vocaloid_AudioDataset

# ============================================================ #
# Config 
# ============================================================ #
# Load config from yaml files 
cmd = {
    'config': './configs/cnn/tcn/film_tcn.yml'
}

args = utils.load_config(cmd['config'])
print(' > config:', cmd['config'])

# loss functions 
customized_loss_func = None
# ============================================================ #
# You can customize your own loss function here 
# ============================================================ #
if args.loss.loss_func == 'customized': 
    customized_loss_func = None

loss_func_tra = utils.setup_loss_funcs(args, customized_loss_func) 
loss_func_val = utils.setup_loss_funcs(args, customized_loss_func) 
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
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.train.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.train.lr_patience, verbose=True)


# to device
model.to(args.device)
for func in loss_funcs:
    func.to(args.device)

# ============================================================ #
# Functions
# ============================================================ #
from torch.nn.utils.rnn import pad_sequence

import os
import shutil
import random

import os
import shutil
import random

def shuffle_and_distribute_paired_files(train_path, valid_path, test_path, ratio=(0.7, 0.15, 0.15)):
    # Ensure the ratio sums to 1
    assert sum(ratio) == 1.0, "Ratio values must sum to 1."

    parent_dir = os.path.dirname(train_path)  # Get the common parent directory

    # Collect all unique file names (assuming both x/ and y/ have the same file names)
    all_files = set()
    for base_path in [train_path, valid_path, test_path]:
        x_path = os.path.join(base_path, "x")
        if os.path.exists(x_path):
            all_files.update(os.listdir(x_path))  # Collect file names from x/

    all_files = list(all_files)
    random.shuffle(all_files)  # Shuffle file names

    # Compute split indices based on ratio
    total_files = len(all_files)
    train_split = int(ratio[0] * total_files)
    valid_split = train_split + int(ratio[1] * total_files)

    # Temporary directories to store shuffled data before moving them back
    temp_train_x = os.path.join(parent_dir, "temp_train_x")
    temp_train_y = os.path.join(parent_dir, "temp_train_y")
    temp_valid_x = os.path.join(parent_dir, "temp_valid_x")
    temp_valid_y = os.path.join(parent_dir, "temp_valid_y")
    temp_test_x = os.path.join(parent_dir, "temp_test_x")
    temp_test_y = os.path.join(parent_dir, "temp_test_y")

    os.makedirs(temp_train_x, exist_ok=True)
    os.makedirs(temp_train_y, exist_ok=True)
    os.makedirs(temp_valid_x, exist_ok=True)
    os.makedirs(temp_valid_y, exist_ok=True)
    os.makedirs(temp_test_x, exist_ok=True)
    os.makedirs(temp_test_y, exist_ok=True)

    # Move files to temporary shuffled directories
    for idx, file_name in enumerate(all_files):
        if idx < train_split:
            dest_x, dest_y = temp_train_x, temp_train_y
        elif idx < valid_split:
            dest_x, dest_y = temp_valid_x, temp_valid_y
        else:
            dest_x, dest_y = temp_test_x, temp_test_y

        # Move files from the original directories to the temporary directories
        for base_path in [train_path, valid_path, test_path]:
            src_x = os.path.join(base_path, "x", file_name)
            src_y = os.path.join(base_path, "y", file_name)

            if os.path.exists(src_x):
                shutil.move(src_x, os.path.join(dest_x, file_name))
            if os.path.exists(src_y):
                shutil.move(src_y, os.path.join(dest_y, file_name))

    # Clear original directories before moving files back
    for base_path in [train_path, valid_path, test_path]:
        shutil.rmtree(os.path.join(base_path, "x"), ignore_errors=True)
        shutil.rmtree(os.path.join(base_path, "y"), ignore_errors=True)
        os.makedirs(os.path.join(base_path, "x"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "y"), exist_ok=True)

    # Move shuffled files back to original directories
    for temp_dir, target_dir in [
        (temp_train_x, os.path.join(train_path, "x")),
        (temp_train_y, os.path.join(train_path, "y")),
        (temp_valid_x, os.path.join(valid_path, "x")),
        (temp_valid_y, os.path.join(valid_path, "y")),
        (temp_test_x, os.path.join(test_path, "x")),
        (temp_test_y, os.path.join(test_path, "y")),
    ]:
        for file_name in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, file_name), os.path.join(target_dir, file_name))

        shutil.rmtree(temp_dir)  # Remove temporary directory

    print(f"Shuffled and distributed {total_files} file pairs:")
    print(f"- Train: {train_split} files ({ratio[0] * 100:.1f}%)")
    print(f"- Valid: {valid_split - train_split} files ({ratio[1] * 100:.1f}%)")
    print(f"- Test: {total_files - valid_split} files ({ratio[2] * 100:.1f}%)")

def collate_fn(batch):
    wav_x_s = []
    wav_y_s = []
    cond_s = []

    for idx in range(len(batch)):
        wav_x, wav_y, cond = batch[idx]
        wav_x_s.append(wav_x[None, ...])
        wav_y_s.append(wav_y[None, ...])
        cond_s.append([cond])

    x_final = np.concatenate(wav_x_s, axis=0)
    y_final = np.concatenate(wav_y_s, axis=0)
    c_final = np.concatenate(cond_s, axis=0)

    return torch.from_numpy(x_final), torch.from_numpy(y_final), torch.from_numpy(c_final)



def inference(path_to_dataset, path_savedir, exp_dir_val):
    global model
    print(' >>>>> inference')
    print(' [data]      dataset:', path_to_dataset)

    # load model 
    model = utils.load_model(
                exp_dir_val,
                model,
                device=args.device, 
                name='best_params.pt')
    
    path_to_x = os.path.join(path_to_dataset, 'x')
    path_to_y = os.path.join(path_to_dataset, 'y')

    filelist_x = os.listdir(path_to_x)
    filelist_y = os.listdir(path_to_y)
    
    # data
    for (fn_x, fn_y) in zip(filelist_x, filelist_y):
        valid_set = Inference_Vocaloid_AudioDataset(
            os.path.join(path_to_x, fn_x),
            os.path.join(path_to_y, fn_y),
            pre_room=PRE_ROOM,
            win_len=args.data.buffer_size,
            norm_tensor=args.data.norm_tensor,
            sr=args.data.sampling_rate,
            cond_size=args.data.num_conds)

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
            path_save=path_outdir,
            concat=True)
        
    amount, amount_train = model.compute_num_of_params()
    print(' > params amount: {:,d} | trainable: {:,d}'.format(amount, amount_train))

def train():
    global model

    if LOAD_DIR:
        print(' >>>>> fine-tuning')
        model = utils.load_model(
                LOAD_DIR,
                model,
                device=args.device, 
                name='best_params.pt')
    else:
        print(' >>>>> training')

    # datasets
    
    train_set = Full_Modeling_Vocaloid_AudioDataset(
        args.data.train_path, 
        pre_room=PRE_ROOM,
        win_len=args.data.buffer_size, 
        norm_tensor=args.data.norm_tensor,
        sr=args.data.sampling_rate,
        cond_size=args.data.num_conds)
    
    loader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print('> train dataset ready ...........')

    
    valid_set = Full_Modeling_Vocaloid_AudioDataset(
        args.data.valid_path, 
        pre_room=PRE_ROOM,
        win_len=args.data.buffer_size,
        norm_tensor=args.data.norm_tensor,
        sr=args.data.sampling_rate,
        cond_size=args.data.num_conds)
    
    
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



# ============================================================ #
# Main  
# ============================================================ #
utils.check_configs(args)
# Example usage
shuffle_and_distribute_paired_files(args.data.train_path, args.data.valid_path, args.data.test_path, ratio=(0.8, 0.1, 0.1))  # Adjust ratio as needed
train()
inference(args.data.test_path, 'valid_gen', args.env.expdir)

