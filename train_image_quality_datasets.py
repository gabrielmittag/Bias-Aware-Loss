import time
import copy
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from PIL import Image
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.models as models
from biasLoss import biasLoss

#%% ---- TRAINING OPTIONS  ----------------------------------------------------

# Main training options
opts = {
    'lr': 0.0001, # learning rate
    'bs': 32, # mini-batch size
    'epochs': 1000, # maximum training epochs
    'early_stop': 10, # early stopping on validation Pearson's correlation
    'num_workers': 0, # number of workers of DataLoaders
    
    'model_name': 'resnet50',
    'pretrained': True, # image net pretraining
    'augmented': True, # random crop training images
    'bias_fun': 'first_order', # either "first_order" or "third_order" bias estimation
    'r_th': 0.7, # minimum correlation on training set before first bias estimation
    'anchor_db': 'tid_2013', # string with dataset name to which biases should be anchored / None if no anchoring used
    'mse_weight': 0.0, # weight of "vanilla MSE loss" added to bias loss
    }

main_folder = './'
dataset_folder = 'D:/image_quality_datasets/' # Folder with image quality datasets (see readme)
results_subfolder = 'results'

plot_images = True # plot 10 random images
plot_every_epoch = True  # show training process and bias estimation every epoch
plot_final_results = True # show final results

# train and val split
train_dbs = [
    'tid_2013',
    'live_md',
    'live_iqa',
    'live_challenge',
    ]

val_dbs = ['csiq']

#%% ---- LOAD CSV / USE ONE DATASET FOR VALIDATION ----------------------------

# Runname and savepath 
runname = datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")
resultspath = os.path.join(main_folder, results_subfolder)
if not os.path.exists(resultspath):
    os.makedirs(resultspath)
resultspath = os.path.join(resultspath, runname)

# Load dataset csv files.
dfile = pd.read_csv(os.path.join(main_folder, 'image_datasets.csv'))
dfile_train = dfile[dfile.db.isin(train_dbs)].reset_index(drop=True)
dfile_val = dfile[dfile.db.isin(val_dbs)].reset_index(drop=True)

#%% ---- DATASET AND PLOTS ----------------------------------------------------

# Dataset to load images with index that is used to assign samples to dataset during training
class ImageIdxDataset(Dataset):
    '''
    ImageIdxDataset class. 
    
    Loads images, loads them to RAM. Outputs image, MOS, and index, which is 
    needed to assign the images to their corresponding dataset.
    '''
    def __init__(self, main_dir, df, augmented=False):
        self.df = df
        self.main_dir = main_dir
        self.augmented = augmented
        self._get_transform()
        self._load_images()
    def _load_images(self):        
        self.images = [np.asarray(Image.open( os.path.join(self.main_dir, file_name) )) for file_name in self.df.deg_file]
        self.mos = self.df.mos_norm.to_numpy().reshape(-1,1).astype('float32')    
    def _get_transform(self):
        if self.augmented:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])  
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])  
            ])
    def __getitem__(self, index):
        img = self.images[index]
        img = self.transform(img)      
        return img, self.mos[index], index
    def __len__(self):
        return len(self.images)
    
print('Loading datasets to RAM ...')
ds_train = ImageIdxDataset(dataset_folder, dfile_train, augmented=opts['augmented'])
ds_val = ImageIdxDataset(dataset_folder, dfile_val, augmented=False)
print('--> done!')

# Plot 10 random validation images with their MOS value
if plot_images:
    random_index = np.random.choice(len(ds_val), 10, replace=False)
    for random_index in random_index:
        x, y, idx = ds_val[random_index]
        inp = x.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        plt.grid(None) 
        plt.axis('off')
        plt.title('MOS: {:0.2f}'.format(y[0]))
        plt.show()

def calc_mapped(x,b):
    if b.ndim==1:
        x = b[0] + x * b[1] + x**2 * b[2] + x**3 * b[3]
    elif b.ndim==2:
        x = b[:,0] + x * b[:,1] + x**2 * b[:,2] + x**3 * b[:,3]
    else:
        raise ValueError
    return x

#%% ---- MODEL AND EVALUATION FUNCTION ----------------------------------------
     
# Select model 
if opts['model_name']=='resnet18':
    model = models.resnet18(pretrained=opts['pretrained'])
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
elif opts['model_name']=='resnet50':
    model = models.resnet50(pretrained=opts['pretrained'])
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) 
elif opts['model_name']=='resnet101':
    model = models.resnet101(pretrained=opts['pretrained'])
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)          
else:
    raise NotImplementedError

# Model evaluation function
def eval_model(
        model,
        ds, 
        target_mos='mos_norm',
        do_plot=False,
        do_print=False,
        bs=16,
        num_workers=0):
    
    # Dataloader without shuffling
    dl = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers)
        
    # Get predictions
    model.eval()
    with torch.no_grad():
        y_hat = [model(xb.to(dev)).cpu().detach().numpy() for xb, yb, idx in dl]
    y_hat = np.concatenate(y_hat).reshape(-1)
    y = ds.df[target_mos].to_numpy().reshape(-1)
        
    # Evaluate each database
    results_db = []
    for db_name in ds.df.db.unique():
        idx_db = (ds.df.db==db_name).to_numpy().nonzero()[0]
        y_hat_db = y_hat[idx_db]
        y_db = y[idx_db]
        rmse = np.sqrt( np.mean( (y_hat_db-y_db)**2 ) )      
        r = pearsonr(y_db.reshape(-1), y_hat_db.reshape(-1))[0]
        results_db.append({'db': db_name, 'r': r, 'rmse': rmse})
        
        # Plot
        if do_plot:
            plt.figure(figsize=(5.0, 5.0))
            plt.clf()
            plt.plot(y_hat_db, y_db, 'o', label='Original data', markersize=5)
            plt.plot([0, 5], [0, 5], 'k')
            plt.axis([1, 5, 1, 5])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.xticks(np.arange(1, 6))
            plt.yticks(np.arange(1, 6))
            plt.title(db_name)
            plt.ylabel('Subjective MOS')
            plt.xlabel('Predicted MOS')
            plt.show()
            
        # Print
        if do_print:
            print('%-30s r: %0.2f, rmse: %0.2f'
                  % (db_name+':', r, rmse))
            
    results_db = pd.DataFrame(results_db)
    results = {
            'r': results_db.r.to_numpy().mean(), 
            'rmse': results_db.rmse.to_numpy().mean(),
            }
    return results, y, y_hat

#%% --- TRAINING LOOP --------------------------------------------------------

# Load biasLoss class
bias_loss = biasLoss( 
    ds_train.df.db, 
    anchor_db=opts['anchor_db'], 
    mapping=opts['bias_fun'], 
    r_th=opts['r_th'],
    mse_weight=opts['mse_weight'],
    )

dl_train = DataLoader(
    ds_train,
    batch_size=opts['bs'],
    shuffle=True,
    drop_last=True,
    num_workers=opts['num_workers'])

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(dev)
opt = optim.Adam(model.parameters(), lr=opts['lr'])
best_model_wts = copy.deepcopy(model.state_dict())

# Ini early stopping
best_r = 0   
es = 0

tic_overall = time.time()
print('--> start training')
results = []
for epoch in range(1,opts['epochs']+1):
    tic_epoch = time.time()
    
    # Optimize model weights
    k = 0
    loss = 0.0
    model.train()
    for xb, yb, idx in dl_train:
        yb = yb.to(dev)
        yb_hat = model(xb.to(dev))
        lossb = bias_loss.get_loss(yb, yb_hat, idx)
        lossb.backward()
        opt.step()
        opt.zero_grad()
        loss += lossb.item()
        k += 1
    loss = loss/k
    
    # Evaluate after each epoch
    results_train, y_train, y_train_hat = eval_model(model, ds_train, do_plot=False, do_print=False)
    results_val, y_val, y_val_hat = eval_model(model, ds_val, do_plot=False, do_print=False)
    
    # Update bias for loss 
    bias_loss.update_bias(y_train, y_train_hat)
    
    # Plot
    if plot_every_epoch:
        x = np.linspace(-10,20,100)
        plt.figure(figsize=(12, 2))
        dfile_train['mos_hat'] = y_train_hat
        dfile_train['b'] = bias_loss.b.tolist()
        for i, db in enumerate(dfile_train.db.unique()):
            y_db = dfile_train.loc[dfile_train.db==db, 'mos_norm'].to_numpy()
            y_hat_db = dfile_train.loc[dfile_train.db==db, 'mos_hat'].to_numpy()
            b_db = np.vstack(dfile_train.loc[dfile_train.db==db, 'b'])[0]
            y_est = calc_mapped(x, b_db)
            plt.subplot(1,5,i+1)
            plt.plot(y_hat_db, y_db, 'o', markersize=2)
            plt.plot([0, 5], [0, 5], 'k')
            plt.plot(x, y_est)
            plt.yticks(np.arange(-10,10))        
            plt.axis([y_hat_db.min().clip(max=1),y_hat_db.max().clip(min=5),y_db.min().clip(max=1),y_db.max().clip(min=5)])
            plt.title('train: ' + db)
        plt.subplot(1,5,5)
        plt.plot(y_val_hat, y_val, 'o', markersize=2)
        plt.plot([0, 5], [0, 5], 'k')
        plt.yticks(np.arange(-10,10))
        plt.axis([y_val_hat.min().clip(max=1),y_val_hat.max().clip(min=5),y_val.min().clip(max=1),y_val.max().clip(min=5)])
        plt.title('val: ' + val_dbs[0])        
        plt.show()              

    # Early stopping
    if results_val['r'] > best_r:
        best_r = results_val['r']
        best_model_wts = copy.deepcopy(model.state_dict()) 
        es = 0
    else:
        es+=1  
        if es>=opts['early_stop']:
            break
     
    # Print results
    toc_epoch = time.time() - tic_epoch
    print('epoch {}, runtime {:.2f}s, loss {:.3f}, r_train_mean {:.3f}, rmse_val {:.3f}, r_val {:.3f}'.format(
        epoch, toc_epoch, loss, results_train['r'], results_val['rmse'], results_val['r']) )

    # Save results history
    results.append({
        'runname': runname,
        'epoch': epoch,
        **opts,
        'train_dbs': train_dbs,
        'val_dbs': val_dbs,
        **results_val,
        })
    pd.DataFrame(results).to_csv(resultspath+'__results.csv', index=False)
    
#%% --- EVALUATE BEST MODEL ---------------------------------------------------
print('training finished!')
model.load_state_dict(best_model_wts)
results_train, y_train, y_train_hat = eval_model(model, ds_train, do_print=True)
results_val, y_val, y_val_hat = eval_model(model, ds_val, do_print=True)
toc_overall = time.time() - tic_overall
print('epochs {}, runtime {:.0f}s, rmse_val {:.3f}, r_val {:.3f}'.format(epoch+1, toc_overall, results_val['rmse'], results_val['r']) )

# Plot
if plot_final_results:
    x = np.linspace(-10,20,100)
    plt.figure(figsize=(12, 2))
    dfile_train['mos_hat'] = y_train_hat
    bias_loss.update_bias(y_train, y_train_hat)
    dfile_train['b'] = bias_loss.b.tolist()
    for i, db in enumerate(dfile_train.db.unique()):
        y_db = dfile_train.loc[dfile_train.db==db, 'mos_norm'].to_numpy()
        y_hat_db = dfile_train.loc[dfile_train.db==db, 'mos_hat'].to_numpy()
        b_db = np.vstack(dfile_train.loc[dfile_train.db==db, 'b'])[0]
        y_est = calc_mapped(x, b_db)
        plt.subplot(1,5,i+1)
        plt.plot(y_hat_db, y_db, 'o', markersize=2)
        plt.plot([0, 5], [0, 5], 'k')
        plt.plot(x, y_est)
        plt.yticks(np.arange(-10,10))        
        plt.axis([y_hat_db.min().clip(max=1),y_hat_db.max().clip(min=5),y_db.min().clip(max=1),y_db.max().clip(min=5)])
        plt.title('train: ' + db)
    plt.subplot(1,5,5)
    plt.plot(y_val_hat, y_val, 'o', markersize=2)
    plt.plot([0, 5], [0, 5], 'k')
    plt.yticks(np.arange(-10,10))
    plt.axis([y_val_hat.min().clip(max=1),y_val_hat.max().clip(min=5),y_val.min().clip(max=1),y_val.max().clip(min=5)])
    plt.title('val: ' + val_dbs[0])        
    plt.show()  
