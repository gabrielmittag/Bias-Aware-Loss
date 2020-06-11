import time
import copy
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from PIL import Image
import numpy as np
import pandas as pd
from skimage.filters import gaussian
from skimage.util import img_as_ubyte
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
    'early_stop': 100, # early stopping on validation Pearson's correlation
    'num_workers': 0, # number of workers of DataLoaders
    
    'model_name': 'resnet18',
    'pretrained': False, # image net pretraining
    'augmented': True, # random crop training images
    'bias_fun': 'first_order', # either "first_order" or "third_order" bias estimation
    'r_th': 0.7, # minimum correlation on training set before first bias estimation
    'anchor_db': 'train_1', # string with dataset name to which biases should be anchored / None if no anchoring used
    'mse_weight': 0.0, # weight of "vanilla MSE loss" added to bias loss
    }

main_folder = './'
dataset_folder = './LIVE IQA R2/refimgs'
results_subfolder = 'results'
images_mixed = False # if True, the same reference images are used in different datasets

plot_sigma2mos_mapping = True # show mapping between sigma and mos
plot_images = True # plot 10 images with applied bluriness
plot_biases = True # plot artifically introduced biases
plot_every_epoch = True  # show training process and bias estimation every epoch
plot_final_results = True # show final results

# Artificially introduced biases
b = np.array([
    [0,  1, 0,  0],
    [0.5,  0.5, 0,  0],
    [3, 0.3,  0, 0],
    [-2.3,  5.03133896, -1.6883704 ,0.19968759]])

#%% ---- LOAD CSV / SIMULATE MOS ----------------------------------------------

# Runname and savepath 
runname = datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")
resultspath = os.path.join(main_folder, results_subfolder)
if not os.path.exists(resultspath):
    os.makedirs(resultspath)
resultspath = os.path.join(resultspath, runname)

# Load dataset csv files. If True use same reference images in different datasets
if images_mixed:
    dfile_train = pd.read_csv(os.path.join(main_folder, 'iqb_train_mixed.csv'))
else:
    dfile_train = pd.read_csv(os.path.join(main_folder, 'iqb_train.csv'))
dfile_val = pd.read_csv(os.path.join(main_folder, 'iqb_val.csv'))

# Map the bluriness factor sigma to simulated MOS values
def sigma2mos(sigma):
    sigma_min = 1
    sigma_max = 3
    mos = (sigma-sigma_min) * 100 / (sigma_max-sigma_min)
    mos = -mos+100
    mos = 1 + 0.035*mos+mos*(mos-60)*(100-mos)*7e-6      
    mos = mos.clip(min=1).reshape(-1,1).astype('float32')
    return mos
dfile_train['mos'] = sigma2mos( dfile_train['sigma'].to_numpy() )
dfile_val['mos'] = sigma2mos( dfile_val['sigma'].to_numpy() )

# Get unique dataset names and apply artifical bias 
def calc_mapped(x,b):
    if b.ndim==1:
        x = b[0] + x * b[1] + x**2 * b[2] + x**3 * b[3]
    elif b.ndim==2:
        x = b[:,0] + x * b[:,1] + x**2 * b[:,2] + x**3 * b[:,3]
    else:
        raise ValueError
    return x
train_dbs = dfile_train.db.unique()
val_dbs = dfile_val.db.unique()
for i, db in enumerate(dfile_train.db.unique()):
    dfile_train.loc[dfile_train.db==db, 'mos'] = calc_mapped( 
        dfile_train.loc[dfile_train.db==db, 'mos'], b[i] ) 

#%% ---- DATASET AND PLOTS ----------------------------------------------------

# Dataset to load images with index that is used to assign samples to dataset during training
class ImageBlurIdxDataset(Dataset):
    '''
    ImageBlurIdxDataset class. 
    
    Loads images, applies bluriness, loads them to RAM. Outputs image, MOS, 
    and index, which is needed to assign the images to their corresponding
    dataset.
    '''
    def __init__(self, main_dir, df, augmented=False):
        self.df = df
        self.main_dir = main_dir
        self.augmented = augmented
        self._get_transform()
        self._load_images()
    def _load_images(self):        
        self.images = []
        for index, row in self.df.iterrows():
            image = np.asarray(Image.open( os.path.join(self.main_dir, row['src_image']) )) 
            image = gaussian(image, sigma=row['sigma'], multichannel=True)
            image = img_as_ubyte( image.clip(min=-1, max=1) )      
            self.images.append(image)
        self.mos = self.df['mos'].to_numpy().reshape(-1,1).astype('float32')    
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
    
ds_train = ImageBlurIdxDataset(dataset_folder, dfile_train, augmented=opts['augmented'])
ds_val = ImageBlurIdxDataset(dataset_folder, dfile_val, augmented=False)

# plot the mapping of bluriness sigma to MOS
if plot_sigma2mos_mapping:
    x = np.linspace(1,3,1000)
    y = sigma2mos(x)
    plt.figure(figsize=(3.0, 3.0))
    plt.plot(x,y)
    plt.xlabel('$\sigma$')
    plt.ylabel('MOS')
    plt.yticks(np.arange(1,5,0.5))
    plt.show()

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

    
# plot the artifical biases applied to the datasets
if plot_biases: 
    plt.figure(figsize=(3.0, 3.0))
    x = np.linspace(1,5,100)
    for i in range(len(b)):
        y = calc_mapped(x, b[i])
        plt.plot(x,y)
        plt.axis([1,4.5,1,4.5])
    plt.xlabel('Artificial biases')
    plt.xlabel('MOS')
    plt.ylabel('Biased MOS')
    plt.yticks(np.arange(1,5,0.5))
    plt.xticks(np.arange(1,5,0.5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

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
        target_mos='mos',
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
print('running on:')
print(dev)
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
            y_db = dfile_train.loc[dfile_train.db==db, 'mos'].to_numpy()
            y_hat_db = dfile_train.loc[dfile_train.db==db, 'mos_hat'].to_numpy()
            b_db = np.vstack(dfile_train.loc[dfile_train.db==db, 'b'])[0]
            y_est = calc_mapped(x, b_db)
            y_orig = calc_mapped(x, b[i])
            plt.subplot(1,5,i+1)
            plt.plot(y_hat_db, y_db, 'o', markersize=2)
            plt.plot(x, y_est)
            plt.plot(x, y_orig)
            plt.yticks(np.arange(-10,10))        
            plt.axis([y_hat_db.min().clip(max=1),y_hat_db.max().clip(min=5),y_db.min().clip(max=1),y_db.max().clip(min=5)])
            plt.title(db)
        y_est = calc_mapped(x, np.array([0,  1, 0,  0]))
        y_orig = calc_mapped(x, np.array([0,  1, 0,  0]))
        plt.subplot(1,5,5)
        plt.plot(y_val_hat, y_val, 'o', markersize=2)
        plt.plot(x, y_est)
        plt.plot(x, y_orig)
        plt.yticks(np.arange(-10,10))
        plt.axis([y_val_hat.min().clip(max=1),y_val_hat.max().clip(min=5),y_val.min().clip(max=1),y_val.max().clip(min=5)])
        plt.title('val')        
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
    plt.figure(figsize=(12, 3.5))
    dfile_train['mos_hat'] = y_train_hat
    bias_loss.update_bias(y_train, y_train_hat)
    dfile_train['b'] = bias_loss.b.tolist()
    for i, db in enumerate(dfile_train.db.unique()):
        y_db = dfile_train.loc[dfile_train.db==db, 'mos'].to_numpy()
        y_hat_db = dfile_train.loc[dfile_train.db==db, 'mos_hat'].to_numpy()
        b_db = np.vstack(dfile_train.loc[dfile_train.db==db, 'b'])[0]
        y_est = calc_mapped(x, b_db)
        y_orig = calc_mapped(x, b[i])
        plt.subplot(1,5,i+1)
        plt.plot(y_hat_db, y_db, 'o', markersize=2)
        plt.plot(x, y_est)
        plt.plot(x, y_orig)
        plt.xticks(np.arange(-10,10))
        plt.yticks(np.arange(-10,10))
        plt.axis([y_hat_db.min().clip(max=1),y_hat_db.max().clip(min=5),y_db.min().clip(max=1),y_db.max().clip(min=5)])
        plt.xlabel('Predicted MOS')
        plt.ylabel('Subjective MOS')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(db)
    y_est = calc_mapped(x, np.array([0,  1, 0,  0]))
    y_orig = calc_mapped(x, np.array([0,  1, 0,  0]))
    plt.subplot(1,5,5)
    plt.plot(y_val_hat, y_val, 'o', markersize=2)
    plt.plot(x, y_est)
    plt.plot(x, y_orig)
    plt.xticks(np.arange(-10,10))
    plt.yticks(np.arange(-10,10))
    plt.axis([y_val_hat.min().clip(max=1),y_val_hat.max().clip(min=5),y_val.min().clip(max=1),y_val.max().clip(min=5)])
    plt.xlabel('Predicted MOS')
    plt.ylabel('Subjective MOS')
    plt.title('val final')        
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()     
        
