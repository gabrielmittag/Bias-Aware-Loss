import numpy as np
import torch
from scipy.stats import pearsonr
from scipy.optimize import least_squares
from scipy.optimize import minimize

class biasLoss(object):
    '''
    Bias loss class. Calculates loss while considering database biases.
    
    biasLoss should be initialized before training with a pandas series of 
    dataset names "db" for each training sample. Use the "get_loss()" function
    to calculate the loss during training and "update_bias()" to update the 
    biases after each epoch.
    
    Initalizing input argurments:
    - db (pandas series): dataset names for each training sample.
    - anchor_db (string): anchors the biases to the samples of this dataset
    - mapping (string): use either "first_order" or "third_order" mapping
    - r_th (float): minimum Pearson's correlation between predicted and 
        subjective MOS. When this threshold is met the bias will be updated.
    - mse_weight (float): use either "first_order" or "third_order" mapping
    
    Calculate loss during training:
        Use get_loss(yb, yb_hat, idx), where "yb" are the target values of the 
        mini-batch and "yb_hat" the predicted values. "idx" are the indices
        of those mini-batch samples.
        
    Update bias:
        Use update_bias(y, y_hat) after each epoch, where "y" are all target
        values and "y_hat" are all predicted values.
    
    '''    
    def __init__(self, db, anchor_db=None, mapping='first_order', r_th=0.7, mse_weight=0.0):
        
        self.db = db
        self.mapping = mapping
        self.r_th = r_th
        self.anchor_db = anchor_db
        self.mse_weight = mse_weight
        
        self.b = np.zeros((len(db),4))
        self.b[:,1] = 1
        self.do_update = False
        
        if anchor_db:
            if not (self.anchor_db in self.db.unique()):
                raise ValueError('anchor_db not found in dataset list')
            
    def get_loss(self, yb, yb_hat, idx):
        b = torch.tensor(self.b, dtype=torch.float).to(yb_hat.device)
        b = b[idx,:]
        yb_hat_map = (b[:,0]+b[:,1]*yb_hat[:,0]+b[:,2]*yb_hat[:,0]**2+b[:,3]*yb_hat[:,0]**3).view(-1,1)
        loss_bias = torch.mean( (yb_hat_map-yb)**2 )   
        loss_normal = torch.mean( (yb_hat-yb)**2 )   
        loss = loss_bias + self.mse_weight * loss_normal
        return loss
    
    def update_bias(self, y, y_hat):
        
        # update only if minimum correlation r_th is met
        if not self.do_update:
            r = pearsonr(y.reshape(-1), y_hat.reshape(-1))[0]
            if r>self.r_th:
                self.do_update = True
            else:
                print('--> bias not updated. r: {:0.2f}'.format(r))
            
        if self.do_update:
            print('--> bias updated')
            for db_name in self.db.unique():
                
                db_idx = (self.db==db_name).to_numpy().nonzero()
                y_hat_db = y_hat[db_idx].astype('float64')
                y_db = y[db_idx].astype('float64')
                
                if self.mapping=='first_order':
                    b_db = self._calc_bias_first_order(y_hat_db, y_db, bounds=False)
                elif self.mapping=='third_order':
                    b_db = self._calc_bias_third_order(y_hat_db, y_db, bounds=False)
                else:
                    raise NotImplementedError
                                
                if not db_name==self.anchor_db:
                    self.b[db_idx,:len(b_db)] = b_db       
                
    def _calc_bias_first_order(self, y_hat, y, bounds=None):
        if bounds:
            def fun(p, x, y):
                return (p[0] + p[1] * x) - y
            x0 = np.array([0, 1])
            res_1 = least_squares(fun, x0, args=(y_hat, y), bounds=([-1.2, 0.2], [3.5, 1.1]))
            b = np.zeros((4))
            b[0:2] = res_1.x     
        else:
            def fun(p, x, y):
                return (p[0] + p[1] * x) - y
            x0 = np.array([0, 1])
            res_1 = least_squares(fun, x0, args=(y_hat, y), bounds=([-np.inf, 0], [np.inf, np.inf]))
            b = np.zeros((4))
            b[0:2] = res_1.x   
        return b
        
    def _calc_bias_third_order(self, x, y, bounds=None, min_val=1, max_value=5):
        constr_step = 0.001
        def polynomial(p, x):
            return p[0]+p[1]*x+p[2]*x**2+p[3]*x**3
        def constraint_1st_der(p):
            x_1st = np.arange(min_val, max_value, constr_step)
            return p[1]+2*p[2]*x_1st+3*p[3]*x_1st**2
        def objective(p):
            x_map = polynomial(p, x)
            err = x_map-y
            return (err**2).sum()
        if bounds:
            bnds = ((-5, 5), (-5, 5), (-5, 5), (-5, 5))
        else:
            bnds = ((None, None), (None, None), (None, None), (None, None))
        cons = dict(type='ineq', fun=constraint_1st_der)
        res = minimize(objective, x0=np.array([0., 1., 0., 0.]), method='SLSQP', constraints=cons, bounds=bnds)
        return res.x                
                
