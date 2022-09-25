#! /usr/bin/env python
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from plnn.modules import Flatten
from plnn.proxlp_solver.utils import LinearOp, ConvOp, BatchConvOp, BatchLinearOp

'''
Training test file for the deep model transferrability studies
1. added norm weight in the backward pass
2. has been tested on wide and deep networks --- satisfactory performances

Currently treated as the correct model 
'''

class LayerUpdate(nn.Module):
    '''
    this class updates layer one time
    '''
    def __init__(self, p):
        super(LayerUpdate, self).__init__()
        self.p = p

        #inputs
        self.inp_f = nn.Linear(3,p)
        self.inp_f_1 = nn.Linear(p,p)

        # for activation nodes
        self.fc1 = nn.Linear(6, p)
        self.fc1_1 = nn.Linear(p, p)
        self.fc3 = nn.Linear(2*p, p)     
        self.fc3_2 = nn.Linear(2*p, p)  
        self.fc4 = nn.Linear(2*p, p)
        self.fc4_2 = nn.Linear(p, p)

        # outputs
        self.out1 = nn.Linear(4, p)
        self.out2 = nn.Linear(2*p, p)
        self.out3 = nn.Linear(p, p)


    def forward(self, lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, mu):

        # NOTE: All bounds should be at the same size as the layer outputs
        #       We have assumed that the last property layer is linear       

        batch_size = len(lower_bounds_all[0])
        p = self.p

        ## FORWARD PASS
        # for the forward pass the first layer is not updated
        #print(th.cat([lower_bounds_all[0].unsqueeze(-1),upper_bounds_all[0].unsqueeze(-1)],1))

        # first, deal with the input layer
        inp = th.cat([lower_bounds_all[0].view(-1).unsqueeze(-1),
                        primal_inputs.view(-1).unsqueeze(-1),
                        upper_bounds_all[0].view(-1).unsqueeze(-1)],1)
        temp = self.inp_f_1(F.relu(self.inp_f(inp)))
        # mu[0] = temp.reshape(mu[0].size())
        mu[0] = temp.reshape(batch_size, temp.shape[0]//batch_size, temp.shape[1])

        relu_count_idx = 0
        out_features = [-1]+ th.tensor(lower_bounds_all[0][0].size()).tolist()

        idx = 1
        for layer_idx, layer in enumerate(layers['fixed_layers']):

            if type(layer) in [BatchConvOp, ConvOp, nn.Conv2d]:
                if type(layer) in [BatchConvOp, ConvOp]:
                    layer_weight = layer.weights
                else:
                    layer_weight = layer.weight

                if type(layer) is BatchConvOp:
                    layer_bias = layer.unconditioned_bias.detach().view(-1)
                elif type(layer) is ConvOp:
                    layer_bias = layer.bias.view(-1)
                else:
                    layer_bias = layer.bias
                #reshape 
                mu_inp = th.cat([i for i in mu[relu_count_idx]], 1)
                mu_inp = th.t(mu_inp).reshape(out_features)
                nb_features_pre = F.conv2d(mu_inp, layer_weight , bias=None,
                                        stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
                # record and transfer back
                out_features = th.tensor(nb_features_pre.size()).tolist()
                nb_features_temp = nb_features_pre.reshape(out_features[0],-1)
                #import pdb; pdb.set_trace()
                nb_features_temp = th.cat([nb_features_temp[i*p:(1+i)*p] for i in range(batch_size)], 1)
                nb_features_temp = th.t(nb_features_temp)
                pre_layer_bias = layer_bias.unsqueeze(1).expand(out_features[1],out_features[2]*out_features[3])
                pre_layer_bias = pre_layer_bias.reshape(-1)
                pre_layer_bias = pre_layer_bias.repeat(batch_size)
                layer_lower_pre = lower_bounds_all[idx].view(-1)
                layer_upper_pre = upper_bounds_all[idx].view(-1)
                idx += 1
                #import pdb; pdb.set_trace()

            elif type(layer) in [nn.Linear, LinearOp, BatchLinearOp]: 
                if type(layer) in [LinearOp, BatchConvOp]:
                    layer_weight = layer.weights
                else:
                    layer_weight = layer.weight
                nb_features_temp = layer_weight @ mu[relu_count_idx]
                nb_features_temp = th.cat([i for i in nb_features_temp], 0)
                pre_layer_bias = layer.bias.repeat(batch_size)
                out_features = [-1, nb_features_temp.size()[0]]
                layer_lower_pre = lower_bounds_all[idx].view(-1)
                layer_upper_pre = upper_bounds_all[idx].view(-1)
                idx += 1

            elif type(layer) is nn.ReLU:
            # node features
                ratio_0, ratio_1, beta, ambi_mask = compute_ratio(layer_lower_pre, layer_upper_pre)
                #import pdb; pdb.set_trace()

                # measure relaxation
                layer_n_bounds = th.cat([beta.unsqueeze(-1), 
                                            layer_lower_pre.unsqueeze(-1),
                                            layer_upper_pre.unsqueeze(-1), 
                                            dual_vars[relu_count_idx][:,0].unsqueeze(-1),
                                            dual_vars[relu_count_idx][:,1].unsqueeze(-1),
                                            pre_layer_bias.unsqueeze(-1)],1)
                layer_relax_s1 = self.fc1_1(F.relu(self.fc1(layer_n_bounds)))
                layer_relax = layer_relax_s1 * ambi_mask.unsqueeze(-1)

                #import pdb; pdb.set_trace()
                #print('layer relax forward: ', th.max(abs(layer_relax)))


                # feature updates
                nb_features_input = th.cat([nb_features_temp*ratio_0.unsqueeze(-1), nb_features_temp*ratio_1.unsqueeze(-1)],1)
                layer_nb_features_temp = F.relu(self.fc3(nb_features_input))
                layer_nb_features = th.cat([layer_nb_features_temp, mu[relu_count_idx+1].reshape(layer_nb_features_temp.size())], dim=1)
                layer_nb_features = self.fc3_2(layer_nb_features)

                # update all nodes in a layer
                layer_input = th.cat([layer_relax, layer_nb_features],dim=1)
                layer_mu_new  = self.fc4_2(F.relu(self.fc4(layer_input)))
                layer_mu_new = layer_mu_new*(ratio_0!=0).float().unsqueeze(-1)
                relu_count_idx += 1
                #import pdb; pdb.set_trace()
                mu[relu_count_idx] = layer_mu_new.reshape(mu[relu_count_idx].size())


                if (th.sum(th.isnan(layer_mu_new))!=0):
                    print('mu contains nan')
                    import pdb;pdb.set_trace()

            elif type(layer) is Flatten:
                out_features = [-1] + th.tensor(lower_bounds_all[idx].size()).tolist() 
                pass
            else:
                raise NotImplementedError

        # property layer 
        # forward pass
        nb_features_temp = [layers['prop_layers'][i].weight @ mu[relu_count_idx][i] for i in range(batch_size)]

        pre_layer_bias = [layers['prop_layers'][i].bias for i in range(batch_size)]
        nb_features_temp = th.cat(nb_features_temp, 0)
        pre_layer_bias = th.cat(pre_layer_bias,0)
        layer_lower_pre = lower_bounds_all[idx].view(-1)
        layer_upper_pre = upper_bounds_all[idx].view(-1)
        layer_n_bounds = th.cat([ layer_lower_pre.unsqueeze(-1), 
                                    layer_upper_pre.unsqueeze(-1), 
                                    primals[0].unsqueeze(-1),
                                    pre_layer_bias.unsqueeze(-1)],1)
        layer_relax_output = F.relu(self.out1(layer_n_bounds))
        layer_input = th.cat([layer_relax_output, nb_features_temp],dim=1)
        layer_mu_new  = self.out3(F.relu(self.out2(layer_input)))
        relu_count_idx += 1
        mu[relu_count_idx] = layer_mu_new.reshape(mu[relu_count_idx].size())
            
        return mu





class FeatureUpdates(nn.Module):
    '''
    this class updates feature vectors from t=1 and t=T
    '''

    def __init__(self, p):
        '''
        p_list contains the input and output dimensions of feature vectors for all layers
        len(p_list) = T+1
        '''
        super(FeatureUpdates, self).__init__()
        self.p = p
        self.update = LayerUpdate(p)


    def forward(self, lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, pre_mu=None):
        mu = pre_mu if pre_mu else init_mu(lower_bounds_all, self.p)
        mu = self.update(lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, mu)
        return mu



class ComputeFinalScore(nn.Module):
    '''
    this class computes a final score for each node

    p: the dimension of feature vectors at the final stage
    '''
    def __init__(self, p):
        super(ComputeFinalScore,self).__init__()
        self.p = p
        self.fnode = nn.Linear(p, p)
        self.fscore = nn.Linear(p, 1)
    
    def utils(self, mu):
        scores = []
        for layer in mu[1:-1]:
            scores_current_layer = mu[0].new_full((layer.size()[0],), fill_value=0.0) 
            scores.append(scores_current_layer)
        return scores


    def forward(self, mu, masks):
        scores = []
        for batch_idx in range(len(mu[0])):
            mu_temp = th.cat([i[batch_idx] for i in mu[1:-1]],dim=0)
            mu_temp = mu_temp[masks[batch_idx].nonzero().view(-1)]
            temp = self.fnode(mu_temp)
            score = self.fscore(F.relu(temp))
            scores.append(score.view(-1))
        return scores


class ExpNet(nn.Module):
    def __init__(self, p):
        super(ExpNet, self).__init__()
        self.FeatureUpdates = FeatureUpdates(p)
        self.ComputeFinalScore = ComputeFinalScore(p)

    def forward(self, lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, masks, pre_mu=None):
        mu = self.FeatureUpdates(lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, pre_mu=pre_mu)
        scores = self.ComputeFinalScore(mu, masks)

        return scores, mu


def init_mu(lower_bounds_all, p):
    mu = []
    batch_size = len(lower_bounds_all[0])
    for i in lower_bounds_all:
        required_size = i[0].view(-1).size()
        mus_current_layer = lower_bounds_all[0].new_full((batch_size,required_size[0],p), fill_value=0.0) 
        mu.append(mus_current_layer)
    
    return mu


def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound - F.relu(lower_bound)
    upper_temp = F.relu(upper_bound)
    diff = upper_temp-lower_temp
    zero_ids = (diff==0).nonzero()
    if len(zero_ids)>0:
        if th.sum(upper_temp[zero_ids])==0:
            diff[zero_ids] = 1e-5
    slope_ratio0 = upper_temp/diff

    intercept = -1*lower_temp*slope_ratio0
    ambi_mask = (intercept>0).float()
    slope_ratio1 = (1 - 2*(slope_ratio0*ambi_mask))*ambi_mask + slope_ratio0

    return slope_ratio0, slope_ratio1, intercept, ambi_mask


