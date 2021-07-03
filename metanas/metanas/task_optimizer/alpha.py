import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from metanas.utils import genotypes as gt


def sample_gumbel(shape, eps=1e-20):
    """Generate sample from Gumbel distribution
    """
    U = torch.Tensor(shape).uniform_(0, 1).cuda()
    sample = -(torch.log(-torch.log(U + eps) + eps))
    return sample


def gumbel_softmax_sample(logits, temperature):
    """Generate samples from Gumbel Softmax distribution
    """
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


class Alpha(nn.Module):
    def __init__(self, alpha_normal, alpha_reduce):
        super(Alpha, self).__init__()
        # Initialization of these alphas and binding to 
        # a_optim is done in metanas main
        self.alpha_normal = alpha_normal
        self.alpha_reduce = alpha_reduce
        
    def forward(self, temperature=0.4, softeps_weights=0.0):
        def sample_weight_edge(edge):
            if softeps_weights > 0:
                q = F.softmax(edge, dim=-1)
                edge = torch.log(q + softeps_weights)

            return gumbel_softmax_sample(edge, temperature)
            
        sample_normal, sample_reduce = [],  []
        
        for alpha in self.alpha_reduce:
            weight = sample_weight_edge(alpha)
            sample_reduce.append(weight)
        
        for alpha in self.alpha_normal:
            weight = sample_weight_edge(alpha)
            sample_normal.append(weight)

        return sample_normal, sample_reduce 
    
    def log_probability(self, w_normal, w_reduce, d_normal, d_reduce):
        def log_q(d, a):
            return torch.sum(torch.sum(d * a, dim=-1) - torch.logsumexp(
                a, dim=-1), dim=0)

        log_q_d = 0
        for n_edges, d_edges in zip(w_normal, d_normal):
            for i in range(len(n_edges)):
                a = n_edges[i]
                w = d_edges[i]
                log_q_d += log_q(w, a)

        for n_edges, d_edges in zip(w_reduce, d_reduce):
            for i in range(len(n_edges)):
                a = n_edges[i]
                w = d_edges[i]
                log_q_d += log_q(w, a)
        return log_q_d
    
    def entropy_loss(self, w_normal, w_reduce):
        def entropy(logit):
            q = F.softmax(logit, dim=-1)
            return - torch.sum(torch.sum(
                q * logit, dim=-1) - torch.logsumexp(logit, dim=-1), dim=0)

        entr = 0
        for n_edges in w_normal:
            for i in range(len(n_edges)):
                logit = n_edges[i]
                entr += entropy(logit)

        for n_edges in w_reduce:
            for i in range(len(n_edges)):
                logit = n_edges[i]
                entr += entropy(logit)
        return torch.mean(entr, dim=0)

    def clone_weights(self, sample_normal, sample_reduce):
        # detach if we want to remove the grad_fn
        return [w.clone() for w in sample_normal], [w.clone() for w in sample_reduce]
        