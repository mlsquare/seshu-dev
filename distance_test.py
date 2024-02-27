import torch
from transformers import AutoModelForCausalLM , 

import numpy as np
from scipy.stats import entropy
from scipy.special import softmax
from sklearn.preprocessing import normalize


model = AutoModelForCausalLM.from_pretrained('Q-bert/Mamba-130M', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('Q-bert/Mamba-130M')

def get_logits(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    logits = model(input_ids)[0].detach().numpy()
    logits = logits[0,:,:]
    return logits

x1 =  get_logits("Hi there, how are you doing today?")
print(x1.shape)

x2 =  get_logits("Hi here, how were you doing today?")
print(x2.shape)


# upper and lower bounds on relative entropy
# https://www.sciencedirect.com/science/article/pii/S0898122100000894
# lower bound, Eqn 1.2
# upper bound, Eqn 2.1

def get_bounds(p1,p2):
    lb = 0.5*np.square(np.sum(np.abs(p1-p2), axis=1))
    ub = np.sum(np.square(p1)/p2, axis=1)-1
    ub[ub<0] = 0
    lb[lb<0] = 0    
    return lb, ub


def get_entropy(x1,x2,logits=True, eps=1e-5):
    
    if logits:
        p1 = softmax(x1, axis=1)
        p2 = softmax(x2, axis=1)
    else:
        p1 = x1
        p2 = x2

    p1+=eps
    p2+=eps

    p1 = normalize(p1, axis=1, norm='l1')
    p2 = normalize(p2, axis=1, norm='l1')
    

    en1 = entropy(p1,axis=1)
    en2 = entropy(p2,axis=1)
    kl1 = entropy(p1,p2, axis=1)
    kl2 = entropy(p2,p1, axis=1)

    
    lb1, ub1 = get_bounds(p1,p2)
    lb2, ub2 = get_bounds(p2,p1)

    kl1_scaled = (kl1-lb1)/(ub1-lb1)
    kl2_scaled = (kl2-lb2)/(ub2-lb2)

    ind  = np.where(ub1==lb1)
    kl1_scaled[ind]= ub1[ind]

    ind  = np.where(ub2==lb2)
    kl2_scaled[ind]= lb2[ind]

    return en1, en2, kl1, kl2, lb1, lb2, ub1, ub2, kl1_scaled, kl2_scaled

def distance(x1,x2, k=10):
    
    en1, en2, kl1, kl2, lb1, lb2, ub1, ub2, kl1s, kl2s = get_entropy(x1,x2, logits=True)
    
    ind = np.argsort(x1,axis=1)
    dp1 = np.take_along_axis(x1, ind, axis=1)
    dp2 = np.take_along_axis(x2, ind, axis=1)
    dp1 = dp1[:,:k+1]
    dp2 = dp2[:,:k+1]
    dp1[:,k] = 1-np.sum(dp1[:,:k],axis=1)
    dp2[:,k] = 1-np.sum(dp2[:,:k],axis=1)

    en1k, en2k, kl1k, kl2k, lb1k, lb2k, ub1k, ub2k, kl1ks, kl2ks = get_entropy(dp1,dp2, logits=True)

    result = {}
    result['full'] = {'h1': en1, 'h2': en2, 'kl1': kl1, 'kl2': kl2, 'lb1': lb1, 'lb2': lb2, 'ub1': ub1, 'ub2': ub2, "skl1": kl1s, "skl2": kl2s}
    result['topk'] = {'h1': en1k, 'h2': en2k, 'kl1': kl1k, 'kl2': kl2k, 'lb1': lb1k, 'lb2': lb2k, 'ub1': ub1k, 'ub2': ub2k, 'skl1': kl1ks, 'skl2': kl2ks}

    return result


en = distance(x1,x2, k=10)
kl1 = en['full']['kl1']

lb1 = en['full']['lb1']
ub1 = en['full']['ub1']

kl1k = en['topk']['kl1']
lb1k = en['topk']['lb1']
ub1k = en['topk']['ub1']



print('KL of x1|x2', kl1)
print('LB of KL of x2|x1', lb1)
print('UB of KL of x2|x1', ub1)

print('Top-k KL of x1|x2', kl1k)
print('Top-k LB of KL of x2|x1', lb1k)
print('Top-k UB of KL of x2|x1', ub1k)


print('scaled kl1', en['full']['skl1'])
print('scaled kl1', en['full']['skl2'])

print('top-k scaled kl1', en['topk']['skl1'])
print('top-k scaled kl2', en['topk']['skl2'])




