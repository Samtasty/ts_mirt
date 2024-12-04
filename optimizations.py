import numpy as np


# log posterior to find the map
def log_posterior_map(w,corpus,learning_trace,prior):
    items_id,outcomes=zip(*learning_trace)      
    corrects=[np.log(corpus.get_item(j).expected_response(w)+1e-10) for j in items_id]
    errors=[np.log(1-corpus.get_item(j).expected_response(w)+1e-10) for j in items_id]
    L=np.sum([outcomes[j]*corrects[j]+(1-outcomes[j])*errors[j] for j in range(len(outcomes))])+prior.logpdf(w)
    return L

# log maximum lilikeihood
def log_posterior(w,corpus,learning_trace):
    
    items_id,outcomes=zip(*learning_trace)   
    corrects=[np.log(corpus.get_item(j).expected_response(w)+1e-10) for j in items_id]
    errors=[np.log(1-corpus.get_item(j).expected_response(w)+1e-10) for j in items_id]
    L=np.sum([outcomes[j]*corrects[j]+(1-outcomes[j])*errors[j] for j in range(len(outcomes))])
    return L
# laplace approximation
def laplace_approx(w, w_map, H):
    detH =  np.linalg.det(H)
    constant = np.sqrt(detH)/(2*np.pi)**(2.0/2.0)
    density = np.exp(-0.5 * (w-w_map).dot(H).dot(w-w_map))
    return constant * density

def fisher_information(w,corpus,learning_trace):
    items_id,outcomes=zip(*learning_trace)   
    H = np.zeros((len(w)+1,len(w)+1))
    for j in range(len(outcomes)):
        item = corpus.get_item(items_id[j])
        v=np.r_[item.kcs,-item.difficulty]
        outer_product = np.outer(v,v)
        p = item.expected_response(w)
        q = 1 - p
        H += p * q * outer_product
    return H