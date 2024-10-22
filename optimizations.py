import numpy as np


# log posterior to find the map
def log_posterior_map(w,corpus,learning_trace,prior):
    items_id,outcomes=zip(*learning_trace)      
    corrects=[np.log(corpus.get_item(j).expected_response(w)) for j in items_id]
    errors=[np.log(1-corpus.get_item(j).expected_response(w)) for j in items_id]
    L=np.sum([outcomes[j]*corrects[j]+(1-outcomes[j])*errors[j] for j in range(len(outcomes))])+prior.logpdf(w)
    return L

# log maximum lilikeihood
def log_posterior(w,corpus,learning_trace):
    
    items_id,outcomes=zip(*learning_trace)   
    corrects=[np.log(corpus.get_item(j).expected_response(w)) for j in items_id]
    errors=[np.log(1-corpus.get_item(j).expected_response(w)) for j in items_id]
    L=np.sum([outcomes[j]*corrects[j]+(1-outcomes[j])*errors[j] for j in range(len(outcomes))])
    return L
# laplace approximation
def laplace_approx(w, w_map, H):
    detH =  np.linalg.det(H)
    constant = np.sqrt(detH)/(2*np.pi)**(2.0/2.0)
    density = np.exp(-0.5 * (w-w_map).dot(H).dot(w-w_map))
    return constant * density