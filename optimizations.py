import numpy as np


# log posterior to find the map
def log_posterior_map(w,corpus,learning_trace,prior):
    outcomes=[learning_trace[j][1] for j in range(len(learning_trace))]
    items_id=[learning_trace[j][0] for j in range(len(learning_trace))]   
    corrects=[np.log(corpus.dic_item[items_id[j]].expected_response(w)) for j in range(len(learning_trace))]
    errors=[np.log(1-corpus.dic_item[items_id[j]].expected_response(w)) for j in range(len(learning_trace))]
    L=np.sum([outcomes[j]*corrects[j]+(1-outcomes[j])*errors[j] for j in range(len(learning_trace))])+prior.logpdf(w)
    return L

# log maximum lilikeihood
def log_posterior(w,corpus,learning_trace):
    outcomes=[learning_trace[j][1] for j in range(len(learning_trace))]
    items_id=[learning_trace[j][0] for j in range(len(learning_trace))]   
    corrects=[np.log(corpus.dic_item[items_id[j]].expected_response(w)) for j in range(len(learning_trace))]
    errors=[np.log(1-corpus.dic_item[items_id[j]].expected_response(w)) for j in range(len(learning_trace))]
    L=np.sum([outcomes[j]*corrects[j]+(1-outcomes[j])*errors[j] for j in range(len(learning_trace))])
    return L
# laplace approximation
def laplace_approx(w, w_map, H):
    detH =  np.linalg.det(H)
    constant = np.sqrt(detH)/(2*np.pi)**(2.0/2.0)
    density = np.exp(-0.5 * (w-w_map).dot(H).dot(w-w_map))
    return constant * density