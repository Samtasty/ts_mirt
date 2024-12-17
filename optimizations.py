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

def reg_log_likelihood(w,corpus,learning_trace):
    items_id,outcomes=zip(*learning_trace)      
    corrects=[corpus.dic_rewards[j]*np.log(corpus.get_item(j).expected_response(w)+1e-10) for j in items_id]
    errors=[corpus.dic_rewards[j]*np.log(1-corpus.get_item(j).expected_response(w)+1e-10) for j in items_id]
    L=np.sum([outcomes[j]*corrects[j]+(1-outcomes[j])*errors[j] for j in range(len(outcomes))])
    return L


def ellipsoid_over_items(a,b,inv_H,corpus,list_of_index_items): 
    """
    Compute the ellipsoid over the items.
    
    Args:
        a (np.array): Center of the ellipsoid.
        b (float): Radius of the ellipsoid.
        inv_H (np.array): Inverse of the Hessian matrix.
        list_of_items (list of np.array): List of items.
    
    Returns:
        list of np.array: List of items in the ellipsoid.
    """

    
    g_a=np.sum([corpus.dic_rewards[i]*corpus.dic_item[i].expected_response(a)*corpus.dic_item[i].kcs for i in list_of_index_items],axis=0)

    g_b=np.sum([corpus.dic_rewards[i]*corpus.dic_item[i].expected_response(b)*corpus.dic_item[i].kcs for i in list_of_index_items],axis=0)

    return (g_a-g_b)@inv_H@(g_a-g_b)
