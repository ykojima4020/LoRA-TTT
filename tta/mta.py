import torch
import torch.nn.functional as F 

def gaussian_kernel(mu, bandwidth, datapoints):
    dist = torch.norm(datapoints - mu,dim=-1, p=2)
    density = torch.exp(-dist**2/(2*bandwidth**2))
    return density


def solve_mta(model, images):
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            text_features = model.clip.get_text_features()
            image_features = model.clip.image_encode(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logits = model.clip.logit_scale.exp() * (image_features @ text_features.t())

    # default values from https://github.com/MaxZanella/MTA/blob/main/main.py#L346-L347 
    lambda_y = 0.2
    lambda_q = 4
    max_iter = 5
    temperature = 1
    
    batch_size = image_features.shape[0]
    
    # bandwidth
    dist = torch.cdist(image_features, image_features)
    sorted_dist, _ = torch.sort(dist, dim=1)
    k = int(0.3 * (image_features.shape[0]-1))
    selected_distances = sorted_dist[:, 1:k+1]**2  # exclude the distance to the point itself 
    mean_distance = torch.mean(selected_distances, dim=1)
    bandwidth = torch.sqrt(0.5 * mean_distance) 
    
    # Affinity matrix based on logits
    affinity_matrix = (logits/temperature).softmax(1) @ (logits/temperature).softmax(1).t()
    
    # Inlierness scores initialization: uniform
    y = torch.ones(batch_size, device=image_features.device)/batch_size
    
    # Mode initialization: original image embedding
    mode_init = image_features[0]
    mode = mode_init
    
    convergence = False
    th = 1e-6
    iter = 0
    
    while not convergence:
        
        ###################
        # Inlierness step #
        ###################
        
        density = gaussian_kernel(mode, bandwidth, image_features)
    
        convergence_inlierness = False
        i = 0
        while not convergence_inlierness:
            i+=1
            old_y = y
            weighted_affinity = affinity_matrix * y.unsqueeze(0)
            y = F.softmax(1/lambda_y * (density + lambda_q * torch.sum(weighted_affinity, dim=1)), dim=-1)

            if torch.norm(old_y - y)<th or i>= max_iter:
                convergence_inlierness = True
        
        #############
        # Mode step #
        #############
        
        convergence_mode = False
        i=0
        while not convergence_mode:
            i+=1
            old_mode = mode
            density = gaussian_kernel(mode, bandwidth, image_features)
            weighted_density = density *  y
            mode = torch.sum(weighted_density.unsqueeze(1)* image_features, dim=0)/torch.sum(weighted_density)
            mode /= mode.norm(p=2, dim=-1)
            
            if torch.norm(old_mode - mode)<th or i>= max_iter:
                convergence_mode = True
        
        iter +=1
        if iter >= max_iter:
            convergence = True
    
    output = mode.unsqueeze(0) @ text_features.t() * model.clip.logit_scale.exp()
    return output