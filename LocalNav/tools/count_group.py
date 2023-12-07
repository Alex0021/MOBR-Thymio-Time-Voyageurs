import numpy as np

# Add subgroups of obstacles into 1 group of obstacle if distance between groups is less than 11 cm (width of thymio)
def count_transitions(mask, X, y_pred):
    transitions = 0
    edge_x = 0
    edge_y = 0
    j = 0

    if len(mask) > 1:
        current_state = mask[0]
        # Look through each detectors
        for i, state in enumerate(mask):
            if state is True:
                edge_x = X[i]
                edge_y = y_pred[i]
                j = i

            if state != current_state:
                # Make a wall if distance between detected groups of obstacles is less than lenght of thymio
                if current_state == True and np.sqrt( (edge_x-X[i])*(edge_x-X[i]) + (edge_y-y_pred[i])*(edge_y-y_pred[i])) < 11: 
                    mask[j:i] = True
                else:
                    transitions += 1
                current_state = state
                
        if all(mask):
            transitions = 1
    
    return transitions, mask

# Cluster group of obstacles
def count_group(mask, mask_back, X, y_pred, X_back, y_pred_back):
    nb_group_front, mask_front = count_transitions(mask, X, y_pred)
    nb_group_back, mask_back = count_transitions(mask_back, X_back, y_pred_back)
    
    nb_group = nb_group_front + nb_group_back
    mask = np.concatenate((mask_front, mask_back), axis=0)
    
    return nb_group, mask_front, mask_back