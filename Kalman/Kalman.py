import numpy as np
 
# System matrices 
A_k_minus_1 = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]) 
Q_k = np.array([[6.128,-10.530625, -12.55625], [-10.530625, 48.62571429, 37.6], [-12.55625, 37.6, 56]])
H_k = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
R_k = np.array([[0.0001, 0, 0], [0, 0.0001, 0], [0, 0, 0.0001]])
sensor_noise = np.array([0.07, 0.07, 0.04]) 

def calculate_control_matrix(gamma, delta_k):
    """Calculate control matrix based on yaw and time delta."""
    return np.array([[np.cos(gamma) * delta_k, 0], [np.sin(gamma) * delta_k, 0], [0, delta_k]])
               
def Kalman(z_k_observation_vector,state_estimate_k_minus_1, control_vector_k_minus_1, P_k_minus_1,vision):
    delta_k = 5
    state_estimate_k = np.dot(A_k_minus_1,state_estimate_k_minus_1) + np.dot(calculate_control_matrix(state_estimate_k_minus_1[2], delta_k),control_vector_k_minus_1) + sensor_noise
    P_k = np.dot(np.dot(A_k_minus_1,P_k_minus_1),A_k_minus_1.T) + Q_k
    if (vision) : 
        measurement_residual_y_k = z_k_observation_vector - (np.dot(H_k,state_estimate_k) + sensor_noise)
        S_k = np.dot(np.dot(H_k,P_k),H_k.T) + R_k
        K_k = np.dot(np.dot(P_k,H_k.T),np.linalg.pinv(S_k))
               
    state_estimate_k += np.dot(K_k,measurement_residual_y_k)
    P_k -= np.dot(np.dot(K_k,H_k),P_k)

    return state_estimate_k, P_k
