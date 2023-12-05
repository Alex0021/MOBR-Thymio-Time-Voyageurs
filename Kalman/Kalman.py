import numpy as np
 
np.set_printoptions(precision=3,suppress=True)

A_k_minus_1 = np.array([[1.0,0,0],[0,1.0,0],[0,0,1.0]]) #Identity matrix touche pas

process_noise_v_k_minus_1 = np.array([0.01,0.01,0.003]) #modifier erreur 

Q_k = np.array([[1.0,0,0], [0, 1.0,   0],[0,0,1.0]])# State model noise covariance matrix Q_k

# If Q big, Kalman Filter tracks large changes in 
# the sensor measurements more closely than for smaller Q.
# Q is a square matrix that has the same number of rows as states.
                 

H_k = np.array([[1.0,0,0],[0,1.0,0],[0,0,1.0]]) #touche pas 
                         

R_k = np.array([[0.0001,0,0],[0,0.0001,0],[0,0,0.0001]]) # Sensor measurement noise covariance matrix R_k
# Has the same number of rows and columns as sensor measurements.
# If we are sure about the measurements, R will be near zero.

sn = np.array([0.07,0.07,0.04]) #a modifier erreur 
 
def getB(yaw, deltak):
    B = np.array([[np.cos(yaw)*deltak, 0],[np.sin(yaw)*deltak, 0],[0, deltak]])
    return B
 
def efilter(z_k_observation_vector, state_estimate_k_minus_1, 
        control_vector_k_minus_1, P_k_minus_1):
    dk=10 #intervalle de temps à modifier
    state_estimate_k = A_k_minus_1 @ (state_estimate_k_minus_1) + (getB(state_estimate_k_minus_1[2],dk)) @ (control_vector_k_minus_1) + (process_noise_v_k_minus_1)    
    
    #print(f'State Estimate Before EKF={state_estimate_k}')
  #  print(f'control vector={control_vector_k_minus_1}')
    
    P_k = A_k_minus_1 @ P_k_minus_1 @ A_k_minus_1.T + (Q_k)
    measurement_residual_y_k = z_k_observation_vector-((H_k @ state_estimate_k) + (sn))
    
 #   print(f'Observation={z_k_observation_vector}')          
    
    S_k = H_k @ P_k @ H_k.T + R_k
    K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)
    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)    
    P_k = P_k - (K_k @ H_k @ P_k)
    
    #print(f'State Estimate After EKF={state_estimate_k}')
    
    return state_estimate_k, P_k
     
def Kalman(state,control,state_k_minus_1,P_k_minus_1):          
    
    optimal_state_estimate_k, covariance_estimate_k = efilter(state,state_k_minus_1, control, P_k_minus_1) 
    return optimal_state_estimate_k, covariance_estimate_k
        