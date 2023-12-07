import numpy as np
 
# System matrices 
A_t_minus_1 = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]) #identity matrix 
Q_t = np.array([[6.128,-10.530625, -12.55625], [-10.530625, 48.62571429, 37.6], [-12.55625, 37.6, 56]]) #determined with the covariance matrix according x,y and gamma
H_t = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
R_t = np.array([[0.0001, 0, 0], [0, 0.0001, 0], [0, 0, 0.0001]])
sensor_noise = np.array([3, 9.15, 10]) #determined with the difference between the biggest and the smallest values of the tests in x,y and gamma

def calculate_control_matrix(gamma, delta_t):
    return np.array([[np.cos(gamma) * delta_t, 0], [np.sin(gamma) * delta_t, 0], [0, delta_t]])
               
import numpy as np

def Kalman(z_t_observation_vector, state_0, control_vector_t_minus_1, P_t_minus_1, vision):
    """
    Kalman filter implementation for state estimation.

    Args:
        z_t_observation_vector (numpy.ndarray): The observation vector at time t. [m_x[m], m_y[m], m_theta[rad]]
        state_0 (numpy.ndarray): The initial state vector at time t-1 [m_x[m],m_y[m],m_theta[rad]].
        control_vector_t_minus_1 (numpy.ndarray): The control vector at time t-1 [m/s,rad/s].
        P_t_minus_1 (numpy.ndarray): The error covariance matrix at time t-1.
        vision (bool): Flag indicating whether camera vision is used.

    Returns:
        tuple: A tuple containing the new predicted state estimate and the updated error covariance matrix.
    """

    delta_t = 5
    
    # 1st step: prediction state estimate 
    new_predicted_state_t = np.dot(A_t_minus_1, state_0) + np.dot(calculate_control_matrix(state_0[2], delta_t), control_vector_t_minus_1) + sensor_noise
    P_t = np.dot(np.dot(A_t_minus_1, P_t_minus_1), A_t_minus_1.T) + Q_t
    
    if vision:
        # If camera is used, we implement the second step: update state estimate. Otherwise, we only do the first step.
        measurement_residual_y_t = z_t_observation_vector - (np.dot(H_t, new_predicted_state_t) + sensor_noise)
        S_t = np.dot(np.dot(H_t, P_t), H_t.T) + R_t
        K_t = np.dot(np.dot(P_t, H_t.T), np.linalg.pinv(S_t))
        new_predicted_state_t += np.dot(K_t, measurement_residual_y_t)
        P_t -= np.dot(np.dot(K_t, H_t), P_t)

    return new_predicted_state_t, P_t
