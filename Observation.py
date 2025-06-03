import numpy as np

def observation(accuracy, state_number, Matrix_type = True):
    """
    the observation matrix is manually designed, based on the state space
    Args:
        accuracy: the accurate level of visual inspections(such as 70% =input 0.7)
        state_number: the state number of a component
        matrix_type: the discrete distributions of observation matrix
    Returns:
    observation_matrix
     """
    if Matrix_type:
        observation_matrix = np.zeros((state_number, state_number))
        observation_matrix[0, 0] = accuracy
        observation_matrix[0, 1] = 1 - accuracy
        observation_matrix[state_number - 1, state_number - 2] = 1 - accuracy
        observation_matrix[state_number - 1, state_number - 1] = accuracy
        for step in range(state_number - 2):
            observation_matrix[step + 1, step] = (1 - accuracy) / 2
            observation_matrix[step + 1, step + 1] = accuracy
            observation_matrix[step + 1, step + 2] = (1 - accuracy) / 2

        return observation_matrix
    else:
        observation_matrix = np.zeros((state_number, state_number))
        observation_matrix[0, 0] = accuracy
        observation_matrix[0, 1] = (1 - accuracy) * 2 / 3
        observation_matrix[0, 2] = (1 - accuracy) * 1 / 3

        observation_matrix[1, 0] = (1 - accuracy) * 2 / 5
        observation_matrix[1, 1] = accuracy
        observation_matrix[1, 2] = (1 - accuracy) * 2 / 5
        observation_matrix[1, 3] = (1 - accuracy) * 1 / 5

        observation_matrix[state_number - 2, state_number - 4] = (1 - accuracy) * 1 / 5
        observation_matrix[state_number - 2, state_number - 3] = (1 - accuracy) * 2 / 5
        observation_matrix[state_number - 2, state_number - 2] = accuracy
        observation_matrix[state_number - 2, state_number - 1] = (1 - accuracy) * 2 / 5

        observation_matrix[state_number - 1, state_number - 3] = (1 - accuracy) * 1 / 3
        observation_matrix[state_number - 1, state_number - 2] = (1 - accuracy) * 2 / 3
        observation_matrix[state_number - 1, state_number - 1] = accuracy
        for step in range(state_number - 4):
            observation_matrix[step + 2, step] = (1 - accuracy) / 6
            observation_matrix[step + 2, step + 1] = (1 - accuracy) / 3
            observation_matrix[step + 2, step + 2] = accuracy
            observation_matrix[step + 2, step + 3] = (1 - accuracy) / 3
            observation_matrix[step + 2, step + 4] = (1 - accuracy) / 6
        return observation_matrix