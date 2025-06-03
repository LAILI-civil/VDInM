import numpy as np


def Markov_state_transition_matrix(state_number, duration):
    """
        :param state_number: 5
        :param duration: a vector size is state_number - 1 to consider the duration of state i deteriorate to i+1
        :return: transition_matrix:
        """
    # assumption is the one-jump state can happen in a defined period
    transition_matrix = np.zeros((state_number, state_number))
    transition_matrix[state_number - 1, state_number - 1] = 1

    for step, ri in enumerate(duration):
        Pi = ri / (1 + ri)
        Qi = 1 / (1 + ri)
        transition_matrix[step, step] = Pi
        transition_matrix[step, step + 1] = Qi
    return transition_matrix


"""
state transition in Markov chain
"""

def state_evolution(state, time, old_hidden, state_T, state_T_D, normalized_time):
    """
    :param state: a size of number * 5 state vector
    :param time: a size of number * 1 vector
    :param old_hidden: a size of number * 1 vector
    :param state_T: state transition in normal service. size 5 * 5
    :param state_T_D: state transition in corrosion condition, size 5 * 5
    :param normalized_time: scale value
    :return: new_state, new_time, new_hidden
    """
    new_hidden = 0
    new_state = np.zeros((len(state)))

    if time <= 0:
        new_state[0: 5] = state[0: 5] @ state_T_D
        new_state[0: 5] = new_state[0: 5] / np.sum(new_state[0: 5])
        new_time = 0

        random_number = np.random.uniform(0, 1)
        state_mark = 0.
        for j in range(len(state)):
            state_mark = state_mark + state_T_D[old_hidden, j]
            if random_number <= state_mark:
                new_hidden = j
                break

    else:
        new_state[0: 5] = state[0: 5] @ state_T
        new_state[0: 5] = new_state[0: 5] / np.sum(new_state[0: 5])
        new_time = time - 1 / normalized_time
        if np.abs(new_time) < 1e-10:
            new_time = 0

        random_number = np.random.uniform(0, 1)
        state_mark = 0.
        for j in range(len(state)):
            state_mark = state_mark + state_T[old_hidden, j]
            if random_number <= state_mark:
                new_hidden = j
                break

    return new_state, new_time, new_hidden
