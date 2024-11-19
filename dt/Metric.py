import numpy as np

def is_two_level_nested_list(data):
    if isinstance(data, list):
        for element in data:
            if not isinstance(element, list):
                return False
            return True
    return False

def calculate_mean_performance(data):
    '''
        data: List<List<>>
    '''
    is_two_level = is_two_level_nested_list(data)

    if not is_two_level:
        raise NotImplementedError('Please use the nested array as input')

    performance = []
    for row in data:
        performance.append(np.mean(row))
    
    return performance

def calculate_forgetting(data):
    '''
        data: List<List<>>
        env: cw10 or cw20
    '''
    is_two_level = is_two_level_nested_list(data)

    if not is_two_level:
        raise NotImplementedError('Please use the nested array as input')

    forgetting = []
    for i, row in enumerate(data):
        forgetting.append(row[i] - data[-1][i])
    
    return np.mean(forgetting)

def calculate_forward_transfer(data, init_data):
    '''
        data: List<List<>>
    '''
    is_two_level = is_two_level_nested_list(data)

    if not is_two_level:
        raise NotImplementedError('Please use the nested array as input')
    
    forward_transfer = []
    for i, row in enumerate(data):
        if i+1 >= len(row):
            break
        forward_transfer.append(row[i+1] - init_data[i+1])
    
    return np.mean(forward_transfer)

def calculate_backward_transfer(data):
    '''
        data: List<List<>>
    '''
    is_two_level = is_two_level_nested_list(data)

    if not is_two_level:
        raise NotImplementedError('Please use the nested array as input')

    backward_transfer = []

    for i, row in enumerate(data):
        if i == len(row) - 1:
            break
        backward_transfer.append(data[-1][i] - row[i])
    
    return np.mean(backward_transfer)

