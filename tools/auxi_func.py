import pandas as pd

def calc_acc(file, u_t, sublst):
    '''
    file: action files
    u_t : standard actions
    sublst: extreme subj needed to be removed
    '''
    acc = []
    for i in range(u_t.shape[1]):#range(file1.shape[1])
        if i not in sublst:
            resp = file[i]  
            resp.reset_index(drop=True, inplace=True)
            correct_count = sum(
                (resp == u_t[0])
            )
            accuracy = correct_count / len(resp)
            acc.append(accuracy)
    return acc
