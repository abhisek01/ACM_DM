from dialogue_config import FAIL, SUCCESS


def convert_list_to_dict(lst):
    """
    Convert list to dict where the keys are the list elements, and the values are the indices of the elements in the list.

    Parameters:
        lst (list)

    Returns:
        dict
    """

    if len(lst) > len(set(lst)):
        raise ValueError('List must be unique!')
    return {k: v for v, k in enumerate(lst)}


def remove_empty_slots(dic):
    """
    Removes all items with values of '' (ie values of empty string).

    Parameters:
        dic (dict)
    """

    for id in list(dic.keys()):
        for key in list(dic[id].keys()):
            if dic[id][key] == '':
                dic[id].pop(key)


def reward_function(success, max_round,tot,prev_slt_len,current_slt_len,stp):
    """
    Return the reward given the success.

    Return -1 + -max_round if success is FAIL, -1 + 2 * max_round if success is SUCCESS and -1 otherwise.

    Parameters:
        success (int)

    Returns:
        int: Reward
    """

 

    if success == FAIL:
        reward = -max_round
        print('&&&failed&&&&')
        print('Step:',stp)
    elif success == SUCCESS:
        print('*****Success***')
        print('Stp:',stp)
        reward = 5 * (max_round - stp)
    #elif(prev_slt_len==current_slt_len and prev_slt_len!=0):
        #reward = -5
    elif((prev_slt_len-current_slt_len)>0):      ##for Reducing REst SLot/ Increasing Slot
        reward = prev_slt_len-current_slt_len
    else:
        reward = -2     ### For Each utterance
        
    print('Steppp:',stp)
    return reward

##Neutral and Negative (1.Agent Action Repeatition 2.User Action Repeatition )
def Sent_Score(a_r,u_r,p):  
    ss = +1
    if(a_r==1):             ##Agent Action Repeatition
         ss = ss - 7          
    if(u_r==1):             ##User Action Repeatition 
        ss = ss -2
    if(p == 1):             ##Inappropriate Reply by Agent (req by user -> req by agent)
        ss = ss - 3
    #print('Sentiment Score: ',ss)
    return ss
    
