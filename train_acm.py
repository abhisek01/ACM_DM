from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from state_tracker import StateTracker
import pickle, argparse, json, math
from utils import remove_empty_slots
from user import User
from utils import Sent_Score
from dialogue_config import agent_actions
from advantage_acm import AdvantageACM
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":
    # Can provide constants file path in args OR run it as is and change 'CONSTANTS_FILE_PATH' below
    # 1) In terminal: python train.py --constants_path "constants.json"
    # 2) Run this file as is
    parser = argparse.ArgumentParser()
    parser.add_argument('--constants_path', dest='constants_path', type=str, default='')
    args = parser.parse_args()
    params = vars(args)

    # Load constants json into dict
    CONSTANTS_FILE_PATH = 'constants.json'
    if len(params['constants_path']) > 0:
        constants_file = params['constants_path']
    else:
        constants_file = CONSTANTS_FILE_PATH

    with open(constants_file) as f:
        constants = json.load(f)

    # Load file path constants
    file_path_dict = constants['db_file_paths']
    DATABASE_FILE_PATH = file_path_dict['database']
    DICT_FILE_PATH = file_path_dict['dict']
    USER_GOALS_FILE_PATH = file_path_dict['user_goals']

    # Load run constants
    run_dict = constants['run']
    USE_USERSIM = run_dict['usersim']
    WARMUP_MEM = run_dict['warmup_mem']
    NUM_EP_TRAIN = run_dict['num_ep_run']
    TRAIN_FREQ = run_dict['train_freq']
    MAX_ROUND_NUM = run_dict['max_round_num']
    SUCCESS_RATE_THRESHOLD = run_dict['success_rate_threshold']

    # Load movie DB
    # Note: If you get an unpickling error here then run 'pickle_converter.py' and it should fix it
    database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')

    # Clean DB
    remove_empty_slots(database)

    # Load movie dict
    db_dict = pickle.load(open(DICT_FILE_PATH, 'rb'), encoding='latin1')

    # Load goal File
    user_goals = pickle.load(open(USER_GOALS_FILE_PATH, 'rb'), encoding='latin1')

    # Init. Objects
    if USE_USERSIM:
        user = UserSimulator(user_goals, constants, database)
    else:
        user = User(constants)
    emc = ErrorModelController(db_dict, constants)
    state_tracker = StateTracker(database, constants)
    acm = AdvantageACM(state_tracker.get_state_size(), constants)
    
    
    


def run_round(state,Agent_Actions,User_Actions,tot_slt_len,stp,q, warmup):
    u_r =0                 ##User Repeatition 
    a_r = 0                ##Agent Repeatition
    a_q = 0                ##Agent Question
    u_q = q                ##User Question
    pen = 0         ##User asked Question Agent replied Question
    # 1) Agent takes action given state tracker's representation of dialogue (state)
    agent_action_index,agent_action = acm.act(state,warmup)
    print('Agent Action_Index:',agent_action_index)
    #print('Agent_Action:',agent_action)   
    if(agent_action['intent']=='request'):
        a_q = 1
    if(agent_action_index in Agent_Actions):
        a_r = 1
    else:
        Agent_Actions = Agent_Actions + [agent_action_index]
    if(u_q == 1 and a_q ==1):
        pen = 1
    # 2) Update state tracker with the agent's action
    state_tracker.update_state_agent(agent_action)
    # 3) User takes action given agent action
    user_action, reward, done, success = user.step(agent_action,tot_slt_len,stp)
    #print('User_Action:',user_action)
    #print('Task Oriented Reward: ',reward)
    p = user_action
    res = {key: p[key] for key in p.keys() 
                               & {'intent', 'request_slots','inform_slots'}}
    User_Actions = User_Actions + list(res)
    if(p['intent']=='request'):
        q = 1
    if not done:
        # 4) Infuse error into semantic frame level of user action
        emc.infuse_error(user_action)

    ss = Sent_Score(a_r,u_r,pen)
    reward = reward + ss
    # 5) Update state tracker with user action and reward
    state_tracker.update_state_user(user_action,reward)
    # 6) Get next state and add experience
    next_state = state_tracker.get_state(done)

    acm.remember(state,agent_action_index, reward, next_state, done)
    print('Total Reward:',reward)
    return next_state, reward, done, success,Agent_Actions,q


def warmup_run():
    """
    Runs the warmup stage of training which is used to fill the agents memory.

    The agent uses it's rule-based policy to make actions. The agent's memory is filled as this runs.
    Loop terminates when the size of the memory is equal to WARMUP_MEM or when the memory buffer is full.

    """

    print('Warmup Started...')
    total_step = 0
    s = 0
    while total_step != WARMUP_MEM and not acm.is_memory_full():
        print('WarmupPhase:',total_step)
        # Reset episode
        tot_slot_len = episode_reset()
        Agent_Actions = []
        User_Actions = []
        l = 0
        done = False
        # Get initial state from state tracker
        state = state_tracker.get_state()
        while not done:
            l = l + 1
            next_state, R, done, Succ,Agent_Actions,q = run_round(state,User_Actions,Agent_Actions,tot_slot_len,l,0,1)
            total_step += 1
            state = next_state
        if(Succ==1):
            s = s+1
    print('Total Success:',s)    
    print('...Warmup Ended')


def train_run():
    """
    Runs the loop that trains the agent.

    Trains the agent on the goal-oriented chatbot task. Training of the agent's neural network occurs every episode that
    TRAIN_FREQ is a multiple of. Terminates when the episode reaches NUM_EP_TRAIN.

    """

    print('Training Started...')
    #ac.initialize_episode()
    episode = 0
    period_reward_total = 0
    period_success_total = 0
    success_rate_best = 0.0
    succ_rate = []
    avg_rewd = []
    l = 0
    L = []
    x = 0
    r=0
    N = []
    while episode < NUM_EP_TRAIN:
        print('Episode:  ',episode)
        Agent_A = []
        prev_user_action = {}
        User_A = []
        qq = 0
        tot_slt_len = episode_reset()
        episode += 1
        done = False
        state = state_tracker.get_state()
        while not done:
            x = x + 1
            l = l + 1
            next_state, reward, done, success,Agent_Actions,q = run_round(state,Agent_A,User_A,tot_slt_len,x,qq,0)
            qq = q
            Agent_A = Agent_Actions
            #User_A = User_Actions
            period_reward_total += reward
            r = r + reward
            state = next_state
            

        print('Taken Length:',x)
        x = 0
        print('Cummulative Reward:',r)
        r = 0
        period_success_total += success

        # Train
        if episode % TRAIN_FREQ == 0:
            #print('Episode No:',episode)
            avg_len = l/TRAIN_FREQ
            
            #print('Episodic Avg Length:',avg_len)
            L = L + [avg_len]
            N = N + [period_success_total]
            # Check success rate
            success_rate = period_success_total / TRAIN_FREQ
            succ_rate = succ_rate + [success_rate]
            avg_reward = period_reward_total / TRAIN_FREQ
            #print('Episodic Reward:',period_reward_total)
            avg_rewd = avg_rewd + [avg_reward]
            # Flush
            print('Episode:',episode)
            print('Avg Reward:',avg_reward)
            print('Avg length:',avg_len)
            if success_rate >= success_rate_best and success_rate >= SUCCESS_RATE_THRESHOLD:
                acm.empty_memory()
            # Update current best success rate
            if success_rate > success_rate_best:
                print('Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {}' .format(episode, success_rate, avg_reward))
                acm.save()
                success_rate_best = success_rate
            period_success_total = 0
            period_reward_total = 0
            l = 0
            # Copy
            # Train
            acm.train_advantage_actor_critic()
    print('!!Training Result  ')
    print('Success Rate:',succ_rate)
    print('No of Success:',N)
    print('Avg Reward: ',avg_rewd)
    print('Avg Length:',L)
    
    print('...Training Ended')


def episode_reset():
    """
    Resets the episode/conversation in the warmup and training loops.

    Called in warmup and train to reset the state tracker, user and agent. Also get's the initial user action.

    """

    # First reset the state tracker
    state_tracker.reset()
    # Then pick an init user action
    tot_slt,user_action = user.reset()
    print('intial User Action::',user_action)
    # Infuse with error
    emc.infuse_error(user_action)
    # And update state tracker
    intial_reward = 0
    state_tracker.update_state_user(user_action,intial_reward)
    # Finally, reset agent
    acm.reset()
    return tot_slt


warmup_run()
train_run()
