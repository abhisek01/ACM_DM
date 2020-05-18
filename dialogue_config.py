# Special slot values (for reference)
'PLACEHOLDER'  # For informs
'UNK'  # For requests
'anything'  # means any value works for the slot with this value
'no match available'  # When the intent of the agent is match_found yet no db match fits current constraints

#######################################
# Usersim Config
#######################################
# Used in EMC for intent error (and in user)
usersim_intents = ['inform', 'request', 'thanks', 'Feedback','reject', 'done']

# The goal of the agent is to inform a match for this key
usersim_default_key = 'Phn_Booking'

# Required to be in the first action in inform slots of the usersim if they exist in the goal inform slots
usersim_required_init_inform_keys = ['model','brand']

#######################################
# Agent Config
#######################################

# Possible inform and request slots for the agent
agent_inform_slots = ['brand','model','battery','ram','pcamera','scamera','radio','displaysize','status','sim','gps','os','color','internalram','weight','releasedyear','releasedmonth','price', usersim_default_key]
agent_request_slots = ['brand','model','battery','ram','pcamera','scamera','radio','displaysize','status','sim','gps','os','color','internalram','weight','releasedyear','releasedmonth','price']

# Possible actions for agent
agent_actions = [
    {'intent': 'done', 'inform_slots': {}, 'request_slots': {}},  # Triggers closing of conversation
    {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}
]
for slot in agent_inform_slots:
    # Must use intent match found to inform this, but still have to keep in agent inform slots
    if slot == usersim_default_key:
        continue
    agent_actions.append({'intent': 'inform', 'inform_slots': {slot: 'PLACEHOLDER'}, 'request_slots': {}})
for slot in agent_request_slots:
    agent_actions.append({'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}})
print(agent_actions[16])
#agent_actions.append({'intent': 'reject', 'inform_slots': {}, 'request_slots': {}})
#agent_actions.append({'intent': 'thanks', 'inform_slots': {}, 'request_slots': {}})
#agent_actions.append({'intent': 'Rephrase', 'inform_slots': {}, 'request_slots': {}})
# Rule-based policy request list'
rule_requests = ['brand','model','battery','ram','pcamera','scamera','displaysize','color','internalram','price']

# These are possible inform slot keys that cannot be used to query
no_query_keys = [usersim_default_key]

#######################################
# Global config
#######################################

# These are used for both constraint check AND success check in usersim
FAIL = -1
NO_OUTCOME = 0
SUCCESS = 1

# All possible intents (for one-hot conversion in ST.get_state())
all_intents = ['inform', 'request', 'done', 'Feedback','Rephrase','match_found', 'thanks', 'reject']

# All possible slots (for one-hot conversion in ST.get_state())
all_slots = ['brand','model','battery','ram','pcamera','scamera','radio','displaysize','status','sim','gps','os','color','internalram','weight','releasedyear','releasedmonth','price', usersim_default_key]
