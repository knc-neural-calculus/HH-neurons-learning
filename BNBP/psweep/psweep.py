"""contains configuration data and some methods for processing it"""

from typing import *
import numpy as np
import math

LEN_ID = 12

# UGLY: typenames defined here are not very nice

t_Key = Literal[
	'RUN_ID',
	'CONFIG_ID',
	'DIRNAME',

	'N_LAYER_0',
	'N_LAYER_1',
	'N_LAYER_2',
	
	'DELTA_T',
	'SIM_STEPS',
	
	'BATCH_SIZE',
	'MAX_EPOCHS',
	
	'COUPLING_HIDDEN',
	'COUPLING_OUT',
	'LEARNING_RATE',
	'OUTPUT_SCALAR',
	'LF_HIDDEN',
	'LF_OUT',
	'USE_BIAS',
	'BETA_PHASE_2',
	'NUM_SNIFFS',

	'GNA',
	'GK',
	'GL',
	'ENA',
	'EK',
	'EL',
]
t_Val = Union[float,int,str]
cn_Dict = Dict[t_Key,t_Val]
cn_Dict_R = Dict[t_Key,List[t_Val]]

LossMode = Literal['abs', 'rel']

CONSTS_DEFAULT_KEYS = get_args(t_Key)


CONSTS_DEFAULT_KEYS_META = [
	'RUN_ID',
	'CONFIG_ID',
	'DIRNAME',
]



CONSTS_DEFAULT = {
	'RUN_ID'         : '0' * 3,
	'CONFIG_ID'	 : '0' * LEN_ID,
	'DIRNAME'        : 'NOT_PROCESSED',
	
	'N_LAYER_0'      : 28*28,
	'N_LAYER_1'      : 100,
	'N_LAYER_2'      : 10,

	'DELTA_T'        : 0.03,
	'SIM_STEPS'      : 1000,

	'BATCH_SIZE'     : 5,
	'MAX_EPOCHS'     : 10,

	'COUPLING_HIDDEN': 500.0,
	'COUPLING_OUT'   : 500.0,
	'LEARNING_RATE'  : 10,
	'OUTPUT_SCALAR'  : 1.5,
	'LF_HIDDEN'      : 1.0,
	'LF_OUT'         : 10.0,
	'NUM_SNIFFS'     : 1,
	'USE_BIAS'       : 0.0,
	'BETA_PHASE_2'   : 0.0,

	'GNA'            : 120.0,
	'GK'             : 36.0,
	'GL'             : 0.3,
	'ENA'            : 115.0,
	'EK'             : -12.0,
	'EL'             : 10.613,
}              
               
               

# MODIFY ME
CONSTS_RANGES = {
    'LF_OUT'           : [ 0.1, 2.0 ],
    'LF_HIDDEN'        : [ 1.0, 20.0 ],
    'GNA'              : [ 100.0, 140.0 ], 
    'GK'               : [ 30.0, 42.0 ], 
    # 'NUM_SNIFFS'       : [   10],
    # 'OUTPUT_SCALAR'    : [ 2.2],
    # 'LEARNING_RATE'    : [ 10 ],
    # 'MAX_EPOCHS'       : [ 100],
    # 'COUPLING_HIDDEN'  : [ 100],
    # 'COUPLING_OUT'     : [ 500],
}

CONSTS_SWEEP_TYPE = {
	'N_LAYER_0'      : 'scalar',
	'N_LAYER_1'      : 'scalar',
	'N_LAYER_2'      : 'scalar',

	'DELTA_T'        : 'scalar',
	'SIM_STEPS'      : 'scalar',

	'BATCH_SIZE'     : 'scalar',
	'MAX_EPOCHS'     : 'scalar',

	'COUPLING_HIDDEN': 'scalar',
	'COUPLING_OUT'   : 'scalar',
	'LEARNING_RATE'  : 'log',
	'OUTPUT_SCALAR'  : 'scalar',
	'LF_HIDDEN'      : 'log',
	'LF_OUT'         : 'log',
	'NUM_SNIFFS'     : 'scalar',
	'USE_BIAS'       : 'scalar',
	'BETA_PHASE_2'   : 'scalar',

	'GNA'            : 'scalar',
	'GK'             : 'scalar',
	'GL'             : 'scalar',
	'ENA'            : 'scalar',
	'EK'             : 'scalar',
	'EL'             : 'scalar',
}

# np.logspace(-3, 1, 16, endpoint=False)



# for reading old style configs
CONFIG_KEYS_MAP = {
	'a_hidden'      : 'COUPLING_HIDDEN',
	'a_out'         : 'COUPLING_OUT',
	'm_n_in'        : 'N_LAYER_0',
	'm_n_hidden'    : 'N_LAYER_1',
	'm_n_out'       : 'N_LAYER_2',
	'm_dt'          : 'DELTA_T',
	'sim_time_ms'   : 'SIM_STEPS',
	'learning_rate' : 'LEARNING_RATE',
	'lateral_factor': 'LATERAL_FACTOR',
}



LOSS_TYPES = {
	'LOSS_REL' : 'rel',
	'LOSS_ABS' : 'abs',
	'TEST_ACCURACY' : 'test',
}

ENABLE_TESTING_DATA = True


REMOVE_COLS = []
# 'N_LAYER_0',
# 'N_LAYER_1',
# 'N_LAYER_2',
# 'DELTA_T',
# 'SIM_STEPS',
# 'BATCH_SIZE',
# 'MAX_EPOCHS',


TYPE_MAP = {
	**{
		k : type(CONSTS_DEFAULT[k])
		for k in CONSTS_DEFAULT
	},
	**{
		k : float
		for k in LOSS_TYPES
	}
}


if __name__ == '__main__':
	raise Exception('psweep.py is not meant to be run!')