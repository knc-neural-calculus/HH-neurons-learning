import numpy as np

LEN_ID = 8



CONSTS_DEFAULT_KEYS = [
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
	'LATERAL_FACTOR',
	'OUTPUT_SCALAR',
]

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
	'MAX_EPOCHS'     : 300,

	'COUPLING_HIDDEN': 500.0,
	'COUPLING_OUT'   : 500.0,
	'LEARNING_RATE'  : 15,
	'LATERAL_FACTOR' : 5.0,
	'OUTPUT_SCALAR'  : 2.2,
}



# MODIFY ME
CONSTS_RANGES = {
	'LEARNING_RATE'  : [ 40, 50, 60, 70, 80 ],
	'LATERAL_FACTOR' : [ round(0.5 * (i + 1), 3) for i in range(30) ],
	'OUTPUT_SCALAR'  : [ 2.0 ]
}



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
