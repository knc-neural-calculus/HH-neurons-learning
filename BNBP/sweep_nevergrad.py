from os import listdir, system
from os.path import isfile, join
import sys
from typing import *
from copy import deepcopy
from time import sleep
from concurrent import futures

from hashlib import md5
import numpy as np
import nevergrad as ng

import psweep.psweep as ps
from psweep.gen_configs import dict_to_string,collapse_dict

Num = Union[float,int]


# NOTE: these global vars are changed thru command line args
# interval (seconds) between checking that the training is done
WALLTIME_WAIT : float = 1.0
# maximum steps of `WALLTIME_WAIT` before aborting
WALLTIME_LIMIT : int = 10000

# GLOBAL_ID_COUNTER = 1

SWEEP_TYPE_TO_NG_FUNC = {
	'scalar' : ng.p.Scalar,
	'log' : ng.p.Log,
}



def list_to_bounding_kwargs(lst : List[Num]) -> Dict[str,Num]:
	return {
		'lower' : min(lst),
		'upper' : max(lst)
	}


def setup_instr(
		ranges : Dict[str,List[float]] = ps.CONSTS_RANGES,
		sweep_type : Dict[str,str] = ps.CONSTS_SWEEP_TYPE,
	) -> ng.p.Instrumentation:

	# init dict
	out_kwargs = dict()
	
	# each key is the name of a variable we want to pass to ng
	for key in ranges:
		# ngFunc takes a lower and upper bound,
		# and returns an object that is to be passed to ng.p.Instrumentation
		ngFunc = SWEEP_TYPE_TO_NG_FUNC[sweep_type[key]]
		# give ngFunc those bounds, and store in dict
		kw_temp = list_to_bounding_kwargs(ranges[key])
		out_kwargs[key] = ngFunc(**kw_temp)
	
	# pass that dict of objects to the constructor
	return ng.p.Instrumentation(**out_kwargs)




def create_config(
		params : Dict[str, Num],
		run_ID : str,
		cfg_dir : str,
		default_data : ps.cn_Dict = ps.CONSTS_DEFAULT,
		default_order : Sequence[ps.t_Key] = ps.CONSTS_DEFAULT_KEYS,
	) -> str:

	# read defaults, update with given values
	c = deepcopy(default_data)
	c.update(params)

	c['RUN_ID'] = run_ID

	c['CONFIG_ID'] = 'NULL'
	c['DIRNAME'] = 'NULL'

	hexdig = md5(str(collapse_dict(c, default_order)).encode('utf-8')).hexdigest()
	cfg_ID = str( int(hexdig, 16) % int(10**ps.LEN_ID) )
	# print('CONFIG_ID:\t' + cfg_ID)
	c['CONFIG_ID'] = cfg_ID
	# create filename
	fname = '%s_ID%s' % (c['RUN_ID'], c['CONFIG_ID'])
	c['DIRNAME'] = fname
	fname = cfg_dir + fname + '.txt'

	# convert each combo to string and save
	with open(fname, 'w') as fout:
		print(
			dict_to_string(c, default_data, default_order),
			file = fout,
		)
	# print('> created file:\t' + fname)

	return cfg_ID



def run_on_config_sbatch(
		cfg_ID : str,
		run_ID : str,
		cfg_dir : str,
	) -> None:

	cmd = "sbatch --job-name=HH_" + run_ID + "_ID" + cfg_ID + " myJobIndividual.sh "
	cmd += " " + str(cfg_ID)
	cmd += " " + str(cfg_ID)
	cmd += " " + run_ID

	# print('CALLING:\t' + cmd)
	system(cmd)


def run_on_config_basic(
		cfg_ID : str,
		run_ID : str,
		cfg_dir : str,
	):

	cmd = "./hh_psweep 1 {cfg_ID} {run_ID}".format(cfg_ID = cfg_ID, run_ID = run_ID)
	print('CALLING:\t' + cmd)
	system(cmd)


def read_accuracy(
		dirname : str,
	):

#	print('> reading file:\t')

	i = 0
	while not isfile(dirname + 'DONE.txt'):
		if i > WALLTIME_LIMIT:
			print('\n\n\nABORTING: process took too long. returning accuracy of 0 \n%s' % dirname)
			return 0.0

		# print('waiting...\ti=\t%i' % i, end = '\r')
		sleep(WALLTIME_WAIT)
		i += 1 

#	print('\n> found file!')

	try:
		data = np.genfromtxt(dirname + 'percent0.txt')
	except ValueError:
		return 0.0
	
	accuracy = data[-1, -1]

	print('> dirname:\t%s\n\taccuracy:\t%s' % (dirname, str(accuracy)))

	return accuracy


def read_loss(
		dirname : str,
		last_n : int = 5,
	):

	# print('> reading file:\t')

	i = 0
	while not isfile(dirname + 'DONE.txt'):
		if i > WALLTIME_LIMIT:
			print('\n\n\nABORTING: process took too long. returning NAN loss \n%s' % dirname)
			return float('nan')

		# print('waiting...\ti=\t%i' % i, end = '\r')
		sleep(WALLTIME_WAIT)
		i += 1 

	# print('\n> found file!')

	try:
		data = np.genfromtxt(dirname + 'loss.txt', delimiter=',')
	except ValueError:
		return float('nan')

	try:
		data = data[:,:-1]
	except:
		return float('nan')
	
	avg_final_loss = np.average(data[-last_n:])

	print('> dirname:\t%s\n\tloss:\t%s' % (dirname, str(avg_final_loss)))

	return avg_final_loss


def eval_parameter_set(
		params : Dict[str, Num],
		run_ID : str = 'NG_CROSS',
		cfg_dir : str = 'psweep/config/',
		default_data : dict = ps.CONSTS_DEFAULT,
        default_order : Sequence = ps.CONSTS_DEFAULT_KEYS,
	):

	# create the config file
	cfg_ID = create_config(
		params = params,
		run_ID = run_ID,
		cfg_dir = cfg_dir,
		default_data = default_data,
		default_order = default_order,
	)

	# run the C++ code
	# run_on_config(
	run_on_config_sbatch(
		cfg_ID = cfg_ID,
		run_ID = run_ID,
		cfg_dir = cfg_dir,
	)

	# read the output accuracy
#	loss = read_loss('../../psweep_data/{run_ID}_ID{cfg_ID}/'.format(cfg_ID = cfg_ID, run_ID = run_ID))
	percent = read_accuracy('../../psweep_data/{run_ID}_ID{cfg_ID}/'.format(cfg_ID = cfg_ID, run_ID = run_ID))
	return 100 - percent


def eval_wrapper(**kwargs):
	loss = eval_parameter_set(
		params = kwargs,
	)
	return loss




def read_cmd(
		argv : List[str],
		alias : Dict[str,str] = dict(),
		defaults : Dict[str,str] = dict(),
		typecast : Dict[str, Callable[[str], Any]] = dict(),
		to_strip : str = '-',
		splitchar : str = '=',
	) -> Dict[str, Any]:
	"""parses command line args

	splits a list of strings '--flag=val' with combined flags and args
	into a dict: `{'flag' : 'val'}` (lstrips the dashes)

	if flag 'flag' passed without value returns for that flag: `{'flag' : True}`

	Args:
		argv (List[str]): input argument list
		alias (Dict[str,str], optional): aliases for shorthand of flags. Defaults to dict().
		defaults (Dict[str,str], optional): default values. Defaults to dict().
		typecast (Dict[str, Callable[[str], Any]], optional): callables to apply to the given flags (and their aliases). Defaults to dict().
		to_strip (str, optional): this will be stripped from start of every argument. Defaults to '-'.
		splitchar (str, optional): separator beteen flag and value. Defaults to '='.

	Raises:
		KeyError: if flag passed more than once (given aliases)

	Returns:
		Dict[str, Any]: flags mapping to their values. aliases handled through duplication of values, callables applied to all variants
	"""

	output = deepcopy(defaults)
		
	# read the flags
	for x in argv:
		x = x.lstrip(to_strip)
		pair = x.split(splitchar,1)
		if len(pair) < 2:
			pair = pair + [True]

		if (pair[0] not in output) or (pair[0] in defaults):
			output[pair[0]] = pair[1]
		else:
			raise KeyError(
				'same flag passed more than once:\t%s'
				% (pair[0])
			)

	# make copies for aliases
	copied_aliases = dict()
	for key,val in output.items():
		if key in alias:
			if (alias[key] in output) and (key in defaults and output[alias[key]] != defaults[key]):
				raise KeyError(
					'same flag passed more than once (alias):\t%s,\t%s'
					% (key, alias[key])
				)
			else:
				copied_aliases[alias[key]] = val
	output.update(copied_aliases)

	# invert aliases to convert all types
	aliases_inv : Dict[str,Set[str]] = dict()
	for key,val in alias.items():
		if val in aliases_inv:
			aliases_inv[val].add(key)
		else:
			aliases_inv[val] = set((key, val))


	# cast types
	for key,func in typecast.items():
		for k2 in aliases_inv[key]:
			if k2 in output:
				output[k2] = func(output[k2])

	return output


def run_nevergrad_old(argv : List[str]):
	global NG_BUDGET
	global WALLTIME_WAIT
	global WALLTIME_LIMIT
	global NG_WORKERS

	cmds = read_cmd(
		argv,
		alias = {
			'b' : 'budget',
			'w' : 'walltime-wait',
			'L' : 'walltime-limit',
			'n' : 'num-workers',
		},
		typecast = {
			'budget' : int,
			'walltime-wait' : float,
			'walltime-limit' : int,
			'num-workers' : int,
		},
		defaults = {
			'budget' : 10, # number of samples nevergrad gets to try
			'walltime-wait' : WALLTIME_WAIT,
			'walltime-limit' : WALLTIME_LIMIT,
			'num-workers' : 1, # number of workers
		},
	)
	
	# write to global vars,
	# because passing it all the way down to the relevant function is a hassle
	WALLTIME_WAIT = cmds['walltime-wait']
	WALLTIME_LIMIT = cmds['walltime-limit']

	# echo settings
	print('running with settings:')
	print('\t{k}  \t: {v}'.format(k = 'budget', v = cmds['budget']))
	print('\t{k}\t: {v}'.format(k = 'walltime-wait', v = cmds['walltime-wait']))
	print('\t{k}\t: {v}'.format(k = 'walltime-limit', v = cmds['walltime-limit']))
	print('\t{k}\t: {v}'.format(k = 'num-workers', v = cmds['num-workers']))

	parametrization = setup_instr()
	optimizer = ng.optimizers.NGOpt(
		parametrization = parametrization,
		budget = cmds['budget'],
		num_workers = cmds['num-workers'],
	)
	
	with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
		recc = optimizer.minimize(eval_wrapper, executor=executor, batch_mode=False)

	with open('NG_out.txt', 'w') as f:
		print(recc.kwargs, file = f)
		print(recc, file = f)



def run_nevergrad(
		ng_budget : int, 
		walltime_wait : float = WALLTIME_WAIT, 
		walltime_limit : int = WALLTIME_LIMIT, 
		ng_workers : int = 1,
		file_out : str = 'NG_out.txt',
		ranges : Dict[str,List[float]] = ps.CONSTS_RANGES,
		sweep_type : Dict[str,str] = ps.CONSTS_SWEEP_TYPE,
	):
	"""tries to optimize hyperparameters using nevergrad
	
	### Parameters:
	 - `ng_budget : int`   
	   budget of how many hyperparameter sets to try
	 - `walltime_wait : float`   
	   interval (seconds) between checking that the training is done
	   (defaults to `WALLTIME_WAIT`)
	 - `walltime_limit : int`   
	   maximum steps of `WALLTIME_WAIT` before aborting
	   (defaults to `WALLTIME_LIMIT`)
	 - `ng_workers : int`   
	   number of nevergrad workers
	   (defaults to `1`)
	 - `file_out : str`   
	   where to save the nevergrad output
	   (defaults to `'NG_out.txt'`)
	 - `ranges : Dict[str,List[float]]`   
	   ranges for which hyperparameters to check (psweep/psweep.py)
	   (defaults to `ps.CONSTS_RANGES`)
	 - `sweep_type : Dict[str,str]`   
	   type to sweep (logarithmic, scalar, int, etc)
	   (defaults to `ps.CONSTS_SWEEP_TYPE`)
	"""
	global NG_BUDGET
	global WALLTIME_WAIT
	global WALLTIME_LIMIT
	global NG_WORKERS
	
	# write to global vars,
	# because passing it all the way down to the relevant function is a hassle
	WALLTIME_WAIT = walltime_wait
	WALLTIME_LIMIT = walltime_limit

	# echo settings
	print('running with settings:')
	print('\t{k}  \t: {v}'.format(k = 'budget', v = ng_budget))
	print('\t{k}\t: {v}'.format(k = 'walltime-wait', v = walltime_wait))
	print('\t{k}\t: {v}'.format(k = 'walltime-limit', v = walltime_limit))
	print('\t{k}\t: {v}'.format(k = 'num-workers', v = ng_workers))

	parametrization = setup_instr(
		ranges = ranges,
		sweep_type = sweep_type,
	)

	optimizer = ng.optimizers.NGOpt(
		parametrization = parametrization,
		budget = ng_budget,
		num_workers = ng_workers,
	)
	
	with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
		recc = optimizer.minimize(eval_wrapper, executor=executor, batch_mode=False)

	with open(file_out, 'w') as f:
		print(recc.kwargs, file = f)
		print(recc, file = f)


if __name__ == "__main__":
	import fire
	fire.Fire(run_nevergrad)
