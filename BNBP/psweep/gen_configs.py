"""generates configuration files

should only take a `run_ID` - passing range data thru command line not reccomended, you should modify psweep.py
"""

import sys
from typing import *
from copy import deepcopy
from itertools import product

if __name__ == '__main__':
	sys.path.insert(0, "..")
else:
	sys.path.insert(0, "BNBP")

from psweep.psweep import *


def collapse_dict(
        data : cn_Dict,
        default_order : Union[ None, Sequence[t_Key] ],
    ) -> Tuple[Tuple[t_Key, t_Val], ...]:
    """
    collpases a (non-hashable) dict into a sequence
    if `default_order` given, then use that to order the list
    """

    if default_order is None:
        default_order = [ k for k in data ]

    output = []

    for k in default_order:
        if k in data:
            output.append((k, data[k]))
    
    return tuple(output)




def generate_and_add_ID(
        data : cn_Dict, 
        i : int
    ) -> str:
    data['CONFIG_ID'] = str(i)
    return i

# TYPE_MAP = {
#     int   : 'int',
#     float : 'double',
# }

def item_to_line_defn(
        key : t_Key,
        val : t_Val = None,
        default : cn_Dict = CONSTS_DEFAULT,
    ) -> str:
    
    # typeString = TYPE_MAP[type(default[key])]

    if val is None:
        val = default[key]

    return '%s \t=\t %s' % (
        key,
        str(val)
    )




def dict_to_string(
        data : cn_Dict,
        default : cn_Dict = CONSTS_DEFAULT,
        default_order : Sequence[t_Key] = CONSTS_DEFAULT_KEYS,
    ) -> str:

    output = []

    for k in default_order:
        output.append(item_to_line_defn(
            k, 
            data.get(k, None), 
            default,
        ) )

    return '\n'.join(output)





def dict_cartesian_product(*args : List[List[Dict[t_Key,t_Val]]]):
    """
    takes in a list of lists of dicts, returns the cartesian product
    each argument should be a list of dicts, each mapping the same key to different values
    different arguments will have different keys, but only 1 key each

    might work for some other cases but im too lazy to think about that rn
    """
    if len(args) == 1:
        # base case
        return args[0]
    else:
        a, *b_args = args

        recur_out = dict_cartesian_product(*b_args)

        output = []

        for x in recur_out:
            for y in a:
                output.append({**x, **y})

        return output





def generate_all_index_combos(
        data : cn_Dict_R = CONSTS_RANGES,
        default_data : cn_Dict = CONSTS_DEFAULT,
        default_order : Sequence[t_Key] = CONSTS_DEFAULT_KEYS,
    ) -> List[Dict[t_Key, Union[None, t_Val]]]:
    """
    generates a list of dicts,
    each dict representing a unique set of constants
    
    REVIEW : this whole function is probably not implemented in the cleanest or most efficient way
    """

    print('generating configs:')


    print('  > splitting arrays')
    # first, split the ranges dict by key
    # then split each possible value in the range into a separate dict
    data_split = [
        [
            { k : v }
            for v in data[k]
        ]
        for k in data
    ]

    print('  > product')
    # generate all combos of values
    combos = dict_cartesian_product(*data_split)

    
    print('  > merge with default, get ID')
    combos_updated = []

    i = 0
    for c in combos:
        # update with data from default dict
        c = {**default_data, **c}

        # give each one a unique ID
        generate_and_add_ID(c, i)
        
        combos_updated.append(c)
        i += 1

    print('  > done!')

    return combos_updated


def save_all_combos(
        run_ID : str,
        combos : List[Dict[t_Key, Union[None, t_Val]]],
        default_data : cn_Dict = CONSTS_DEFAULT,
        default_order : Sequence[t_Key] = CONSTS_DEFAULT_KEYS,
        directory : str = 'config/'
    ):

    print('saving configs:')
    print(len(combos))
    for c in combos:
        # create filename
        c['RUN_ID'] = run_ID
        fname = '%s_ID%s' % (c['RUN_ID'], c['CONFIG_ID'])
        c['DIRNAME'] = fname
        fname = directory + fname + '.txt'

        # print('\t' + fname)
        
        # convert each combo to string and save
        with open(fname, 'w') as fout:
            print(
                dict_to_string(c, default_data, default_order),
                file = fout,
            )






def main(run_ID : str, data : Dict[t_Key, Iterable[t_Val]] = CONSTS_RANGES):
    combos = generate_all_index_combos(
        data = data,
        default_data = CONSTS_DEFAULT,
        default_order = CONSTS_DEFAULT_KEYS,
    )

    save_all_combos(
        run_ID = run_ID,
        combos = combos,
        default_data = CONSTS_DEFAULT,
        default_order = CONSTS_DEFAULT_KEYS,
    )


if __name__ == '__main__':
    import fire
    fire.Fire(main)











