
import time

from multiprocessing import cpu_count, Process

import functools
import uuid
import traceback

from monkeys.tools.display import node_graph
from monkeys.search import require,optimize, tournament_select, next_generation
from monkeys.trees import make_input

from monkeys.typing import params, rtype, ignore
from monkeys.typing import constant


from datetime import datetime
import pandas as pd
import numpy as np
import os,sys
from copy import copy



# @ray.remote
@params('otherinfo_dict', 'datadict')
@rtype('FinalDF_wInfo')  # monkeys will, if provided a string, make a new type for us
def get_final_df_wBtInfo(btinfo, datadict):
    df = (datadict['a'] * datadict['b']).gt(0)
    return df, btinfo



def clean_tree_str(tree):
    return str(tree).replace('()', '').replace('_const_', '').\
        replace('datadict', 'data').replace('otherinfo_dict', 'other_info').replace("[",'\"').replace("]",'\"')

def some_function(outdf,other_info,data):
    return (data['a'] - outdf).abs().sum().sum()


def score_iml(best_equation_str, pid, inputs):
    '''

    :param best_equation_str: str to evaluate
    :param pid: parrallel_eval make pids
    :param inputs: inputs = (data, other_info) in __main__
    :return:
    '''

    # these 2 local variable used by following eval function !!!!!
    data, other_info = inputs
    # clean equation so python interpreter can recognize the varnames 'data' and 'other_info'
    best_equation = clean_tree_str(best_equation_str)
    print(best_equation)
    print('start eval {}'.format(best_equation))

    try:
        outdf, other_info = eval(best_equation)  # (datadict=data)
    except Exception as e:
        with open('eval_err.log','a') as h:
            h.write('{} {}\n'.format(datetime.now(), best_equation_str))
            h.write('{}\n'.format(e))
            h.write('{}\n'.format(traceback.format_exc()))
        print(e)
        print('******',best_equation_str)
        #
        # print(traceback.format_exc())

    print('end eval {}'.format(best_equation))

    try:
        res_score = some_function(outdf,other_info,data)
        # print(res_df)
        print('score:{}'.format(res_score))


    except Exception as e:
        print(e)
        print(traceback.format_exc())
        res_score = 0

    import gc
    gc.collect()
    # print("res_score:{}".format(res_score))
    # return sharpe ratio as fitness function
    return best_equation_str, res_score, pid




def data_loader(dlst,oinfo):
    d = dict()
    d['other_info'] = copy(oinfo)
    for _key in dlst:
        d[_key] = pd.DataFrame(np.random.random((100,20)))

    return d


def get_result_from_eq(best_equation,uu, data, other_info):
    pass



if __name__ == '__main__':

    for n in [50, 100, 500]:
        globals()['lt_n_{}'.format(n)] = constant('nlowest', n)
        globals()['gt_n_{}'.format(n)] = constant('nlargest', n)

    for q in [1, 5, 10, 20, 30]:
        globals()['lt_q_{}'.format(q)] = constant('lt_quantile', q / 100)
        globals()['gt_q_{}'.format(100 - q)] = constant('gt_quantile', 1 - q / 100)

    datadict = make_input('datadict', name='datadict')
    backtestinfo_dict = make_input('otherinfo_dict', name='otherinfo_dict')

    popsize = int(sys.argv[1])


    data_list = ['a','b']
    other_info = {
        'hold_days':5,
        'constrains':'other_constrains',
    }

    # load data
    # with multiprocessing.Manager() as manager:

    @require(datadict, backtestinfo_dict)
    @params('FinalDF_wInfo')
    def score(tree):
        # run rule unit and return stock pool
        # stock_pool, paras_string = eval(rule_unit)(data, paras)
        return str(tree)

    # multiprocessing shared memory? with LOCK very slow!!!
    data = dict()
    # data = manager.dict()
    # data = dict()
    # data = data_loader(factor_list, backtest_info)
    data.update(data_loader(data_list, other_info,))


    my_tournament_selection = functools.partial(tournament_select, selection_size=int(popsize*0.1))
    my_next_generation = functools.partial(next_generation,
                                           select_fn=my_tournament_selection,
                                           scoring_fn_iml=score_iml)
    inputs = (data, other_info)

    time.sleep(1)

    best_equation = optimize(inputs, score,
                             population_size=popsize,
                             next_generation=my_next_generation,
                             iterations=10,
                             scoring_fn_iml=score_iml,
                             max_workers=min(cpu_count() ,40),  # workers 30-40  cpu max efficiency
                             )

    uu = uuid.uuid4()
    print(str(best_equation))
    with open('best_eq', 'a') as h:
        h.write('{} || {}\n'.format(uu, str(best_equation)))

    # post processing the best equation
    get_result_from_eq(best_equation,uu, data, other_info)
