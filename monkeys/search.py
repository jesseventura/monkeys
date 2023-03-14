"""Search functionality and objective function tooling."""

from __future__ import print_function

import re
import ast
import sys
import copy
import random
import inspect
import functools
import contextlib
import collections

import numpy
import astpath
from six import iteritems, itervalues
from past.builtins import xrange

from monkeys.trees import get_tree_info, build_tree, crossover, mutate
from monkeys.exceptions import UnsatisfiableType

import gc
from multiprocessing import cpu_count, set_start_method
try:
    set_start_method('spawn')
except:
    pass
from concurrent.futures import ProcessPoolExecutor, as_completed
from progressbar import ProgressBar
import os,sys


class Optimizations(object):
    COVARIANT_PARSIMONY = object()  # Poli & McPhee 2008
    RANDOM_PARSIMONY = object()  # Poli 2003
    PSEUDO_PARETO = object()  # Engelbrecht 2002


DEFAULT_OPTIMIZATIONS = {
    Optimizations.PSEUDO_PARETO,
}


def parallel_eval(score_iml, data, tree_tree_str_dict, max_workers):
    pop_scores = list()
    pid_tree_dict = dict()


    with ProcessPoolExecutor(max_workers=max_workers,) as executor:
        plist = list()
        pid = 0
        for tree in list(tree_tree_str_dict.keys()):
            pid += 1
            pid_tree_dict[pid] = tree

            task = executor.submit(score_iml, best_equation_str=str(tree), pid=pid, inputs=data)
            plist.append(task)
        for k in ProgressBar(max_values=len(plist),
                             prefix='$$$$$$$$ generation evalutaion progress ',
                             suffix='$$$$$$')(as_completed(plist)):
            tree_str, score, pid = k.result()
            print("tree: {} || {}".format(score, tree_str))
            pop_scores.append((tree_str, score, pid))
    return pid_tree_dict, pop_scores


def tournament_select(pid_tree_dict, pop_scores, trees, scoring_fn, selection_size, requires_population=False, optimizations=DEFAULT_OPTIMIZATIONS, random_parsimony_prob=0.33, score_callback=None):
    """
    Perform tournament selection on population of trees, using the specified
    objective function for comparison, and conducting tournaments of the
    specified selection size.
    """

    _scoring_fn = scoring_fn(trees) if requires_population else scoring_fn
    # print("scoring")
    # sys.stdin.readline()


    avg_size = 0
    sizes = {}

    using_covariant_parsimony = Optimizations.COVARIANT_PARSIMONY in optimizations
    using_random_parsimony = Optimizations.RANDOM_PARSIMONY in optimizations
    using_pseudo_pareto = Optimizations.PSEUDO_PARETO in optimizations
    
    if using_covariant_parsimony or using_random_parsimony:
        sizes = {tree: get_tree_info(tree).num_nodes for tree in trees}
        avg_size = sum(itervalues(sizes)) / float(len(sizes))
    
    if using_random_parsimony:
        # scores = collections.defaultdict(lambda: -sys.maxsize)
        scores_str = collections.defaultdict(lambda: '')
        scores_str.update({
            # tree: _scoring_fn(tree)
            tree: str(tree)
            for tree in trees
            if sizes[tree] <= avg_size or random_parsimony_prob < random.random() 
        })
    else:
        # scores = {tree: _scoring_fn(tree) for tree in trees}
        scores_str = {tree: str(tree) for tree in trees}

    # TODO: parallel eva ########
    # pop_scores<list> : tree_str, score, pid
    # pid_tree_dict<map<pid, tree> >
    # pid_tree_dict, pop_scores = parallel_eval(data, scores_str)
    all_scores = {pid_tree_dict[pop_score[2]]: pop_score[1] for pop_score in pop_scores}
    scores = {k:all_scores[k] for k in scores_str.keys()}
    # max_pop_tree_score = max(pop_spcores, key=lambda x: x[1])
    # new_pop = [pid_tree_dict[max_po_tree_score[2]]]


    if using_covariant_parsimony:
        covariance_matrix = numpy.cov(numpy.array([(sizes[tree], scores[tree]) for tree in trees]).T)
        size_variance = numpy.var([sizes[tree] for tree in trees])
        c = -(covariance_matrix / size_variance)[0, 1]  # 0, 1 should be correlation... is this the wrong way around?
        scores = {tree: score - c * sizes[tree] for tree, score in iteritems(scores)}

    if using_pseudo_pareto:
        non_neg_inf_scores = [s for s in itervalues(scores) if s != -sys.maxsize]
        print('non_neg_inf_scores = [s for s in itervalues(scores) if s != -sys.maxsize]')
        # sys.stdin.readline()
        try:
            avg_score = sum(non_neg_inf_scores) / float(len(non_neg_inf_scores))
        except ZeroDivisionError:
            avg_score = -sys.maxsize
        scores = {
            tree: -sys.maxsize if score < avg_score and sizes.get(tree, 0) > avg_size else score
            for tree, score in iteritems(scores)
        }

    if callable(score_callback):
        score_callback(scores)

    gc.collect()

    while True:
        tree = max(
            random.sample(trees, selection_size),
            key=lambda t: scores.get(t, -sys.maxsize)
        )
        if scores.get(tree, -sys.maxsize) == -sys.maxsize:
            try:
                print('-sys.maxsize: build_tree')
                new_tree = build_tree_to_requirements(scoring_fn)
            except UnsatisfiableType:
                continue
        else:
            try:
                with recursion_limit(1000):
                    # print('deepcopy a tree')
                    # sys.stdin.readline()
                    new_tree = copy.deepcopy(tree)
                    # sys.stdin.readline()
            except RuntimeError:
                try:
                    print('RuntimeError: build_tree')
                    new_tree = build_tree_to_requirements(scoring_fn)
                except UnsatisfiableType:
                    continue
        yield new_tree


DEFAULT_TOURNAMENT_SELECT = functools.partial(tournament_select, selection_size=25)
        
        
def pre_evaluate(scoring_fn):
    """
    Evaluate trees before passing to the scoring function.
    """
    @functools.wraps(scoring_fn)
    def wrapper(tree):
        try:
            evaluated_tree = tree.evaluate()
        except Exception:
            return -sys.maxsize
        return scoring_fn(evaluated_tree)
    return wrapper


def minimize(scoring_fn):
    """Minimize score."""
    @functools.wraps(scoring_fn)
    def wrapper(tree):
        return -scoring_fn(tree)
    return wrapper


class AssertionReplacer(ast.NodeTransformer):
    """Transformer used in assertions_as_score."""
    
    def __init__(self, score_var_name):
        self.score_var_name = score_var_name
        self.max_score = 0
        
    def visit_Assert(self, node):
        """Replace assertions with augmented assignments."""
        self.max_score += 1
        return ast.AugAssign(
            op=ast.Add(),
            target=ast.Name(
                id=self.score_var_name,
                ctx=ast.Store()
            ),
            value=ast.Call(
                args=[node.test],
                func=ast.Name(
                    id='bool',
                    ctx=ast.Load()
                ),
                keywords=[],
                kwargs=None,
                starargs=None
            )
        )


def assertions_as_score(scoring_fn):
    """
    Create a scoring function from a multi-assert test, allotting
    one point per successful assertion.
    
    Nota bene: if used in conjunction with other decorators, must
    be the first decorator applied to the function.
    """
    score_var_name = '__score__'
    
    function_source = inspect.getsource(scoring_fn)
    initial_indentation = re.search(r'^\s+', function_source)
    if initial_indentation:
        indentation = len(initial_indentation.group())
        function_source = '\n'.join(
            line[indentation:]
            for line in
            function_source.splitlines()
        )
        
    fn_ast, = ast.parse(function_source).body
    fn_ast.body.insert(
        0,
        ast.Assign(
            targets=[ast.Name(
                id=score_var_name,
                ctx=ast.Store()
            )],
            value=ast.Num(n=0)
        )
    )
    fn_ast.body.append(
        ast.Return(
            value=ast.Name(
                id=score_var_name,
                ctx=ast.Load()
            )
        )
    )
    fn_ast.decorator_list = []
    assertion_replacer = AssertionReplacer(score_var_name)
    fn_ast = assertion_replacer.visit(fn_ast)
    
    code = compile(
        ast.fix_missing_locations(
            ast.Module(body=[fn_ast])
        ), 
        '<string>', 
        'exec'
    )
    context = {}
    exec(code, scoring_fn.__globals__, context)
    new_scoring_fn, = context.values()
    
    # Assess whether max score can be determined:
    xml_ast = astpath.file_contents_to_xml_ast(function_source)
    invalidating_ancestors = 'While', 'For'
    invalidating_expressions = (
        './/{}//Assert'.format(ancestor) 
        for ancestor in
        invalidating_ancestors
    )
    invalid = any(
        astpath.find_in_ast(xml_ast, expr)
        for expr in
        invalidating_expressions
    )
    
    if not invalid:
        new_scoring_fn.__max_score = assertion_replacer.max_score
    
    return functools.wraps(scoring_fn)(new_scoring_fn)


def build_tree_to_requirements(scoring_function, build_tree=build_tree):
    params = getattr(scoring_function, '__params', ())
    if len(params) != 1:
        raise ValueError("Scoring function must accept a single parameter.")
    return_type, = params

    for __ in xrange(99999):
        with recursion_limit(1000):
            tree = build_tree(return_type, convert=False)
        requirements = getattr(scoring_function, 'required_inputs', ())
        if not all(req in tree for req in requirements):
            continue
        return tree

    raise UnsatisfiableType("Could not meet input requirements.")


def next_generation(data,
        trees, scoring_fn,
        select_fn=DEFAULT_TOURNAMENT_SELECT,
        build_tree=build_tree_to_requirements, mutate=mutate,
        crossover_rate=0.75, mutation_rate=0.15,
        score_callback=None,
        optimizations=DEFAULT_OPTIMIZATIONS,
        scoring_fn_iml=None,
        max_workers=round(cpu_count() - 4)
    ):
    """
    Create next generation of trees from prior generation, maintaining current
    size.
    """

    # print('start next gene')
    # import time
    # sys.stdin.readline()
    # selector = select_fn(data, trees, scoring_fn, score_callback=score_callback, optimizations=optimizations)
    # print('finish next gene')
    # import time
    # sys.stdin.readline()

    pop_size = len(trees)
    # tree_strs_dict = {scoring_fn(tree):tree for tree in trees}
    pid_tree_dict, pop_scores = parallel_eval(scoring_fn_iml, data,
                                              {tree: str(tree) for tree in trees },
                                              max_workers=max_workers)
    max_pop_tree_scores = sorted(pop_scores, key=lambda x:x[1])
    max_pop_tree_score = max_pop_tree_scores[-1]
    with open('best_eq', 'a') as h:
        # log top5 scores trees
        for sc in max_pop_tree_scores[-5:]:
            h.write('{} || {}\n'.format(sc[1], sc[0]))

    new_pop = [pid_tree_dict[max_pop_tree_score[2]]]

    selector = select_fn(pid_tree_dict, pop_scores, trees, scoring_fn, score_callback=score_callback, optimizations=optimizations)

    # new_pop = [max(trees, key=scoring_fn)]
    for __ in xrange(pop_size - 1):

        gc.collect()
        print("start build next pop {}".format(__))
        # sys.stdin.readline()

        if random.random() <= crossover_rate:
            for __ in xrange(99999):
                try:
                    p1 = next(selector)
                    # print("p1 next")
                    # sys.stdin.readline()
                    p2 = next(selector)
                    # print("p2 next")
                    # sys.stdin.readline()
                    # new_pop.append(crossover(next(selector), next(selector)))
                    cp = crossover(p1, p2)
                    # print("crossover ")

                    # sys.stdin.readline()
                    new_pop.append(cp)
                    # print("append crossover")

                    # sys.stdin.readline()
                    print("new_pop.append(crossover(next(selector), next(selector)))  SATISFIED!!!!")
                    break
                except (UnsatisfiableType, RuntimeError):
                    continue
            else:

                new_pop.append(build_tree(scoring_fn))
                print(" new_pop.append(build_tree(scoring_fn))")
                # sys.stdin.readline()

        elif random.random() <= mutation_rate / (1 - crossover_rate):

            new_pop.append(mutate(next(selector)))
            print("new_pop.append(mutate(next(selector)))")
            # sys.stdin.readline()

        else:

            new_pop.append(next(selector))
            print("new_pop.append(next(selector))")
            # sys.stdin.readline()

    return new_pop


def require(*inputs):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(tree):
            if not all(i in tree for i in inputs):
                return -sys.maxsize
            return fn(tree)
        wrapper.required_inputs = inputs
        return wrapper
    return decorator


@contextlib.contextmanager
def recursion_limit(limit):
    orig_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(orig_limit)
    

def optimize(data,
            scoring_function,
            population_size=250,
            iterations=25,
            build_tree=build_tree,
            next_generation=next_generation,
            show_scores=True,
            optimizations=DEFAULT_OPTIMIZATIONS,
            scoring_fn_iml=None,
            max_workers=round(cpu_count() - 4)
    ):
    print("Creating initial population of {}.".format(population_size))
    sys.stdout.flush()

    build_to_requirements = functools.partial(
        build_tree_to_requirements,
        build_tree=build_tree,
    )
    
    population = []
    for __ in xrange(population_size):
        try:
            tree = build_to_requirements(scoring_function)
            population.append(tree)
        except UnsatisfiableType:
            raise UnsatisfiableType(
                "Could not meet input requirements. Found only {} satisfying trees.".format(
                    len(population)
                )
            )
    best_tree = [random.choice(population)]
    early_stop = []
    
    def score_callback(iteration, scores):
        if not show_scores:
            return
        
        non_failure_scores = [
            score 
            for score in 
            scores.values()
            if score != -sys.maxsize
        ]
        try:
            average_score = sum(non_failure_scores) / len(non_failure_scores)
        except ZeroDivisionError:
            average_score = -sys.maxsize
        best_score = max(scores.values())
        
        best_tree.append(max(scores, key=scores.get))
        
        print("Iteration {}:\tBest: {:.2f}\tAverage: {:.2f}".format(
            iteration + 1,
            best_score,
            average_score,
        ))
        sys.stdout.flush()
        
        if best_score == getattr(scoring_function, '__max_score', None):
            early_stop.append(True)
    
    print("Optimizing...")
    with recursion_limit(1000):
        for iteration in xrange(iterations):
            with open('{}-iterations.log'.format(os.getpid()),'a') as h:
                h.write('********** iteration {}\n'.format(iteration))

            callback = functools.partial(score_callback, iteration)
            tmp_population = next_generation(data,
                population,
                scoring_function,
                build_tree=build_to_requirements,
                mutate=mutate,
                score_callback=callback,
                optimizations=optimizations,
                scoring_fn_iml=scoring_fn_iml,
                max_workers=max_workers,
            )
            population = tmp_population

            del tmp_population
            import gc
            gc.collect()

            if early_stop:
                break

    pid_tree_dict, pop_scores = parallel_eval(scoring_fn_iml, data,
                                              {tree: str(tree) for tree in best_tree},
                                              max_workers=max_workers)
    max_pop_tree_scores = sorted(pop_scores, key=lambda x: x[1])
    max_pop_tree_score = max_pop_tree_scores[-1]
    with open('best_eq', 'a') as h:
        for sc in max_pop_tree_scores:
            h.write('{} || {}\n'.format(sc[1], sc[0]))

    # best_tree_max = max(best_tree, key=scoring_function)
    best_tree_max = pid_tree_dict[max_pop_tree_score[2]]
    return best_tree_max
