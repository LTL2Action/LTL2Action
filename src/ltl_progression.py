from sympy import *
from sympy.logic import simplify_logic
from sympy.logic.boolalg import And, Or, Not
import time, collections

"""
This module contains functions to progress co-safe LTL formulas such as:
    (
        'and',
        ('until','True', ('and', 'd', ('until','True','c'))),
        ('until','True', ('and', 'a', ('until','True', ('and', 'b', ('until','True','c')))))
    )
"""


def _is_prop_formula(f):
    # returns True if the formula does not contains temporal operators
    return 'next' not in str(f) and 'until' not in str(f)

def _subsume_until(f1, f2):
    if str(f1) not in str(f2):
        return False
    while type(f2) != str:
        if f1 == f2:
            return True
        if f2[0] == 'until':
            f2 = f2[2]
        elif f2[0] == 'and':
            if _is_prop_formula(f2[1]) and not _is_prop_formula(f2[2]):
                f2 = f2[2]
            elif not _is_prop_formula(f2[1]) and _is_prop_formula(f2[2]):
                f2 = f2[1]
            else:
                return False
        else:
            return False
    return False


def progress(ltl_formula, truth_assignment):
    if type(ltl_formula) == str:
        # True, False, or proposition
        if len(ltl_formula) == 1:
            # ltl_formula is a proposition
            if ltl_formula in truth_assignment:
                return 'True'
            else:
                return 'False'
        return ltl_formula
    
    if ltl_formula[0] == 'not':
        # negations should be over propositions only according to the cosafe ltl syntactic restriction
        result = progress(ltl_formula[1], truth_assignment)
        if result == 'True':
            return 'False'
        elif result == 'False':
            return 'True'
        else:
            raise NotImplementedError("The following formula doesn't follow the cosafe syntactic restriction: " + str(ltl_formula))

    if ltl_formula[0] == 'and':
        res1 = progress(ltl_formula[1], truth_assignment)
        res2 = progress(ltl_formula[2], truth_assignment)
        if res1 == 'True' and res2 == 'True': return 'True'
        if res1 == 'False' or res2 == 'False': return 'False'
        if res1 == 'True': return res2
        if res2 == 'True': return res1
        if res1 == res2:   return res1
        if _subsume_until(res1, res2): return res2
        if _subsume_until(res2, res1): return res1
        return ('and',res1,res2)

    if ltl_formula[0] == 'or':
        res1 = progress(ltl_formula[1], truth_assignment)
        res2 = progress(ltl_formula[2], truth_assignment)
        if res1 == 'True'  or res2 == 'True'  : return 'True'
        if res1 == 'False' and res2 == 'False': return 'False'
        if res1 == 'False': return res2
        if res2 == 'False': return res1
        if res1 == res2:    return res1
        if _subsume_until(res1, res2): return res1
        if _subsume_until(res2, res1): return res2
        return ('or',res1,res2)
    
    if ltl_formula[0] == 'next':
        return progress(ltl_formula[1], truth_assignment)
    
    if ltl_formula[0] == 'until':
        res1 = progress(ltl_formula[1], truth_assignment)
        res2 = progress(ltl_formula[2], truth_assignment)

        if res1 == 'False':
            f1 = 'False'
        elif res1 == 'True':
            f1 = ('until', ltl_formula[1], ltl_formula[2])
        else:
            f1 = ('and', res1, ('until', ltl_formula[1], ltl_formula[2]))
        
        if res2 == 'True':
            return 'True'
        if res2 == 'False':
            return f1
        return res2 #('or', res2, f1)

