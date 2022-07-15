import numpy as np
import lightgbm as lgb

from dlambdamart.dlambdamart.calculator import Calculator, MIN_ARG, MAX_ARG
from dlambdamart.dlambdamart.lambdaobj import get_unbiased_gradients_fixed_tplus, get_debiased_gradients


class DatasetWithCalculatorRanks0TPlusAndP(lgb.Dataset):
    def __init__(
        self, 
        max_ndcg_pos, 
        ranks, 
        p,
        t_plus, 
        inverse_max_dcgs=None, 
        sigmoids=None,
        logs=None,
        idx_factor=None,
        *args, 
        **kwargs
    ):
        assert len(t_plus) > 0
        lgb.Dataset.__init__(self, *args, **kwargs)
        self.calculator = Calculator(
            self.label, 
            self.get_group(), 
            max_ndcg_pos,
            inverse_max_dcgs, 
            sigmoids,
            logs,
            idx_factor
        )
        self.ranks0 = (ranks - 1).astype(int)
        self.p = p
        self.t_plus = t_plus


def unbiased_lambdarank_objective_fixed_tplus(
    preds, dataset, t_minus, verbose=False
):  
    if verbose:
        print("current scores:\n", preds)

    groups = dataset.get_group()
    
    if len(groups) == 0:
        raise Exception("Group/query data should not be empty.")
    else:
        grad = np.zeros(len(preds))
        hess = np.zeros(len(preds))
        get_unbiased_gradients_fixed_tplus(
            np.ascontiguousarray(dataset.label, dtype=np.double), 
            np.ascontiguousarray(preds),
            np.ascontiguousarray(dataset.ranks0),
            len(preds),
            np.ascontiguousarray(groups),
            np.ascontiguousarray(dataset.calculator.query_boundaries),
            len(dataset.calculator.query_boundaries) - 1,
            np.ascontiguousarray(dataset.calculator.discounts),
            np.ascontiguousarray(dataset.calculator.inverse_max_dcgs),
            np.ascontiguousarray(dataset.calculator.sigmoids),
            len(dataset.calculator.sigmoids),
            MIN_ARG,
            MAX_ARG,
            dataset.calculator.idx_factor,
            np.ascontiguousarray(dataset.calculator.logs),
            len(dataset.calculator.logs),
            MIN_ARG,
            MAX_ARG,
            dataset.calculator.idx_factor,
            dataset.p,
            len(dataset.t_plus),
            dataset.calculator.k,
            np.ascontiguousarray(grad), 
            np.ascontiguousarray(hess),
            np.ascontiguousarray(dataset.t_plus),
            np.ascontiguousarray(t_minus)
        )
        if verbose:
            print("gradient:\n", grad)
            print("hessian:\n", hess)
        return grad, hess


def debiased_lambdarank_objective_fixed_tplus(
    preds, dataset, use_pairs_with_equal_gains
):
    groups = dataset.get_group()
    
    if len(groups) == 0:
        raise Exception("Group/query data should not be empty.")
    else:
        grad = np.zeros(len(preds))
        hess = np.zeros(len(preds))
        get_debiased_gradients(
            np.ascontiguousarray(dataset.label, dtype=np.double), 
            np.ascontiguousarray(preds),
            np.ascontiguousarray(dataset.ranks0),
            len(preds),
            np.ascontiguousarray(groups),
            np.ascontiguousarray(dataset.calculator.query_boundaries),
            len(dataset.calculator.query_boundaries) - 1,
            np.ascontiguousarray(dataset.calculator.discounts),
            np.ascontiguousarray(dataset.calculator.inverse_max_dcgs),
            np.ascontiguousarray(dataset.calculator.sigmoids),
            len(dataset.calculator.sigmoids),
            MIN_ARG,
            MAX_ARG,
            dataset.calculator.idx_factor,
            np.ascontiguousarray(dataset.calculator.logs),
            len(dataset.calculator.logs),
            MIN_ARG,
            MAX_ARG,
            dataset.calculator.idx_factor,
            dataset.p,
            dataset.calculator.max_rank,
            dataset.calculator.k,
            np.ascontiguousarray(grad), 
            np.ascontiguousarray(hess),
            np.ascontiguousarray(dataset.t_plus),
            use_pairs_with_equal_gains
        )
        
        return grad, hess