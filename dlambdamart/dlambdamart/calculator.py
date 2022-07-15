import numpy as np


def get_query_boundaries(groups):
    assert(len(groups) > 0)
    query_boundaries = [0] + list(np.cumsum(groups))
    
    return query_boundaries


N_BINS = 1024 * 1024 * 8
MIN_ARG = -25
MAX_ARG = 25


class Calculator:
    def __init__(
        self, 
        gains, 
        groups, 
        k, 
        inverse_max_dcgs=None, 
        sigmoids=None,
        logs=None,
        idx_factor=None
    ):       
        self.query_boundaries = get_query_boundaries(groups)
        self.gains = gains
        self.k = k
        self.max_rank = np.max(groups)
        self.discounts = Calculator._fill_discount_table(
            self.max_rank
        )
        
        if inverse_max_dcgs is None:
            self.inverse_max_dcgs = Calculator._fill_inverse_max_dcg_table(
                self.gains, 
                self.query_boundaries,
                self.discounts,
                k
            )
        else:
            self.inverse_max_dcgs = inverse_max_dcgs

        if sigmoids is None or logs is None or idx_factor is None:
            self.sigmoids, self.logs, self.idx_factor = \
                Calculator._fill_sigmoid_and_log_table(
                    N_BINS, 
                    MIN_ARG, 
                    MAX_ARG
                )
        else:
            self.sigmoids, self.logs, self.idx_factor = \
                sigmoids, logs, idx_factor


    def get_sigmoid(self, score):
        if score <= MIN_ARG:
            return self.sigmoids[0]
        elif score >= MAX_ARG:
            return self.sigmoids[-1]
        else:
            return self.sigmoids[int((score - MIN_ARG) * self.idx_factor)]
    

    def get_log(self, score):
        if score <= MIN_ARG:
            return self.logs[0]
        elif score >= MAX_ARG:
            return self.logs[-1]
        else:
            return self.logs[int((score - MIN_ARG) * self.idx_factor)]
    

    def compute_ndcg(self, scores):
        dcgs = np.zeros(len(self.query_boundaries) - 1)
        
        for i in range(len(self.query_boundaries) - 1):
            order = np.argsort(scores[self.query_boundaries[i]:self.query_boundaries[i + 1]])[::-1]
            g = np.array(self.gains[self.query_boundaries[i]:self.query_boundaries[i + 1]])[order][:self.k]
            dcgs[i] = np.sum(g * self.discounts[1:(len(g) + 1)])
        return np.mean(dcgs * self.inverse_max_dcgs)


    @staticmethod
    def _fill_discount_table(max_group_length):
        discounts = np.zeros(1 + max_group_length)
        m = max_group_length
        discounts[1:(1 + m)] = 1 / np.log2(1 + np.arange(1, m + 1))
        return discounts


    @staticmethod
    def _fill_inverse_max_dcg_table(gains, query_boundaries, discounts, k):
        inverse_max_dcgs = np.zeros(len(query_boundaries) - 1)
        
        for i in range(len(query_boundaries) - 1):
            g = np.sort(gains[query_boundaries[i]:query_boundaries[i + 1]])[::-1][:k]
            assert(len(discounts) > len(g))
            assert(sum(g) > 0)
            max_dcg = np.sum(g * discounts[1:(len(g) + 1)])
            inverse_max_dcgs[i] = 1 / max_dcg
            
        return inverse_max_dcgs


    @staticmethod
    def _fill_sigmoid_and_log_table(N_BINS, MIN_ARG, MAX_ARG):
        sigmoid_idx_factor = N_BINS / (MAX_ARG - MIN_ARG)       
        sigma = 1.0  # 2.0
        args = MIN_ARG + np.arange(N_BINS) / sigmoid_idx_factor
        sigmoids = sigma / (1 + np.exp(sigma * args))
        logs = np.log(1 + np.exp(-sigma * args))

        return sigmoids, logs, sigmoid_idx_factor