"""
Stattools
"""
from stattool.splitter import hash_split, random_split, split_data
from stattool.empirical_design import check_aa, apply_pvalue_correction, calculate_fpr_FWER
from stattool.stat_test import (
    bootstrap,
    bootstrap_bucket,
    bucketization,
    calc_cuped_t_test,
    calc_t_test_lin,
    cuped,
    deltamethod,
    linearization,
    proportion_ztest,
)
from stattool.config import DEFAULT_ARGS, DEFAULT_METHOD