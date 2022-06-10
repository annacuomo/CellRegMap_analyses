"""Run tests on simulated data."""
# ===============================================================================
# Imports
# ===============================================================================
import argparse
import time

import numpy as np
import pandas as pd

from pandas_plink import read_plink1_bin

from limix.qc import quantile_gaussianize
from cellregmap import CellRegMap
from cellregmap._cellregmap import get_L_values
from cellregmap._simulate import column_normalize
from crm_fixed import (
    run_cellregmap_fixed_single_env,
    run_cellregmap_fixed_multi_env,
)

import scanpy as sc


def parse_args():
    parser = argparse.ArgumentParser(description="Run test.")
    parser.add_argument("-M", type=str, required=True, help="Model to run")
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="anndata file with raw expression counts (cells x genes)",
    )
    parser.add_argument(
        "-K",
        type=str,
        required=True,
        help=".csv file for kinship matrix (individuals x individuals)",
    )
    parser.add_argument(
        "-C", type=str, required=True, help="Key in adata.obsm for context definition."
    )
    parser.add_argument(
        "-G", type=str, required=True, help="plink or .csv file with genotypes"
    )
    parser.add_argument("-O", type=str, required=True, help="out prefix")
    parser.add_argument(
        "--y_range",
        type=str,
        default="all",
        help="Optional range of Y to test, e.g. '3,4' for range(3, 4) or '4' for range(4)",
    )
    parser.add_argument(
        "--n_C", type=int, default=10, help="Number of contexts for background"
    )
    parser.add_argument(
        "--n_GxC", type=int, default=10, help="Number of contexts for GxC"
    )
    parser.add_argument(
        "--permute",
        action="store_true",
        default=False,
        help="Permute coordinates appropriately",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    model = args.M
    out_prefix = args.O

    ############################################################################
    # LOAD DATA
    ############################################################################
    adata = sc.read(args.adata)
    if args.y_range == "all":
        y_range = [adata.shape[1]]
    else:
        y_range = [int(x) for x in args.y_range.split(", ")]
    if len(y_range) > 2:
        raise ValueError("y_range expects two integers.")
    y_range = range(*y_range)
    adata = adata[:, y_range]

    K = pd.read_csv(args.K, index_col=0)
    donors = adata.obs['donor_long_id'].unique()
    K = K.loc[donors, donors]

    if args.C == "mofa":
        C = np.asarray(adata.obsm["X_%s" % args.C])
    elif args.C in ["leiden12", "leiden24", "day"]:
        C = pd.get_dummies(adata.obs[args.C]).to_numpy()
    else:
        raise ValueError("Unknown context")

    n_C = args.n_C
    n_GxC = args.n_GxC

    if args.G.endswith('.bed'):
        G = read_plink1_bin(args.G)
        G = pd.DataFrame(G.values, index=G.sample, columns=G.snp)
    elif args.G.endswith('.csv'):
        G = pd.read_csv(args.G, index_col=0)
    else:
        raise ValueError('Unrecognized genotype file format')

    ids = range(adata.shape[0])
    if args.permute:
        df = pd.DataFrame(adata.obs["donor_long_id"])
        df["ids"] = range(df.shape[0])
        ids = (
            df.set_index("donor_long_id")
            .loc[rng.permutation(df["donor_long_id"].unique())]["ids"]
            .to_numpy()
        )
    ############################################################################
    # PREPROCESS EXPRESSION AND CONTEXTS
    ############################################################################
    C_test = C[:, :n_GxC]  # used for testing
    C_bg = C[:, :n_C]  # background
    C_test = C_test / C_test.shape[1]
    C_bg = C_bg / C_bg.shape[1]

    Y = column_normalize(quantile_gaussianize(np.log(np.asarray(adata.X) + 1)))
    ############################################################################
    # FACTORIZE AND EXPAND KINSHIP
    ############################################################################
    Lk = (
        pd.DataFrame(np.linalg.cholesky(K.to_numpy()), index=K.index)
        .loc[adata.obs["donor_long_id"].tolist()]
        .to_numpy()
    )
    Ls = get_L_values(Lk, C_bg)

    del K
    ############################################################################
    # META INFO
    ############################################################################
    var_info = adata.var["snpID"]
    donor_info = adata.obs["donor_long_id"]
    del adata

    def run_test(i, model):
        # run test for gene i
        y = Y[:, i].reshape(-1, 1)
        W = np.ones_like(y)

        g = var_info.index[i]
        snps = var_info.loc[g].split(",")
        G_i = G.loc[donor_info, snps].to_numpy()

        t_start = time.time()
        if model == "cellregmap":
            print(C_test.shape)
            model = CellRegMap(
                y=y,
                W=W,
                E=C_test,
                E1=C_bg,
                Ls=Ls,
            )
            pv = model.scan_interaction(G=G_i, idx_G=ids)[0]
        elif model == "cellregmap-association":
            model = CellRegMap(
                y=y,
                W=W,
                E=C_test,
                E1=C_bg,
                hK=Lk,
            )
            pv = model.scan_association(G=G_i[ids, :])[0]
        elif model == "cellregmap-association2":
            model = CellRegMap(
                y=y,
                W=W,
                E=C_test,
                E1=C_bg,
                Ls=Ls,
            )
            pv = model.scan_association(G=G_i[ids, :])[0]
        elif model == "structlmm":
            model = CellRegMap(
                y=y,
                W=W,
                E=C_test,
                E1=C_bg,
            )
            pv = model.scan_interaction(G=G_i, idx_G=ids)[0]
        elif model == "cellregmap-fixed-single-env":
            pv = run_cellregmap_fixed_single_env(
                y=y, W=W, E=C_test, E1=C_bg, Ls=Ls, G=G_i, idx_G=ids
            )
        elif model == "cellregmap-fixed-multi-env":
            pv = run_cellregmap_fixed_multi_env(
                y=y, W=W, E=C_test, E1=C_bg, Ls=Ls, G=G_i, idx_G=ids
            )
        elif model == "test":
            time.sleep(1)
            pv = np.ones(G_i.shape[1])
        else:
            raise ValueError("Unknown model %s" % model)
        time_elapsed = time.time() - t_start
        return np.asarray(pv, float), time_elapsed

    print("Testing %d gene(s) ... " % Y.shape[1])
    start_time = time.time()
    pvals = list()
    time_elapsed = list()
    for i in range(Y.shape[1]):
        results = run_test(i, model=model)
        pvals.append(results[0].reshape(-1, 1))
        time_elapsed.append(results[1])
    print("Done (%.2f minutes)" % ((time.time() - start_time) / 60))

    pvals = np.concatenate(pvals).ravel()

    # save
    pd.DataFrame(pvals).to_csv(out_prefix + "_pvals.txt", header=False)
    pd.DataFrame(time_elapsed).to_csv(out_prefix + "_runtime.txt", header=False)


if __name__ == "__main__":
    main()
