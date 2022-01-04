"""Main simulation script.

By default, i.e. when run as 'python simulate.py', this script will use the 
simulation parameters defined in settings.DEFAULT_PARAMS.

Alternatively, this script can be called from snakemake, as described here:
https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#parameter-space-exploration

In this case, default parameter values from settings.DEFAULT_PARAMS will be
updated if the corresponding keyword is contained in snakemake.params.simulation.
Parameters not contained in settings.DEFAULT_PARAMS will be ignored.
"""
# ===============================================================================
# Imports
# ===============================================================================
import numpy as np
import pandas as pd

import time

from numpy_sugar import ddot
from numpy_sugar.linalg import economic_qs

from limix.qc import quantile_gaussianize
from cellregmap import CellRegMap

from settings import DEFAULT_PARAMS, FILTERED_KINSHIP_PATH
from sim_utils import (
    create_variances,
    set_causal_ids,
    sample_endo,
    create_environment_factors,
    create_kinship_factors,
    simulate_data,
)
from stats import (
    sample_nb,
    run_cellregmap_fixed,
)


def setup_params():
    SNAKEMODE = "snakemake" in locals() or "snakemake" in globals()

    def update_variable(var_name, default):
        """If run from snakemake, update parameter values."""
        if not SNAKEMODE:
            return default
        try:
            x = snakemake.params.simulation[var_name]
            if isinstance(x, pd.Series):
                x = x[0]
            return type(default)(x)  # bit hacky
        except KeyError:
            if default is None:
                raise ValueError("%s needs to be specified" % var_name)
            else:
                return default

    print("Setting parameters ...")
    params = {}
    for key, value in DEFAULT_PARAMS.items():
        # load default parameters & update if specified through snakemake
        params[key] = update_variable(key, value)
        print("%20s: %s" % (key, str(params[key])))
    # params["threads"] = (
    #     snakemake.threads if SNAKEMODE else int(input("Number of threads: "))
    # )
    # print("Running on %d cores (%d avilable)" % (params["threads"], cpu_count()))
    params["out_prefix"] = snakemake.params["out_prefix"] if SNAKEMODE else "results"

    # check parameters
    n_causal = params["n_causal_g"] + params["n_causal_gxe"] - params["n_causal_shared"]
    if n_causal > params["n_snps"]:
        raise ValueError("Number SNPs with genetic effects has to be < n_snps.")
    if (n_causal == 0) ^ (params["v0"] == 0):
        raise ValueError("v0 and specified number of causal variants disagree.")
    if (params["n_causal_g"] == 0) ^ (params["r0"] == 1):
        raise ValueError("Only one of n_causal_g, 1-r0 is zero.")
    if (params["n_causal_gxe"] == 0) ^ (params["r0"] == 0):
        raise ValueError("Only one of n_causal_gxe, r0 is zero.")
    if params["n_env_tested"] > params["n_env"]:
        raise ValueError("n_env_tested has to be smaller than n_env")
    if params["cells_per_individual"] not in ["fixed", "variable"]:
        raise ValueError("Unrecognized cells_per_individual value.")
    if params["respect_individuals"] and params["n_individuals"] > 124:
        raise ValueError("n_individuals has to be < 124 when respect_individuals.")

    return params


def main():
    # get simulation parameters (either from snakemake or settings.py)
    params = setup_params()

    # initialize random number generator
    random = np.random.default_rng(params["seed"])

    # update cells per donor if cells_per_individual == variable
    if params["cells_per_individual"] == "variable":
        n = params["n_individuals"]
        n_cells = np.linspace(1, n, n) * 2 * params["n_cells"] / (n + 1)
        params["n_cells"] = n_cells.round().astype(int)

    # create environment matrix and decomposition
    start_time = time.time()
    print("Setting up environment matrix ...", end="")
    E = sample_endo(
        params["n_env"],
        params["n_individuals"],
        params["n_cells"],
        random,
        respect_individuals=params["respect_individuals"],
    )
    env = create_environment_factors(E)
    print("done. (%ds)" % (time.time() - start_time))

    # set causal SNPs
    (g_causals, gxe_causals) = set_causal_ids(
        n_causal_g=params["n_causal_g"],
        n_causal_gxe=params["n_causal_gxe"],
        n_causal_shared=params["n_causal_shared"],
    )

    # set variances
    v = create_variances(
        r0=params["r0"],
        v0=params["v0"],
        include_noise=params["likelihood"] == "gaussian",
    )

    # create and factorize kinship matrix
    print("Setting up kinship matrix ...", end="")
    start_time = time.time()
    if params["real_genotypes"]:
        if params["n_individuals"] > 100:
            raise ValueError("n_individuals > 100. Use artificial genotypes.")
        K = pd.read_csv(FILTERED_KINSHIP_PATH, index_col=0)
        donor_ids = K.index[: params["n_individuals"]].tolist()
        K = K.iloc[: params["n_individuals"], : params["n_individuals"]].to_numpy()
    else:
        K = np.eye(params["n_individuals"])
        donor_ids = None
    Lk = create_kinship_factors(K).Lk
    # expand from donors to cells
    Lk = Lk[np.repeat(range(params["n_individuals"]), params["n_cells"]), :]
    print("done. (%ds)" % (time.time() - start_time))

    print("Precomputing factors ... ", end="")
    start_time = time.time()
    # create factors of E + K * E
    us = env.U * env.S
    Ls = tuple([ddot(us[:, i], Lk) for i in range(us.shape[1])])

    # compute QS if using fixed effect model (these models also include E as fixed effects)
    QS = None
    if params["model"].startswith("cellregmap_fixed"):
        print("Factorizing K * EE ... ")
        QS = economic_qs((Lk @ Lk.T) * (env.E @ env.E.T))
    print("done. (%ds)" % (time.time() - start_time))

    print("Running simulations for %d gene(s) ... " % params["n_genes"])
    start_time = time.time()

    def sim_and_test(random: np.random.Generator):
        # simulates data for one gene and computes p-values

        # set ids of environments with GxE effects TODO: randomization needed?
        env_gxe_active = random.choice(env.E.shape[1], params["n_env_gxe"])

        s = simulate_data(
            offset=params["offset"],
            n_individuals=params["n_individuals"],
            n_snps=params["n_snps"],
            n_cells=params["n_cells"],
            env=env,
            env_gxe_active=env_gxe_active,
            Ls=Ls,
            maf_min=params["maf_min"],
            maf_max=params["maf_max"],
            real_genotypes=params["real_genotypes"],
            donor_ids=donor_ids,
            g_causals=g_causals,
            gxe_causals=gxe_causals,
            variances=v,
            random=random,
        )

        # adjust likelihood model
        if params["likelihood"] == "gaussian":
            y = s.y
        elif params["likelihood"] == "negbin":
            y = sample_nb(mu=np.exp(s.y), phi=params["nb_dispersion"], random=random)
            y = np.log(y + 1)
        elif params["likelihood"] == "zinb":
            y = sample_nb(mu=np.exp(s.y), phi=params["nb_dispersion"], random=random)
            y *= random.binomial(1, 1 - params["p_dropout"], size=y.size)
            y = np.log(y + 1)
        elif params["likelihood"] == "poisson":
            y = random.poisson(lam=np.exp(s.y))
            y = np.log(y + 1)
        else:
            raise ValueError("Unknown likelihood %s" % params["likelihood"])

        if params["normalize"] and params["likelihood"] != "gaussian":
            y = quantile_gaussianize(y)

        # set up model & run test
        y = y.reshape(y.shape[0], 1)
        M = np.ones_like(y)

        t_start = time.time()
        if params["model"] == "cellregmap":
            model = CellRegMap(
                y=y,
                W=M,
                E=None,
                E0=env.E[:, : params["n_env_tested"]],
                E1=env.E,
                Ls=Ls,
            )
            pv = model.scan_interaction(s.G)[0]
        elif params["model"] == "cellregmap_assoc":
            model = CellRegMap(
                y=y, W=M, E=None, E0=env.E[:, : params["n_env_tested"]], E1=env.E, hK=Lk
            )
            pv = model.scan_association(s.G)[0]
        elif params["model"] == "structlmm":
            model = CellRegMap(
                y=y, W=M, E=None, E0=env.E[:, : params["n_env_tested"]], E1=env.E
            )
            pv = model.scan_interaction(s.G)[0]
        elif params["model"] == "cellregmap_fixed_single_env":
            pv = run_cellregmap_fixed(
                y=y,
                M=M,
                E0=env.E[:, : params["n_env_tested"]],
                E1=env.E,
                G=s.G,
                QS=QS,
                joint=False,
            )
        elif params["model"] == "cellregmap_fixed_multi_env":
            pv = run_cellregmap_fixed(
                y=y,
                M=M,
                E0=env.E[:, : params["n_env_tested"]],
                E1=env.E,
                G=s.G,
                QS=QS,
                joint=True,
            )
        elif params["model"] == "test":
            time.sleep(5)
            pv = np.ones(params["n_snps"])
        else:
            raise ValueError("Unknown model %s" % params["model"])
        time_elapsed = time.time() - t_start
        return np.asarray(pv, float), s.snp_ids, time_elapsed

    pvals = list()
    snp_ids = list()
    time_elapsed = list()
    for random_state in random.integers(
        0, np.iinfo(np.int32).max, size=params["n_genes"]
    ):
        results = sim_and_test(np.random.default_rng(random_state))
        pvals.append(results[0])
        snp_ids.append(results[1])
        time_elapsed.append(results[2])

    print("Done (%.2f minutes)" % ((time.time() - start_time) / 60))

    # save
    pd.DataFrame(pvals).to_csv(
        params["out_prefix"] + "_pvals.txt", header=False, index=False
    )
    pd.DataFrame(donor_ids).to_csv(
        params["out_prefix"] + "_donor_ids.txt", header=False, index=False
    )
    pd.DataFrame(snp_ids).to_csv(
        params["out_prefix"] + "_snp_ids.txt", header=False, index=False
    )
    pd.DataFrame(time_elapsed).to_csv(
        params["out_prefix"] + "_runtime.txt", header=False, index=False
    )


if __name__ == "__main__":
    main()
