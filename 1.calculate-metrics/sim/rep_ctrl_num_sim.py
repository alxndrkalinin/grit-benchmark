import pickle
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson, nbinom, lognorm

from cytominer_eval import evaluate_metrics
from cytominer_eval.transform import metric_melt, copairs_similarity
from cytominer_eval.utils.transform_utils import check_replicate_groups
from cytominer_eval.utils.operation_utils import assign_replicates, set_pair_ids

from cytominer_eval.operations import (
    replicate_reproducibility,
    precision_recall,
    grit,
    mp_value,
    enrichment,
    hitk,
)

from copairs.map import create_matcher, flatten_str_list, run_pipeline

from scripts.generate_utils import (
    generate_distribution_params,
    generate_distribution_params_differ,
    generate_features,
    aggregate_metrics_results,
    calculate_accuracy,
)


plate_col = "Metadata_Plate"
plate_map_col = "Metadata_Plate_Map"
well_col = "Metadata_Well"
pert_col = "Metadata_Perturbation"
negcon_pert_value = "negative_control"

SEED = 42
rng = np.random.default_rng(SEED)

n_plates = 2
# n_wells = 108  -> # set to n_per_plate + n_controls
n_perts = 100  # constant
# n_controls = 8
n_feats = 100  # constant
n_plate_maps = 1  # constant
n_perts_differ = 0

feature_proportions = {"gaussian": 1.0, "lognormal": 0.0, "poisson": 0.0, "nbinom": 0.0}
# features_differ = {"gaussian": 1, "lognormal": 0, "poisson": 0, "nbinom": 0}

control_params = {
    "gaussian": (0, 1),
    "lognormal": (0, 1),
    "poisson": 3,
    "nbinom": (1, 0.5),
}
differ_params = {
    "gaussian": (1, 1),
    "lognormal": (1, 1),
    "poisson": 2,
    "nbinom": (3, 0.25),
}

replicate_groups = {
    "pos_sameby": {"all": [f"{pert_col} != '{negcon_pert_value}'"], "any": []},
    "pos_diffby": {"all": [], "any": []},
    "neg_sameby": {"all": [], "any": []},
    "neg_diffby": {"all": ["Metadata_Perturbation_Type"], "any": []},
}

n_permutations = 1000
metrics_config = {
    "replicate_reproducibility": {
        "return_median_correlations": True,
        "quantile_over_null": 0.95,
        "replicate_groups": [f"{pert_col}"],
    },
    "mp_value": {
        "control_perts": [f"{negcon_pert_value}"],
        "replicate_id": f"{pert_col}",
        "rescale_pca": True,
        "nb_permutations": n_permutations,
        "random_seed": SEED,
    },
    "mean_ap": {
        "null_size": n_permutations,
        "groupby_columns": [f"{pert_col}"],
    },
}

plate_nums = [2, 3, 4]
# plate_nums = [4]

control_nums = [4, 8, 16]
# control_nums = [16]

differ_params_configs = [
    {"gaussian": (i, 1), "lognormal": (0, 1), "poisson": 3, "nbinom": (1, 0.5)} for i in range(1, 2)
]
print(differ_params_configs)

features_differ_configs = [{"gaussian": 2**i, "lognormal": 0, "poisson": 0, "nbinom": 0} for i in range(7)]
# features_differ_configs = [{"gaussian": 2**i, "lognormal": 0, "poisson": 0, "nbinom": 0} for i in range(3, 4)]
print(features_differ_configs)


# run simulations and compute metrics
gaussian_accuracy_results = []

for n_plates in plate_nums:
    print(f"\nProcessing n_plates: {n_plates}")

    for n_controls in control_nums:
        print(f"\nProcessing n_controls: {n_controls}")

        for differ_params in differ_params_configs:
            print(f"\nProcessing differ_params: {differ_params}")

            pert_params, differ_perts = generate_distribution_params_differ(
                n_perts, n_perts_differ, control_params, differ_params, rng
            )

            for features_differ in features_differ_configs:
                print(f"\nProcessing features_differ: {features_differ}")

                dframe, feats = generate_features(
                    n_plates,
                    n_perts + n_controls,
                    n_perts,
                    n_controls,
                    n_feats,
                    n_plate_maps,
                    feature_proportions,
                    control_params,
                    pert_params,
                    rng,
                    features_differ=features_differ,
                    differ_params=differ_params,
                    resample_perts=False,
                )

                metrics_results = evaluate_metrics(
                    profiles=pd.concat([dframe, feats], axis=1),
                    features=feats.columns,
                    meta_features=dframe.columns,
                    replicate_groups=replicate_groups,
                    metrics_config=metrics_config,
                    use_copairs=True,
                )

                acc = calculate_accuracy(metrics_results)
                acc.update(
                    {
                        "n_plates": n_plates,
                        "n_controls": n_controls,
                        "features_differ": features_differ["gaussian"],
                        "differ_params": differ_params["gaussian"][0],
                    }
                )
                print(acc)
                gaussian_accuracy_results.append(acc)

print(gaussian_accuracy_results)

with open("rep_ctrl_num_sim_results.pkl", "wb") as f:
    pickle.dump(gaussian_accuracy_results, f)