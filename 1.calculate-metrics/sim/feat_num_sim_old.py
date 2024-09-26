import pickle
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson, nbinom, lognorm

from cytominer_eval import evaluate_metrics, evaluate_metric
from cytominer_eval.transform import metric_melt, copairs_similarity
from cytominer_eval.utils.transform_utils import check_replicate_groups
from cytominer_eval.utils.operation_utils import assign_replicates, set_pair_ids


from scripts.generate_utils import (
    generate_distribution_params,
    generate_distribution_params_differ,
    generate_features,
    aggregate_metrics_results,
    calculate_accuracy,
    l2_normalize
)


plate_col = "Metadata_Plate"
plate_map_col = "Metadata_Plate_Map"
well_col = "Metadata_Well"
pert_col = "Metadata_Perturbation"
negcon_pert_value = "negative_control"

SEED = 42

n_plates = 2
# n_wells = 108  -> # set to n_per_plate + n_controls
n_perts = 100  # constant
n_controls = 8
# n_feats = 100  # constant -> set in the config below
n_plate_maps = 1  # constant
n_perts_differ = 0  # constant

# feature_proportions = {"gaussian": 1.0, "lognormal": 0.0, "poisson": 0.0, "nbinom": 0.0}
# features_differ = {"gaussian": 8, "lognormal": 0, "poisson": 0, "nbinom": 0}

control_params = {
    "gaussian": (0, 1),
    "lognormal": (0, 1),
    "poisson": 3,
    "nbinom": (1, 0.5),
}
differ_params = {
    "gaussian": (1, 1),
    "lognormal": (0, 1),
    "poisson": 3,
    "nbinom": (1, 0.5),
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

distances = ["cosine", "euclidean"]

normalize_rows = False

# plate_nums = [2, 3, 4]
# plate_nums = [3]

# control_nums = [4, 8, 16]
# control_nums = [8]

plates_controls = {
    2: [18],
    # 2: [6, 12, 18],
    # 3: [4, 8, 12],
    # 4: [3, 6, 9],
}

# n_feats_config = [100, 200, 500, 1000, 2500, 5000]
# n_feats_config = [500, 1000, 2500]
n_feats_config = [1000]
print(n_feats_config)

feature_proportions = {"gaussian": 1.0, "lognormal": 0.0, "poisson": 0.0, "nbinom": 0.0}

# differ_params_configs = [
#     {"gaussian": (i, 1), "lognormal": (i, 1), "poisson": i+1, "nbinom": (i, 0.25)} for i in range(1, 2)
# ] + [
#     {"gaussian": (1, i), "lognormal": (1, i), "poisson": i+1, "nbinom": (1, i)} for i in range(1, 2)
# ]
def differ_params():
    return {
        "gaussian": (1, 1),
        "lognormal": (1, 1),
        "poisson": 1,
        "nbinom": (1, 0.25),
    }
differ_params_configs = [differ_params()]
print(differ_params_configs)

features_differ_configs = [{"gaussian": 2**i, "lognormal": 0, "poisson": 0, "nbinom": 0} for i in range(4,6)]
# features_differ_configs = [{"gaussian": 2**i, "lognormal": 0, "poisson": 0, "nbinom": 0} for i in range(4, 5)]
print(features_differ_configs)


# run simulations and compute metrics
feature_num_accuracy_results = []

for n_feats in n_feats_config:
    print(f"\nProcessing n_feats: {n_feats}")

    for n_plates in plates_controls.keys():
        print(f"\nProcessing n_plates: {n_plates}")

        for n_controls in plates_controls[n_plates]:
            print(f"\nProcessing n_controls: {n_controls}")

            for differ_params in differ_params_configs:
                print(f"\nProcessing differ_params: {differ_params}")

                pert_params, differ_perts = generate_distribution_params_differ(
                    n_perts, n_perts_differ, control_params, differ_params, SEED
                )

                for features_differ in features_differ_configs:
                    features_differ = {dist: int(num * (n_feats/100)) for dist, num in features_differ.items()}
                    print(f"\nProcessing features_differ: {features_differ} of {n_feats} ({features_differ['gaussian'] / n_feats})")

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
                        SEED,
                        features_differ=features_differ,
                        differ_params=differ_params,
                        resample_perts=False,
                    )

                    if normalize_rows:
                        feats = l2_normalize(feats)

                    metrics_results = evaluate_metrics(
                        profiles=pd.concat([dframe, feats], axis=1),
                        features=feats.columns,
                        meta_features=dframe.columns,
                        replicate_groups=replicate_groups,
                        metrics_config=metrics_config,
                        distance_metric="euclidean",
                        use_copairs=True,
                    )

                    cosine_map, _ = evaluate_metric(
                        profiles=pd.concat([dframe, feats], axis=1),
                        features=feats.columns,
                        meta_features=dframe.columns,
                        replicate_groups=replicate_groups,
                        operation="mean_ap",
                        operation_kwargs=metrics_config["mean_ap"],
                        distance_metric="cosine",
                        use_copairs=True,
                    )

                    print((metrics_results['mean_ap'].drop_duplicates()["p_value"] < 0.05).sum())
                    print((cosine_map.drop_duplicates()["p_value"] < 0.05).sum())
                    print((metrics_results['mp_value'].drop_duplicates()["mp_value"] < 0.05).sum())

#                     acc = calculate_accuracy(metrics_results)
#                     acc.update(
#                         {
#                             "n_feats": n_feats,
#                             "n_plates": n_plates,
#                             "n_controls": n_controls,
#                             "features_differ": features_differ["gaussian"],
#                             "differ_params": differ_params["gaussian"][0],
#                         }
#                     )

#                     cosine_map = cosine_map.drop_duplicates().reset_index(drop=True)
#                     acc.update({
#                         "cosine_mean_ap": (cosine_map["mean_ap"] == 1.0).sum() / cosine_map.shape[0],
#                         "cosine_mean_ap_p_value": (cosine_map["p_value"] < 0.05).sum() / cosine_map.shape[0],
#                     })

#                     print(acc)
#                     feature_num_accuracy_results.append(acc)
#                     with open("feat_num_sim_results_euclidean.pkl", "wb") as f:
#                         pickle.dump(feature_num_accuracy_results, f)

# print(feature_num_accuracy_results)
# with open("feat_num_sim_results_euclidean.pkl", "wb") as f:
#     pickle.dump(feature_num_accuracy_results, f)