import numpy as np
import pandas as pd


def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def generate_plate_map(n_wells, n_perts, n_controls, rng, resample_perts=False, negcon_pert_value="negative_control"):
    wells = [f"w{j}" for j in range(n_wells)]
    perturbations = [f"c{k}" for k in range(n_perts)] + [negcon_pert_value] * n_controls
    shuffled_perts = rng.choice(perturbations, size=len(wells), replace=resample_perts)
    return dict(zip(wells, shuffled_perts))


def generate_distribution_params_differ(n_perts_total, n_perts_differ, control_params, differ_params, seed):
    """Assign distribution parameters for each perturbation, with some perturbations having different parameters."""
    params = {}
    rng = np.random.default_rng(seed)
    differ_perts = rng.choice(range(n_perts_total), size=n_perts_differ, replace=False)
    for i in range(n_perts_total):
        if i in differ_perts:
            params[f"c{i}"] = differ_params
        else:
            params[f"c{i}"] = control_params
    return params, differ_perts


def generate_distribution_params(n_perts, rng):
    """Generate random distribution parameters for each perturbation."""
    params = {}
    for i in range(n_perts):
        params[f"c{i}"] = {
            "gaussian": (rng.uniform(-0.5, 0.5), rng.uniform(0.5, 1.5)),
            "lognormal": (rng.uniform(-0.5, 0.5), rng.uniform(0.5, 1.5)),
            "poisson": rng.uniform(2, 4),
            "nbinom": (rng.uniform(0.75, 1.25), rng.uniform(0.4, 0.6)),
        }
    return params


def generate_features_for_perturbation(
    num_gaussian, num_lognormal, num_poisson, num_nbinom, params, rng, num_cauchy=0, features_differ=None, differ_params=None
):
    """Generate features for a perturbation using provided distribution parameters."""

    gaussian_feats, lognormal_feats, poisson_feats, nbinom_feats, cauchy_feats = [], [], [], [], []

    if features_differ is not None:
        assert differ_params is not None, "Must provide differ_params if features_differ is not None"
        
        gaussian_feats = rng.normal(
            loc=differ_params["gaussian"][0], scale=differ_params["gaussian"][1], size=features_differ.get("gaussian", 0)
        )
        lognormal_feats = rng.lognormal(
            mean=differ_params["lognormal"][0], sigma=differ_params["lognormal"][1], size=features_differ.get("lognormal", 0)
        )
        poisson_feats = rng.poisson(lam=differ_params["poisson"], size=features_differ.get("poisson", 0))
        nbinom_feats = rng.negative_binomial(n=differ_params["nbinom"][0], p=differ_params["nbinom"][1], size=features_differ.get("nbinom", 0))
        if "cauchy" in features_differ:
            cauchy_feats = rng.standard_cauchy(size=features_differ["cauchy"]) * differ_params["cauchy"][1] + differ_params["cauchy"][0]

        num_gaussian -= features_differ.get("gaussian", 0)
        num_lognormal -= features_differ.get("lognormal", 0)
        num_poisson -= features_differ.get("poisson", 0)
        num_nbinom -= features_differ.get("nbinom", 0)
        num_cauchy -= features_differ.get("cauchy", 0)

    gaussian_feats = np.concatenate([
        gaussian_feats, rng.normal(loc=params["gaussian"][0], scale=params["gaussian"][1], size=num_gaussian)
    ])
    lognormal_feats = np.concatenate([
        lognormal_feats, rng.lognormal(mean=params["lognormal"][0], sigma=params["lognormal"][1], size=num_lognormal)
    ])
    poisson_feats = np.concatenate([
        poisson_feats, rng.poisson(lam=params["poisson"], size=num_poisson)
    ])
    nbinom_feats = np.concatenate([
        nbinom_feats, rng.negative_binomial(n=params["nbinom"][0], p=params["nbinom"][1], size=num_nbinom)
    ])
    
    if num_cauchy > 0:
        cauchy_feats = np.concatenate([
            cauchy_feats, rng.standard_cauchy(size=num_cauchy) * params["cauchy"][1] + params["cauchy"][0]
        ])

    return np.concatenate([gaussian_feats, lognormal_feats, poisson_feats, nbinom_feats, cauchy_feats])


def generate_features(
    n_plates,
    n_wells,
    n_perts,
    n_controls,
    n_feats,
    n_plate_maps,
    feature_proportions,
    control_params,
    pert_params,
    seed,
    features_differ=None,
    differ_params=None,
    resample_perts=False,
):
    # Ensure the proportions sum up to 1
    # assert sum(feature_proportions.values()) == 1.0, "Feature proportions must sum up to 1"

    rng = np.random.default_rng(seed)

    # Calculate the number of features for each distribution type
    num_gaussian = int(n_feats * feature_proportions["gaussian"])
    num_lognormal = int(n_feats * feature_proportions["lognormal"])
    num_poisson = int(n_feats * feature_proportions["poisson"])
    num_nbinom = int(n_feats * feature_proportions["nbinom"])
    num_cauchy = int(n_feats * feature_proportions.get("cauchy", 0))

    # Generate unique plate maps
    unique_plate_maps = [
        generate_plate_map(n_wells, n_perts, n_controls, rng, resample_perts=resample_perts)
        for _ in range(n_plate_maps)
    ]

    plate_maps = rng.choice(unique_plate_maps, n_plates, replace=True)
    dframe = pd.DataFrame()
    all_feats = []

    for plate_idx in range(n_plates):
        plate_map_id = f"pm{unique_plate_maps.index(plate_maps[plate_idx])}"
        for well, pert in plate_maps[plate_idx].items():
            dframe = pd.concat([dframe, pd.DataFrame([
                {
                    "Metadata_Plate": f"p{plate_idx}",
                    "Metadata_Well": well,
                    "Metadata_Perturbation": pert,
                    "Metadata_Plate_Map": plate_map_id,
                }])],
                ignore_index=True,
            )

            if pert == "negative_control":
                feats = generate_features_for_perturbation(
                    num_gaussian, num_lognormal, num_poisson, num_nbinom, control_params, rng, num_cauchy=num_cauchy
                )
            else:
                feats = generate_features_for_perturbation(
                    num_gaussian,
                    num_lognormal,
                    num_poisson,
                    num_nbinom,
                    pert_params[pert],
                    rng,
                    num_cauchy=num_cauchy,
                    features_differ=features_differ,
                    # differ_params=differ_params(),
                    differ_params=differ_params,
                )

            all_feats.append(feats)

    feats = pd.DataFrame(all_feats)
    dframe["Metadata_Perturbation_Type"] = dframe["Metadata_Perturbation"].apply(
        lambda x: "negative_control" if x == "negative_control" else "treatment"
    )
    return dframe, feats


def aggregate_metrics_results(metrics_results, pert_col):
    percent_repl, merged_metrics = metrics_results["replicate_reproducibility"]
    for metric, results in metrics_results.items():
        if isinstance(results, pd.DataFrame) and pert_col in results.columns and metric != "replicate_reproducibility":
            merged_metrics = merged_metrics.merge(results.drop_duplicates(), on=pert_col)

    merged_metrics[["-log10(p_value)", "-log10(mp_value)"]] = -np.log10(
        merged_metrics[["p_value", "mp_value"]].replace(0, np.finfo(float).eps)
    )
    merged_metrics["similarity_metric"] = 1 - merged_metrics["similarity_metric"]
    merged_metrics["percent_replicating"] = percent_repl
    merged_metrics = merged_metrics.drop_duplicates().reset_index()
    return merged_metrics


def calculate_accuracy(metrics_results):
    mp_values = metrics_results["mp_value"].drop_duplicates()
    assert (
        mp_values.shape[0]
        == metrics_results["mean_ap"].shape[0]
        == metrics_results["replicate_reproducibility"][1].shape[0]
    ), "Number of perts should be the same"
    return {
        "replicate_reproducibility": metrics_results["replicate_reproducibility"][0],
        "mp_value": (metrics_results["mp_value"].drop_duplicates()["mp_value"] < 0.05).sum() / mp_values.shape[0],
        "mean_ap": (metrics_results["mean_ap"].drop_duplicates()["mean_ap"] == 1.0).sum() / mp_values.shape[0],
        "mean_ap_p_value": (metrics_results["mean_ap"].drop_duplicates()["p_value"] < 0.05).sum() / mp_values.shape[0],
    }
