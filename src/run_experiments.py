import os
import numpy as np
import pandas as pd

from data_loader import load_opencellid_data
from simulator import generate_users_near_towers
from fitness_function import calculate_fitness, compute_normalization_bounds
from genetic_optimizer import run_ga, run_kbga


DATA_DIR = "../data"
OUTPUT_DIR = "../outputs"


def load_instance(filename):
    """Load tower data and filter 5G records."""
    path = os.path.join(DATA_DIR, filename)
    df = load_opencellid_data(path)
    df.columns = df.columns.str.strip().str.lower()
    return df[df["radio"].str.upper() == "NR"].copy()


def stepwise_tuning(df_towers, df_users, ga_func, instance_name):
    """Tune GA hyper-parameters in three sequential steps."""
    results = []
    norm_bounds = compute_normalization_bounds(df_towers, df_users)

    # --- population size ---
    pop_sizes = [20, 50, 100]
    best_pop = pop_sizes[0]
    best_avg = float("inf")
    for pop in pop_sizes:
        scores = [
            ga_func(
                df_towers,
                df_users,
                pop_size=pop,
                num_generations=30,
                mutation_rate=0.1,
                normalization_bounds=norm_bounds,
                verbose=False,
            )[1]
            for _ in range(10)
        ]
        avg = float(np.mean(scores))
        results.append(
            {
                "instance": instance_name,
                "step": "population",
                "pop_size": pop,
                "mutation_type": "flip",
                "crossover": "uniform",
                "avg_fitness": avg,
            }
        )
        if avg < best_avg:
            best_avg = avg
            best_pop = pop

    # --- mutation type ---
    mut_types = ["flip", "swap"]
    best_mut = mut_types[0]
    best_avg = float("inf")
    for m in mut_types:
        scores = [
            ga_func(
                df_towers,
                df_users,
                pop_size=best_pop,
                num_generations=30,
                mutation_rate=0.1,
                mutation_type=m,
                normalization_bounds=norm_bounds,
                verbose=False,
            )[1]
            for _ in range(10)
        ]
        avg = float(np.mean(scores))
        results.append(
            {
                "instance": instance_name,
                "step": "mutation",
                "pop_size": best_pop,
                "mutation_type": m,
                "crossover": "uniform",
                "avg_fitness": avg,
            }
        )
        if avg < best_avg:
            best_avg = avg
            best_mut = m

    # --- crossover method ---
    cross_methods = ["uniform", "one_point"]
    best_cross = cross_methods[0]
    best_avg = float("inf")
    for c in cross_methods:
        scores = [
            ga_func(
                df_towers,
                df_users,
                pop_size=best_pop,
                num_generations=30,
                mutation_rate=0.1,
                mutation_type=best_mut,
                crossover_method=c,
                normalization_bounds=norm_bounds,
                verbose=False,
            )[1]
            for _ in range(10)
        ]
        avg = float(np.mean(scores))
        results.append(
            {
                "instance": instance_name,
                "step": "crossover",
                "pop_size": best_pop,
                "mutation_type": best_mut,
                "crossover": c,
                "avg_fitness": avg,
            }
        )
        if avg < best_avg:
            best_avg = avg
            best_cross = c

    best_params = {
        "pop_size": best_pop,
        "num_generations": 30,
        "mutation_rate": 0.1,
        "num_parents": max(2, best_pop // 3),
        "mutation_type": best_mut,
        "crossover_method": best_cross,
        "normalization_bounds": norm_bounds,
    }
    return best_params, pd.DataFrame(results)


def evaluate_instance(df_towers, df_users, ga_params, kbga_params, instance_name):
    norm_bounds = ga_params["normalization_bounds"]
    baseline = calculate_fitness(
        df_towers, df_users, normalization_bounds=norm_bounds, verbose=False
    )["fitness"]

    ga_scores = [
        run_ga(
            df_towers,
            df_users,
            pop_size=ga_params["pop_size"],
            num_generations=ga_params["num_generations"],
            mutation_rate=ga_params["mutation_rate"],
            num_parents=ga_params["num_parents"],
            mutation_type=ga_params["mutation_type"],
            crossover_method=ga_params["crossover_method"],
            normalization_bounds=norm_bounds,
            verbose=False,
        )[1]
        for _ in range(10)
    ]

    kbga_scores = [
        run_kbga(
            df_towers,
            df_users,
            pop_size=kbga_params["pop_size"],
            num_generations=kbga_params["num_generations"],
            mutation_rate=kbga_params["mutation_rate"],
            num_parents=kbga_params["num_parents"],
            mutation_type=kbga_params["mutation_type"],
            crossover_method=kbga_params["crossover_method"],
            normalization_bounds=norm_bounds,
            verbose=False,
        )[1]
        for _ in range(10)
    ]

    return {
        "instance": instance_name,
        "baseline": float(baseline),
        "ga_best": float(np.min(ga_scores)),
        "ga_worst": float(np.max(ga_scores)),
        "kbga_best": float(np.min(kbga_scores)),
        "kbga_worst": float(np.max(kbga_scores)),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tune_instances = ["germany.csv.gz", "france.csv.gz"]
    ga_records = []
    kbga_records = []
    ga_params_map = {}
    kbga_params_map = {}

    for fname in tune_instances:
        towers = load_instance(fname)
        users = generate_users_near_towers(towers, count=1000)
        ga_params, ga_df = stepwise_tuning(towers, users, run_ga, fname)
        kbga_params, kbga_df = stepwise_tuning(towers, users, run_kbga, fname)
        ga_records.append(ga_df)
        kbga_records.append(kbga_df)
        ga_params_map[fname] = ga_params
        kbga_params_map[fname] = kbga_params

    pd.concat(ga_records).to_csv(os.path.join(OUTPUT_DIR, "tuning_ga.csv"), index=False)
    pd.concat(kbga_records).to_csv(os.path.join(OUTPUT_DIR, "tuning_kbga.csv"), index=False)

    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv.gz")]
    eval_records = []
    for fname in all_files:
        towers = load_instance(fname)
        users = generate_users_near_towers(towers, count=1000)
        ga_params = ga_params_map.get(fname, list(ga_params_map.values())[0])
        kbga_params = kbga_params_map.get(fname, list(kbga_params_map.values())[0])
        eval_records.append(
            evaluate_instance(towers, users, ga_params, kbga_params, fname)
        )

    eval_df = pd.DataFrame(eval_records)
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_summary.csv"), index=False)

    overall = {
        "ga_best_of_best": eval_df["ga_best"].min(),
        "ga_best_worst": eval_df["ga_worst"].min(),
        "kbga_best_of_best": eval_df["kbga_best"].min(),
        "kbga_best_worst": eval_df["kbga_worst"].min(),
    }
    pd.DataFrame([overall]).to_csv(
        os.path.join(OUTPUT_DIR, "evaluation_overall.csv"), index=False
    )


if __name__ == "__main__":
    main()
