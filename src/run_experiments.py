import os
import numpy as np
import pandas as pd
import concurrent.futures
from data_loader import load_opencellid_data
from simulator import generate_users_near_towers
from fitness_function import calculate_fitness, compute_normalization_bounds
from genetic_optimizer import run_ga, run_kbga
from visualizer import plot_towers_on_map


# === Paths ===
DATA_DIR = "../data"
OUTPUT_DIR = "../outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
CLEAN_DIR = os.path.join(OUTPUT_DIR, "clean_data")


def _ga_replicates(ga_func, df_towers, df_users, repeats=10, **kwargs):
    """Run GA function multiple times and return average fitness."""
    scores = []
    for _ in range(repeats):
        _, fit = ga_func(
            df_towers,
            df_users,
            **kwargs,
            verbose=False,
        )
        scores.append(fit)
    return float(np.mean(scores))


def load_instance(filename):
    """Load OpenCellID data and filter only 5G (NR) records."""
    path = os.path.join(DATA_DIR, filename)
    df = load_opencellid_data(path)
    df.columns = df.columns.str.strip().str.lower()
    return df[df["radio"].str.upper() == "NR"].copy()


def stepwise_tuning(df_towers, df_users, ga_func, instance_name):
    """Sequentially tune GA hyperparameters: population size, mutation type, crossover method."""
    results = []
    norm_bounds = compute_normalization_bounds(df_towers, df_users)

    print(f"\n\U0001F3C1 Tuning instance: {instance_name}")

    # Step 1: Population Size
    pop_sizes = [20, 50, 100]
    best_pop = pop_sizes[0]
    best_avg = float("inf")
    print("  • Step 1/3: population size")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(pop_sizes)) as ex:
        futures = {
            ex.submit(
                _ga_replicates,
                ga_func,
                df_towers,
                df_users,
                repeats=1,
                pop_size=pop,
                num_generations=30,
                mutation_rate=0.1,
                normalization_bounds=norm_bounds,
            ): pop
            for pop in pop_sizes
        }
        for fut in concurrent.futures.as_completed(futures):
            pop = futures[fut]
            avg = fut.result()
            print(f"    - pop {pop}: {avg:.4f}")
            results.append({
                "instance": instance_name,
                "step": "population",
                "pop_size": pop,
                "mutation_type": "flip",
                "crossover": "uniform",
                "avg_fitness": avg,
            })
            if avg < best_avg:
                best_avg = avg
                best_pop = pop

    # Step 2: Mutation Type
    mut_types = ["flip", "swap"]
    best_mut = mut_types[0]
    best_avg = float("inf")
    print("  • Step 2/3: mutation type")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(mut_types)) as ex:
        futures = {
            ex.submit(
                _ga_replicates,
                ga_func,
                df_towers,
                df_users,
                pop_size=best_pop,
                num_generations=30,
                mutation_rate=0.1,
                mutation_type=m,
                normalization_bounds=norm_bounds,
            ): m
            for m in mut_types
        }
        for fut in concurrent.futures.as_completed(futures):
            m = futures[fut]
            avg = fut.result()
            print(f"    - mutation {m}: {avg:.4f}")
            results.append({
                "instance": instance_name,
                "step": "mutation",
                "pop_size": best_pop,
                "mutation_type": m,
                "crossover": "uniform",
                "avg_fitness": avg,
            })
            if avg < best_avg:
                best_avg = avg
                best_mut = m

    # Step 3: Crossover Method
    cross_methods = ["uniform", "one_point"]
    best_cross = cross_methods[0]
    best_avg = float("inf")
    print("  • Step 3/3: crossover method")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(cross_methods)) as ex:
        futures = {
            ex.submit(
                _ga_replicates,
                ga_func,
                df_towers,
                df_users,
                pop_size=best_pop,
                num_generations=30,
                mutation_rate=0.1,
                mutation_type=best_mut,
                crossover_method=c,
                normalization_bounds=norm_bounds,
            ): c
            for c in cross_methods
        }
        for fut in concurrent.futures.as_completed(futures):
            c = futures[fut]
            avg = fut.result()
            print(f"    - crossover {c}: {avg:.4f}")
            results.append({
                "instance": instance_name,
                "step": "crossover",
                "pop_size": best_pop,
                "mutation_type": best_mut,
                "crossover": c,
                "avg_fitness": avg,
            })
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


def evaluate_instance(df_towers, df_users, ga_params, kbga_params, instance_name, save_comparison=False):
    norm_bounds = ga_params["normalization_bounds"]
    """Evaluate a single dataset with GA and KBGA and record detailed stats."""

    # === Baseline using all towers (normalized and raw) ===
    baseline_norm = calculate_fitness(
        df_towers,
        df_users,
        normalization_bounds=norm_bounds,
        verbose=False,
    )["fitness"]
    baseline_stats = calculate_fitness(df_towers, df_users, verbose=False)

    # === Run GA multiple times and keep best/worst ===
    ga_fitnesses = []
    ga_solutions = []
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(4, os.cpu_count())
    ) as ex:
        futures = [
            ex.submit(
                run_ga,
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
            )
            for _ in range(10)
        ]
        for fut in concurrent.futures.as_completed(futures):
            sol, fit = fut.result()
            ga_solutions.append(sol)
            ga_fitnesses.append(fit)

    best_ga_idx = int(np.argmin(ga_fitnesses))
    ga_fitness = float(min(ga_fitnesses))
    ga_worst = float(max(ga_fitnesses))
    ga_solution = ga_solutions[best_ga_idx]

    df_ga = df_towers.copy()
    df_ga["active"] = ga_solution
    df_ga_active = df_ga[df_ga["active"] == 1].copy()
    ga_stats = calculate_fitness(df_ga_active, df_users, verbose=False)

    # === Run KBGA multiple times and keep best/worst ===
    kbga_fitnesses = []
    kbga_solutions = []
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(4, os.cpu_count())
    ) as ex:
        futures = [
            ex.submit(
                run_kbga,
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
            )
            for _ in range(10)
        ]
        for fut in concurrent.futures.as_completed(futures):
            sol, fit = fut.result()
            kbga_solutions.append(sol)
            kbga_fitnesses.append(fit)

    best_kbga_idx = int(np.argmin(kbga_fitnesses))
    kbga_fitness = float(min(kbga_fitnesses))
    kbga_worst = float(max(kbga_fitnesses))
    kbga_solution = kbga_solutions[best_kbga_idx]

    df_kbga = df_towers.copy()
    df_kbga["active"] = kbga_solution
    df_kbga_active = df_kbga[df_kbga["active"] == 1].copy()
    kbga_stats = calculate_fitness(df_kbga_active, df_users, verbose=False)

    record = {
        "instance": instance_name,
        "baseline": float(baseline_norm),
        "ga_best": float(ga_fitness),
        "ga_worst": float(ga_worst),
        "kbga_best": float(kbga_fitness),
        "kbga_worst": float(kbga_worst),
        # Raw metrics (no normalization)
        "baseline_active_towers": baseline_stats["active_towers"],
        "baseline_unserved_demand": baseline_stats["unserved_demand"],
        "baseline_overload": baseline_stats["overload"],
        "baseline_excessive_distance": baseline_stats["excessive_distance"],
        "baseline_imbalance": baseline_stats["imbalance"],
        "ga_active_towers": ga_stats["active_towers"],
        "ga_unserved_demand": ga_stats["unserved_demand"],
        "ga_overload": ga_stats["overload"],
        "ga_excessive_distance": ga_stats["excessive_distance"],
        "ga_imbalance": ga_stats["imbalance"],
        "kbga_active_towers": kbga_stats["active_towers"],
        "kbga_unserved_demand": kbga_stats["unserved_demand"],
        "kbga_overload": kbga_stats["overload"],
        "kbga_excessive_distance": kbga_stats["excessive_distance"],
        "kbga_imbalance": kbga_stats["imbalance"],
        # Algorithm settings
        "ga_pop_size": ga_params["pop_size"],
        "ga_num_generations": ga_params["num_generations"],
        "ga_mutation_rate": ga_params["mutation_rate"],
        "ga_num_parents": ga_params["num_parents"],
        "ga_mutation_type": ga_params["mutation_type"],
        "ga_crossover_method": ga_params["crossover_method"],
        "kbga_pop_size": kbga_params["pop_size"],
        "kbga_num_generations": kbga_params["num_generations"],
        "kbga_mutation_rate": kbga_params["mutation_rate"],
        "kbga_num_parents": kbga_params["num_parents"],
        "kbga_mutation_type": kbga_params["mutation_type"],
        "kbga_crossover_method": kbga_params["crossover_method"],
    }

    if save_comparison:
        base = os.path.splitext(os.path.splitext(instance_name)[0])[0]
        cmp_path = os.path.join(OUTPUT_DIR, f"{base}_comparison.csv")
        pd.DataFrame([
            {"Method": "GA", "Fitness": record["ga_best"]},
            {"Method": "KBGA", "Fitness": record["kbga_best"]},
        ]).to_csv(cmp_path, index=False)

    return record

def evaluate_file(fname, ga_params, kbga_params):
    """Wrapper to evaluate a single file. Intended for parallel use."""
    towers = load_instance(fname)
    users = generate_users_near_towers(towers, count=100000)
    print(f"\n🔍 Evaluating {fname}")
    print(f"📊 File: {fname} | Towers: {towers.shape[0]} | Users: {users.shape[0]}")

    base = os.path.splitext(os.path.splitext(fname)[0])[0]
    towers.to_csv(os.path.join(CLEAN_DIR, f"{base}_5g_towers.csv"), index=False)
    users.to_csv(os.path.join(CLEAN_DIR, f"{base}_users.csv"), index=False)

    center = [
        (towers['lat'].min() + towers['lat'].max()) / 2,
        (towers['lon'].min() + towers['lon'].max()) / 2,
    ]
    plot_towers_on_map(
        towers,
        map_center=center,
        df_users=users,
        save_path=os.path.join(FIG_DIR, f"{base}_5g_map.html")
    )

    record = evaluate_instance(towers, users, ga_params, kbga_params, fname, save_comparison=True)
    print(f"    ✅ baseline={record['baseline']:.4f}, GA best={record['ga_best']:.4f}, KBGA best={record['kbga_best']:.4f}")
    return record


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)

    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv.gz")]
    tune_targets = {"Austria.csv.gz", "germany.csv.gz"}

    ga_records, kbga_records = [], []
    ga_params_map, kbga_params_map = {}, {}

    # === First: Only Tune Austria and Germany ===
    for fname in all_files:
        if fname not in tune_targets:
            continue  # Skip non-targets for tuning

        towers = load_instance(fname)
        users = generate_users_near_towers(towers, count=10000)
        print(f"🔧 Tuning File: {fname} | Towers: {towers.shape[0]} | Users: {users.shape[0]}")

        base = os.path.splitext(os.path.splitext(fname)[0])[0]
        towers.to_csv(os.path.join(CLEAN_DIR, f"{base}_5g_towers.csv"), index=False)
        users.to_csv(os.path.join(CLEAN_DIR, f"{base}_users.csv"), index=False)

        center = [
            (towers['lat'].min() + towers['lat'].max()) / 2,
            (towers['lon'].min() + towers['lon'].max()) / 2,
        ]
        plot_towers_on_map(
            towers, map_center=center, df_users=users,
            save_path=os.path.join(FIG_DIR, f"{base}_5g_map.html")
        )

        # Perform tuning
        ga_params, ga_df = stepwise_tuning(towers, users, run_ga, fname)
        kbga_params, kbga_df = stepwise_tuning(towers, users, run_kbga, fname)

        ga_records.append(ga_df)
        kbga_records.append(kbga_df)
        ga_params_map[fname] = ga_params
        kbga_params_map[fname] = kbga_params
        print(f"✅ Finished tuning {fname}")

    # Save tuning results
    if ga_records:
        pd.concat(ga_records).to_csv(os.path.join(OUTPUT_DIR, "tuning_ga.csv"), index=False)
    if kbga_records:
        pd.concat(kbga_records).to_csv(os.path.join(OUTPUT_DIR, "tuning_kbga.csv"), index=False)

    # === Evaluation for All Files (Using Tuned or Default Austria Params) ===
    default_ga_params = ga_params_map.get("Austria.csv.gz")
    default_kbga_params = kbga_params_map.get("Austria.csv.gz")

    eval_records = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = []
        for fname in all_files:
            ga_params = ga_params_map.get(fname, default_ga_params)
            kbga_params = kbga_params_map.get(fname, default_kbga_params)
            tasks.append(
                executor.submit(
                    evaluate_file,
                    fname,
                    ga_params,
                    kbga_params,
                )
            )

        for fut in concurrent.futures.as_completed(tasks):
            eval_records.append(fut.result())
    eval_df = pd.DataFrame(eval_records)
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_summary.csv"), index=False)

    overall = {
        "ga_best_of_best": eval_df["ga_best"].min(),
        "ga_best_worst": eval_df["ga_worst"].max(),
        "kbga_best_of_best": eval_df["kbga_best"].min(),
        "kbga_best_worst": eval_df["kbga_worst"].max(),
    }
    pd.DataFrame([overall]).to_csv(
        os.path.join(OUTPUT_DIR, "evaluation_overall.csv"), index=False
    )


if __name__ == "__main__":
    main()
