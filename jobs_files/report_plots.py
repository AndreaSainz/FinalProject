import os
import pandas as pd
import ast
from pathlib import Path
from ct_reconstruction.utils.plotting import plot_results_distributions
import numpy as np
from collections import defaultdict
import re

def load_metric_file(filepath):
    df = pd.read_csv(filepath)
    df["Values"] = df.iloc[:, 1].apply(ast.literal_eval)
    df = df.drop(columns=df.columns[1])
    return df

def collect_metrics_from_folders(folders):
    metrics_data = {"SSIM": [], "PSNR": [], "MSE": []}
    reference_data = {}
    reference_algorithms = {"EM", "FBP", "NAG_LS", "SIRT", "TV_MIN"}

    for folder in folders:
        folder = Path(folder)
        for file in folder.glob("*.csv"):
            metric_type = None
            lower = file.name.lower()
            if "ssim" in lower:
                metric_type = "SSIM"
            elif "psnr" in lower:
                metric_type = "PSNR"
            elif "mse" in lower:
                metric_type = "MSE"
            else:
                continue

            df = load_metric_file(file)

            for _, row in df.iterrows():
                algorithm = row["Algorithm"]
                values = row["Values"]

                if algorithm in reference_algorithms:
                    if metric_type not in reference_data:
                        reference_data[metric_type] = {}
                    if algorithm not in reference_data[metric_type]:
                        reference_data[metric_type][algorithm] = values
                    elif not np.allclose(reference_data[metric_type][algorithm], values, rtol=1e-2, atol=1e-3):
                        expected = np.array(reference_data[metric_type][algorithm])
                        actual = np.array(values)
                        differences = np.where(np.abs(expected - actual) > (np.abs(expected) * 1e-5 + 1e-6))[0]
                        diff_details = "\n".join(
                            f"  Índice {i}: esperado={expected[i]}, actual={actual[i]}"
                            for i in differences[:10]
                        )
                        raise ValueError(
                            f"Inconsistencia en los valores de {algorithm} para {metric_type} en el archivo: {file.name}\n"
                            f"Total diferencias: {len(differences)} posiciones.\nEjemplos:\n{diff_details}"
                        )

                for v in values:
                    metrics_data[metric_type].append({
                        "Algorithm": algorithm,
                        metric_type: v
                    })

    return (
        pd.DataFrame(metrics_data["PSNR"]),
        pd.DataFrame(metrics_data["SSIM"]),
        pd.DataFrame(metrics_data["MSE"])
    )



traditional_algorithms = ["EM", "FBP", "NAG_LS", "SIRT", "TV_MIN"]

def base_name(alg):
    match = re.match(r"([A-Z_]+?)(?:_[IVXLCDM]+)?$", alg)
    return match.group(1) if match else alg

def get_algorithm_order(df_list):
    all_algorithms = set()
    for df in df_list:
        all_algorithms.update(df["Algorithm"].unique())

    new_algo_groups = defaultdict(list)
    for alg in all_algorithms:
        if alg not in traditional_algorithms:
            new_algo_groups[base_name(alg)].append(alg)

    # Ordenar internamente los grupos por nombre base, luego por sufijo
    new_algos_ordered = []
    for base in sorted(new_algo_groups):
        group = sorted(new_algo_groups[base])
        new_algos_ordered.extend(group)

    return traditional_algorithms + new_algos_ordered

def apply_algorithm_order(df, metric, order):
    df["Algorithm"] = pd.Categorical(df["Algorithm"], categories=order, ordered=True)
    return df.sort_values("Algorithm")



folders = [
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/DeepFBP_phase3/Low_dose/FilterI",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/DeepFBP_phase3/Low_dose/FilterII",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/FusionDBP/Low_dose/FilterI",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/FusionDBP/Low_dose/FilterII"
]

df_psnr, df_ssim, df_mse = collect_metrics_from_folders(folders)

# Obtener orden personalizado y aplicar
algorithm_order = get_algorithm_order([df_psnr, df_ssim, df_mse])
df_psnr = apply_algorithm_order(df_psnr, "PSNR", algorithm_order)
df_ssim = apply_algorithm_order(df_ssim, "SSIM", algorithm_order)
df_mse = apply_algorithm_order(df_mse, "MSE", algorithm_order)

# Dibujar gráficas
plot_results_distributions(
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures/report_metrics_lowdose",
    df_psnr, df_ssim, df_mse
)

# Imprimir estadísticas
def print_summary_stats(df, metric):
    summary = df.groupby("Algorithm")[metric].agg(['mean', 'std'])\
                 .sort_values("mean", ascending=False)
    print(f"\nResumen para {metric}:")
    print(summary.to_string(float_format="%.4f"))

print_summary_stats(df_psnr, "PSNR")
print_summary_stats(df_ssim, "SSIM")
print_summary_stats(df_mse, "MSE")