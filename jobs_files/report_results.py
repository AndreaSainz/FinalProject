import os
import pandas as pd
import ast

# Ruta base
base_path = "/home/as3628/rds/hpc-work/final_project_dis/as3628/models/figures"
carpetas_permitidas = {'DBP', 'DeepFBP_phase3', 'DeepFusionBP', 'FusionDBP'}

# Recorre todos los subdirectorios permitidos
for root, _, files in os.walk(base_path):
    # Validar si estamos en una carpeta permitida o sus subcarpetas
    partes = os.path.relpath(root, base_path).split(os.sep)
    if partes[0] not in carpetas_permitidas:
        continue

    # Buscar pares de archivos MSE y SSIM
    mse_files = [f for f in files if f.endswith('_psnr_full.csv')]
    for mse_file in mse_files:
        # Buscar el correspondiente SSIM
        prefix = mse_file.replace('_psnr_full.csv', '')
        ssim_file = prefix + '_ssim_full.csv'

        if ssim_file in files:
            mse_path = os.path.join(root, mse_file)
            ssim_path = os.path.join(root, ssim_file)

            # Leer ambos archivos
            try:
                mse_df = pd.read_csv(mse_path)
                ssim_df = pd.read_csv(ssim_path)
            except Exception as e:
                print(f"Error leyendo archivos CSV: {e}")
                continue

            # Mostrar modelo
            print(f"\nModelo: {prefix}")

            # Procesar ambos
            for df, tipo in [(mse_df, 'PSNR'), (ssim_df, 'SSIM')]:
                for _, row in df.iterrows():
                    algoritmo = row[0]
                    try:
                        valores = ast.literal_eval(row[1])
                        if len(valores) > 69:
                            valor_69 = f"{valores[69]:.4f}" 
                        else:
                            valor_69 = 'N/A'
                        print(f"  {tipo} - {algoritmo}: {valor_69}")
                    except Exception as e:
                        print(f"  Error procesando {tipo} - {algoritmo}: {e}")