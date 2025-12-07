import pandas as pd
import numpy as np

def load_clean_transform(file_route, year_file):
    
    df = pd.read_csv(file_route, sep=";", encoding="latin1")

    # Sin respuesta (no sabe, no contesta, no ha lugar)
    missing_codes = [-97, -98, -99, -5,
                     "-97", "-98", "-99", "-5",
                     " -97", " -98", " -99", " -5"]
    
    df = df.replace(missing_codes, np.nan)

    # --- Renombrar columnas a utilizar ---
    rename_dict = {
        'a01': 'year',
        # 'a04': 'age',  # ELIMINADO: 'age' ya existe en el CSV
        'a1100': 'educ',
        'a1500': 'labor',
        'a2000': 'contract_type'
    }

    df.rename(columns=rename_dict, inplace=True)
    
    # Asegurarse de que age sea numérica
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # year
    years_replace = {2022: 2021, 2017: 2016}
    df['year'] = df['year'].replace(years_replace)

    # --- Demográficas ---
    
    df['age_bracket'] = pd.cut(
        df['age'],
        bins=[17, 34, 44, 54, 64, 79],
        labels=['18-34', '35-44', '45-54', '55-64', '65-79'],
        right=True
    )

    edu_map = {
        1: 'Primary', 2: 'Primary', 3: 'Primary',
        4: 'Secondary', 5: 'Secondary',
        6: 'University', 7: 'University', 8: 'University', 9: 'University'
    }
    df['edu_level'] = df['educ'].map(edu_map)
    
    # --- Categoría laboral ---
    if 'labor' in df.columns:
        df['labor_categ'] = None
        
        # Autónomo (1, 11, 12)
        df.loc[df['labor'].isin([1, 11, 12]), 'labor_categ'] = 'Autónomo'

        # Asalariado: labor == 2, 21, 22
        df.loc[df['labor'].isin([2, 21, 22]), 'labor_categ'] = 'Asalariado'

        # Desempleado
        df.loc[df['labor'] == 5, 'labor_categ'] = 'Desempleado'

        # Jubilado
        df.loc[df['labor'] == 6, 'labor_categ'] = 'Jubilado'

        # Otros inactivos (3, 4, 7, 8, 9, 10)
        df.loc[df['labor'].isin([3, 4, 7, 8, 9, 10]), 'labor_categ'] = 'Otros inactivos'
    else:
        df['labor_categ'] = None
    
    # --- Financieras ---
    rename_dict_finance = {
        'b0401': 'hipot',
        'b0402': 'plan_pensiones',
        'b0403': 'fondo_inversion',
        'b0404': 'acciones',
        'b0405': 'renta_fija',
        'b0406': 'prest_pers',
        'b0407': 'tarj_cred',
        'b0408': 'cta_ahorro/depos_pzo',
        'b0409': 'seg_vida',
        'b0410': 'seg_med',
        'b0312': 'cripto'
    }
    df.rename(columns=rename_dict_finance, inplace=True)

    for col in rename_dict_finance.values():
        if col not in df.columns:
            df[col] = np.nan

    # ID
    df['ID'] = [f"{y}_{i+1}" for i, y in enumerate(df['year'])]

    # Group type (savings y debt)
    df['savings'] = np.where(
        (df['plan_pensiones'] == 1) |
        (df['fondo_inversion'] == 1) |
        (df['acciones'] == 1) |
        (df['renta_fija'] == 1) |
        (df['cta_ahorro/depos_pzo'] == 1),
        1, 0
    )

    df['debt'] = np.where(
        (df['hipot'] == 1) | (df['prest_pers'] == 1),
        1, 0
    )

    # --- Final cols ---
    cols_finance = list(rename_dict_finance.values())
    cols_demog = ['ID', 'year', 'age_bracket', 'edu_level', 'labor_categ']
    cols_final = cols_finance + cols_demog

    df_final = df[cols_final].copy()
    
    # Rellenar NaN con 0 en columnas financieras (binarias: 1=tiene, 0=no tiene)
    finance_cols_present = [col for col in cols_finance if col in df_final.columns]
    df_final[finance_cols_present] = df_final[finance_cols_present].fillna(0)
    
    # Convertir columnas financieras a int (son binarias: 0 o 1)
    df_final[finance_cols_present] = df_final[finance_cols_present].astype(int)
    
    df_final = df_final.dropna(subset=['age_bracket', 'edu_level', 'labor_categ'])

    
    return df_final


if __name__ == "__main__":
    df_2016 = load_clean_transform("data/ecf_2016.csv", 2016)
    df_2021 = load_clean_transform("data/ecf_2021.csv", 2021)

    df_all = pd.concat([df_2016, df_2021], ignore_index=True)
    
    print(df_all.columns)
    print(df_all.head())
    print(f"\nNaN por columna:\n{df_all.isna().sum()}")