"""
Script de inferência para o modelo de previsão de produtividade de safra.

Uso:
    python predict.py --input novos_talhoes.csv --output previsoes.csv

Espera um CSV com as mesmas colunas brutas da base de treino (exceto o
target `produtividade_ton_ha`, que obviamente não existe para dados novos).
Aplica a mesma limpeza estrutural e engenharia de features feita no notebook
(Fase 3) e depois o pipeline de pré-processamento + modelo salvos em `models/`.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"


def limpar_dados_novos(df: pd.DataFrame) -> pd.DataFrame:
    """Replica a limpeza estrutural e a engenharia de features da Fase 3 do
    notebook (sem vazamento: nenhuma dessas correções depende de estatísticas
    do treino)."""
    df = df.copy()

    # Normaliza categorias (mesmas correções aplicadas no treino)
    for col in ["regiao", "tipo_solo", "cultivar", "irrigacao"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Remove coluna de identificação, se presente
    if "id_talhao" in df.columns:
        df = df.drop(columns=["id_talhao"])

    # Corrige o mesmo erro de digitação de fertilizante (cap físico plausível)
    if "fertilizante_kg_ha" in df.columns:
        df["fertilizante_kg_ha"] = df["fertilizante_kg_ha"].clip(upper=600)

    # Flag de "índice de pragas não medido", igual ao treino
    if "indice_pragas" in df.columns:
        df["indice_pragas_nao_medido"] = df["indice_pragas"].isna().astype(int)

    # Engenharia de features (igual à Fase 3 do notebook): razão
    # fertilizante/área e termo quadrático de chuva.
    if "fertilizante_kg_ha" in df.columns and "area_plantada_ha" in df.columns:
        df["fertilizante_por_ha"] = df["fertilizante_kg_ha"] / df["area_plantada_ha"].clip(lower=0.1)
    if "chuva_mm" in df.columns:
        df["chuva_mm_sq"] = df["chuva_mm"] ** 2

    return df


def carregar_bundle():
    bundle_path = MODELS_DIR / "modelo_produtividade_safra.joblib"
    if not bundle_path.exists():
        sys.exit(
            f"Não encontrei {bundle_path}. Rode o notebook até a Fase 11 "
            "(Persistência) primeiro para gerar o modelo treinado."
        )
    return joblib.load(bundle_path)


def main():
    parser = argparse.ArgumentParser(description="Prevê produtividade (ton/ha) para novos talhões.")
    parser.add_argument("--input", required=True, help="CSV com os dados novos (mesmas colunas do treino, sem o target).")
    parser.add_argument("--output", default="previsoes.csv", help="Caminho do CSV de saída com as previsões.")
    args = parser.parse_args()

    bundle = carregar_bundle()
    pipeline = bundle["pipeline"]        # Pipeline sklearn completo (pré-processamento + modelo)
    features = bundle["features"]        # Ordem/lista de features usada no treino

    df_novo = pd.read_csv(args.input)
    df_novo = limpar_dados_novos(df_novo)

    faltando = [c for c in features if c not in df_novo.columns]
    if faltando:
        sys.exit(f"Faltam colunas esperadas pelo modelo: {faltando}")

    X_novo = df_novo[features]
    previsoes = pipeline.predict(X_novo)

    df_saida = df_novo.copy()
    df_saida["produtividade_ton_ha_prevista"] = np.round(previsoes, 3)
    df_saida.to_csv(args.output, index=False)
    print(f"Previsões salvas em {args.output} ({len(df_saida)} linhas).")


if __name__ == "__main__":
    main()
