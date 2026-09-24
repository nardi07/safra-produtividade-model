# Safra Produtividade Model — Previsão de Produtividade de Safra

Modelo de regressão para prever a produtividade agrícola (`produtividade_ton_ha`) de um talhão a partir de variáveis de clima, solo e manejo. O projeto cobre o pipeline completo: EDA, pré-processamento, seleção e validação cruzada de modelos, otimização de hiperparâmetros, diagnóstico de resíduos, interpretabilidade e tradução dos resultados técnicos em valor de negócio.

> **Base sintética.** Os dados usados neste projeto são sintéticos (cenário fictício), criados para fins de estudo/entrega acadêmica. O modelo precisa ser revalidado com dados reais antes de qualquer uso operacional.

## Conteúdo do repositório

| Arquivo | Descrição |
|---|---|
| `modelagem_produtividade_safra_v5 (1).ipynb` | Notebook principal e mais recente, com o pipeline completo de modelagem (EDA → relatório final), dividido em 11 fases. |
| `modelo_previsao_produtividade.ipynb` | Versão anterior/exploratória do notebook, com as etapas iniciais de carregamento e EDA. |
| `base_sintetica_previsao_produtividade_safra_v2.csv` | Dataset sintético utilizado no treinamento. |

Este repositório contém os notebooks e a base de dados. Os artefatos de produção citados no checklist do notebook (`models/modelo_produtividade_safra.joblib`, `relatorio_executivo.txt`, `predict.py`, `requirements.txt`) são gerados ao executar o notebook, mas ainda não foram exportados para este repositório.

## Dataset

Uma linha por talhão/safra, com as colunas:

`id_talhao`, `ano_safra`, `regiao`, `cultivar`, `tipo_solo`, `irrigacao`, `area_plantada_ha`, `chuva_mm`, `temperatura_media_c`, `horas_sol_dia`, `ph_solo`, `fertilizante_kg_ha`, `indice_pragas`, `densidade_plantio_mil_sementes_ha`, `produtividade_ton_ha` (target)

## Pipeline do notebook (`modelagem_produtividade_safra_v5 (1).ipynb`)

1. Carregamento e inspeção inicial
2. Análise exploratória — estrutura/nulos/duplicatas, target, variáveis numéricas e categóricas, padrão de nulos, outliers, multicolinearidade
3. Pré-processamento
4. Divisão treino/teste
5. Escalonamento e encoding (ajustados só no treino)
6. Seleção e treinamento de modelos base (Dummy, Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, SVR, KNN)
7. Validação cruzada (K-Fold)
8. Otimização de hiperparâmetros (RandomizedSearchCV)
9. Diagnóstico e testes de qualidade dos resíduos
10. Interpretabilidade (importância de features / SHAP)
11. Persistência e entregáveis

### Principais decisões de pré-processamento

| Item | Decisão | Por quê |
|---|---|---|
| `id_talhao` | Removido | identificador, sem valor preditivo |
| Duplicatas completas (15) | Removidas | erro de exportação/duplo lançamento |
| `regiao`, `tipo_solo`, `cultivar`, `irrigacao` | Normalizados (strip + title case) | inconsistência de fonte de dados |
| `chuva_mm`, `temperatura_media_c`, `ph_solo` (nulos ~4%) | Imputação pela mediana (fit no treino) | poucos nulos, padrão aleatório (MCAR) |
| `indice_pragas` (nulos ~19%) | Flag `indice_pragas_nao_medido` + imputação pela mediana | ausência carrega informação (MNAR ligado à área) |
| `fertilizante_kg_ha` (outliers ~10x) | Clip em 600 kg/ha | erro de digitação, não caso legítimo |

## Resultados do modelo

**Modelo final: Gradient Boosting**, selecionado após comparação com 9 outros modelos e otimizado via `RandomizedSearchCV`.

Métricas no conjunto de teste (20% dos dados, nunca visto no treino):

| Métrica | Valor |
|---|---|
| R² | 0.7841 |
| MAE | 0.4591 ton/ha |
| RMSE | 0.5701 ton/ha |
| MAPE | 6.42% |

Intervalo de confiança (bootstrap, 95%): R² em [0.7426, 0.8163] · MAE em [0.4263, 0.4927] ton/ha.

**Valor de negócio:** comparado a uma estimativa ingênua pela média histórica (MAE 0.98 ton/ha, MAPE 13.8%), o modelo reduz o erro médio de estimativa em **53%**.

Melhores hiperparâmetros encontrados: `n_estimators=500`, `max_depth=2`, `learning_rate=0.1`, `subsample=0.7`.

## Limitações e premissas

- Modelo treinado em dados sintéticos; validar com dados reais da cooperativa antes de uso operacional.
- Não validado para valores de features fora do range observado no treino (ex.: chuva muito acima de 1687mm).
- `ano_safra` foi tratado como covariável transversal, não como dimensão temporal — não usar este modelo para prever tendências futuras ano a ano.

## Próximos passos recomendados

- Coletar `indice_pragas` de forma mais consistente em talhões pequenos (hoje é o principal gerador de nulos ausentes não aleatórios).
- Reavaliar o modelo a cada nova safra para detectar drift entre clima/manejo e produtividade.
- Validar as leituras de importância de features/SHAP com um agrônomo antes de usar o modelo para recomendações de manejo.
- Testar outros modelos de boosting (XGBoost/LightGBM/CatBoost) conforme restrições de infraestrutura.
