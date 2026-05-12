# ACOL — Priorização Inteligente de Solicitações para Revisão Humana

Projeto de Machine Learning para classificar solicitações de compras com um **score de prioridade**, permitindo que analistas foquem nos casos de maior risco. Combina features tabulares, uma camada NLP baseada em documentos de política (TF-IDF) e features de fornecedor com *expanding window* para evitar leakage temporal.

---

## Estrutura do Projeto

```
acol_case/
├── data/                        # Dados brutos (não versionados)
│   ├── solicitacoes.csv
│   ├── base_historica_fornecedores.csv
│   └── documentos_politicas.json
├── notebooks/
│   ├── 01_eda.ipynb             # Análise exploratória completa
│   └── 02_modeling.ipynb        # Feature engineering + modelo + avaliação
├── reports/
│   ├── 01_eda_findings.md       # Achados da EDA com justificativas
│   ├── 02_solution_design.md    # Design da solução + arquitetura GCP
│   └── figures/                 # Plots gerados pelos notebooks
├── pyproject.toml               # Dependências do projeto (uv)
├── uv.lock                      # Lock file
└── Dockerfile / docker-compose.yml
```

---

## Pré-requisitos

- [uv](https://docs.astral.sh/uv/) >= 0.5
- Docker + Docker Compose (opcional, recomendado)
- Python 3.11+

---

## Setup

### Opção 1 — Docker (recomendado)

Sobe o JupyterLab com todas as dependências isoladas:

```bash
docker compose up --build
```

Acesse em `http://localhost:8888` (sem token).

Para parar:

```bash
docker compose down
```

### Opção 2 — Local com uv

```bash
# Instalar dependências
uv sync

# Ativar o ambiente
source .venv/bin/activate

# Iniciar o JupyterLab
jupyter lab --notebook-dir=.
```

---

## Dados

Os arquivos de dados não são versionados (`data/*` está no `.gitignore`). Coloque os arquivos abaixo em `data/` antes de executar os notebooks:

| Arquivo | Descrição |
|---|---|
| `solicitacoes.csv` | Dataset principal — 15.000 linhas, 21 colunas, alvo: `revisao_prioritaria_confirmada` |
| `base_historica_fornecedores.csv` | Histórico agregado por fornecedor (210 fornecedores) |
| `documentos_politicas.json` | 7 políticas documentais usadas pela camada NLP |

---

## Executando os Notebooks

Execute na ordem:

### 1. EDA — `notebooks/01_eda.ipynb`

Análise exploratória em 9 blocos:
- Perfil do dataset, missingness, duplicatas
- Análise da variável-alvo e desbalanceamento (7,6:1)
- Univariadas numéricas, bivariadas e cruzamentos
- Análise temporal e validação de TimeSeriesSplit
- Join com histórico de fornecedores e detecção de leakage
- Triagem de leakage com AUC isolado por feature
- Análise textual de `descricao_item`

Resultados em `reports/01_eda_findings.md`.

### 2. Modelagem — `notebooks/02_modeling.ipynb`

Pipeline completo:
- Remoção de duplicatas e features com leakage crítico
- Feature engineering: expanding window, camada NLP (TF-IDF cosine similarity com políticas + rule flags), log1p, OHE
- Feature selection: filtro de correlação + LightGBM gain (top 95%)
- Treino com TimeSeriesSplit (5 folds): Logistic Regression (baseline) vs LightGBM
- Avaliação: AUC-ROC, Average Precision, F1 no threshold ótimo via curva PR
- Logging de métricas e parâmetros (MLflow ou fallback JSON local)

Resultados em `reports/02_model_summary.json`.

---

## Relatórios

| Arquivo | Conteúdo |
|---|---|
| `reports/01_eda_findings.md` | Achados da EDA com método, justificativa e resultado por bloco |
| `reports/02_solution_design.md` | Arquitetura da solução, NLP layer, feature selection, métricas, deploy GCP + MLflow, loop de melhoria contínua |

Os PDFs (`reports/*.pdf`) são gerados localmente e não são versionados.

---

## Dependências Principais

| Pacote | Uso |
|---|---|
| `pandas >= 3.0` | Manipulação de dados |
| `scikit-learn >= 1.8` | Preprocessing, métricas, modelos baseline |
| `lightgbm >= 4.6` | Modelo principal |
| `seaborn >= 0.13` | Visualizações |
| `jupyterlab >= 4.0` | Ambiente de notebooks |
| `fpdf2` + `pillow` | Geração de PDFs (dev) |

Para regenerar o `uv.lock`:

```bash
uv lock
```

Para adicionar uma dependência:

```bash
uv add <pacote>
```
