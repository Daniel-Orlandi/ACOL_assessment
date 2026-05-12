# Solution Design — ACOL Prioritization
**Caso:** Priorização Inteligente de Solicitações para Revisão Humana
**Notebook:** `notebooks/02_modeling.ipynb`

---

## 1. Visão Geral da Solução

O objetivo é classificar cada solicitação de compra com um **score de prioridade** entre 0 e 1, indicando a probabilidade de ser `revisao_prioritaria_confirmada=1`. Esse score é usado para **ranquear** a fila de revisão humana — analistas veem primeiro os casos de maior risco.

A solução combina três camadas de sinal:
1. **Features tabulares** (categóricas, numéricas, temporais)
2. **NLP policy layer** — extração de sinal dos documentos de política via TF-IDF + regras
3. **Features de fornecedor com expanding window** — histórico sem leakage temporal

---

## 2. Arquitetura Geral

```
╔══════════════════════════════════════════════════════════════════════╗
║               ACOL — SOLUTION ARCHITECTURE                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  DATA SOURCES                                                        ║
║  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────────┐ ║
║  │ solicitacoes   │  │  fornecedores   │  │ documentos_politicas │ ║
║  │    .csv        │  │     .csv        │  │      .json           │ ║
║  └───────┬────────┘  └────────┬────────┘  └──────────┬───────────┘ ║
║          │                   │                       │             ║
║          ▼                   ▼                       ▼             ║
║  ┌─────────────────────────────────────────────────────────────┐   ║
║  │                    FEATURE PIPELINE                         │   ║
║  │                                                             │   ║
║  │  ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐  │   ║
║  │  │ Preprocess  │  │ Expanding Window │  │  NLP Policy   │  │   ║
║  │  │ • dedup     │  │ • taxa_hist      │  │  Layer        │  │   ║
║  │  │ • leakage   │  │ • qtd_hist       │  │  • TF-IDF cos │  │   ║
║  │  │   removal   │  │   (t-1 only)     │  │  • Rule flags │  │   ║
║  │  └──────┬──────┘  └────────┬─────────┘  └───────┬───────┘  │   ║
║  │         └─────────────────┬┴───────────────────┘           │   ║
║  │                            ▼                                │   ║
║  │          ┌──────────────────────────────────┐              │   ║
║  │          │        Feature Matrix X           │              │   ║
║  │          │  • Categorical OHE (20 cols)      │              │   ║
║  │          │  • Numerical + log1p (10 cols)    │              │   ║
║  │          │  • NLP policy scores (7 cols)     │              │   ║
║  │          │  • Rule flags (5 cols)            │              │   ║
║  │          │  • Supplier expanding (2 cols)    │              │   ║
║  │          └──────────────────────────────────┘              │   ║
║  └─────────────────────────────────────────────────────────────┘   ║
║                              │                                       ║
║                              ▼                                       ║
║  ┌─────────────────────────────────────────────────────────────┐   ║
║  │                LIGHTGBM CLASSIFIER                          │   ║
║  │  TimeSeriesSplit (5 folds) | class_weight=balanced          │   ║
║  │  Threshold tuned via Precision-Recall curve                 │   ║
║  └──────────────────────┬──────────────────────────────────────┘   ║
║                          │                                           ║
║           ┌──────────────┴──────────────┐                           ║
║           ▼                             ▼                           ║
║  ┌──────────────────┐        ┌────────────────────────┐            ║
║  │  MLflow Registry │        │  GCP Cloud Run         │            ║
║  │  Experiments     │        │  REST API              │            ║
║  │  Model Versions  │        │  POST /v1/predict      │            ║
║  └──────────────────┘        │  → { score, decision } │            ║
║                               └────────────────────────┘            ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 3. Tratamento de Leakage

### 3.1 Features Excluídas (Leakage Crítico)

| Feature | AUC isolado | Motivo da exclusão |
|---|---|---|
| `decisao_pos_analise` | 1.000 | Resultado da revisão — gerado APÓS a decisão |
| `motivo_fechamento` | 1.000 | Preenchido após encerramento do fluxo |

Ambas as features são **pós-decisão** — estão disponíveis apenas depois que a revisão foi concluída, ou seja, é exatamente o que o modelo tenta prever. Incluí-las seria treinar com a resposta no enunciado.

### 3.2 Leakage Temporal — `taxa_revisao_historica`

A coluna `taxa_revisao_historica` no CSV tem correlação Spearman **1.000** com a taxa observada — foi calculada sobre todo o dataset (passado + futuro). Usando essa coluna diretamente, o modelo veria informação futura durante o treino.

**Solução: Expanding Window**

```
Solicitação em t=100:
  taxa_hist_expanding = média(target) das solicitações t=1..99 do mesmo fornecedor

Solicitação em t=200:
  taxa_hist_expanding = média(target) das solicitações t=1..199 do mesmo fornecedor
```

Implementação:
```python
df['taxa_hist_expanding'] = (
    df.sort_values('data_solicitacao')
    .groupby('fornecedor_id')[TARGET]
    .transform(lambda x: x.shift(1).expanding().mean())
    .fillna(global_rate)   # prior = taxa global para fornecedores novos
)
```

---

## 4. NLP Policy Layer

### 4.1 Motivação

Com apenas 18 descrições únicas em `descricao_item`, o sinal textual direto é insuficiente (AUC=0.567). O valor da camada NLP está nos **documentos de política** — extrair regras e relevância de cada política para cada solicitação.

### 4.2 Pipeline NLP

```
  documentos_politicas.json           solicitacao_i
  ┌──────────────────────┐          ┌─────────────────────────────┐
  │ POL-001: "Itens da   │          │ context_text_i =            │
  │  categoria equip.    │          │  "equipamento padrao portal  │
  │  com quantidade > 10 │          │   alta novo Sul"            │
  │  exigem revisão..."  │          └───────────────┬─────────────┘
  │ POL-002: ...         │                          │
  │ ...                  │                          │
  └──────────┬───────────┘                          │
             │                                      │
             └─────────────┬────────────────────────┘
                           ▼
             ┌─────────────────────────┐
             │  TF-IDF Vectorizer      │
             │  (unigramas + bigramas) │
             │  fit on all texts       │
             └───────────┬─────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │  Cosine Similarity      │
             │  sol_i · pol_j          │
             │  → nlp_pol_001_i        │
             │  → nlp_pol_002_i        │
             │  → ... (7 scores)       │
             └─────────────────────────┘
```

### 4.3 Rule-Based Flags (complemento)

![*Fig. 1 — Lift empírico das flags de política derivadas dos documentos de política.*](figures/05_policy_flags_lift.png)

*Fig. 1 — Lift empírico das flags de política derivadas dos documentos de política.*


| Flag | Regra | Lift EDA |
|---|---|---|
| `flag_pol001` | `categoria=equipamento AND qtd > 10` | **2.89×** |
| `flag_pol004` | `tipo=emergencial AND possui_anexo=0` | **1.89×** |
| `flag_pol005` | `possui_anexo=0 AND valor > p75(categoria)` | **1.51×** |
| `flag_pol003` | `qtd_hist ≤ Q25 AND valor > mean(categoria)` | **1.26×** |
| `policy_flag_score` | soma das 4 flags acima (0–4) | — |

### 4.4 Justificativa da Abordagem NLP

- **Por que não embeddings/LLMs?** Vocabulário de `descricao_item` tem 18 valores únicos — embeddings não acrescentam. TF-IDF sobre políticas é suficiente e interpretável.
- **Por que cosine similarity com políticas?** Captura a *relevância semântica* de cada solicitação para cada política sem precisar de dados rotulados por política.
- **Por que manter os rule flags?** Têm lift empírico comprovado, são determinísticos, auditáveis e complementam o score contínuo do TF-IDF.

---

## 5. Estratégia de Seleção de Features

```
     Todas as features (≈ 51)
              │
              ▼
     ┌─────────────────────┐
     │  Filtro de          │     Remove features com
     │  Correlação         │ ──► correlação entre si > 0.95
     │  (Pearson > 0.95)   │     (reduz redundância linear)
     └──────────┬──────────┘
                │
                ▼
     ┌─────────────────────┐
     │  LightGBM           │     Treina LightGBM no split
     │  Feature Importance │ ──► temporal 80/20
     │  (gain)             │     Seleciona top 95%
     └──────────┬──────────┘     do gain acumulado
                │
                ▼
     ┌─────────────────────┐
     │  Feature Matrix     │
     │  Final (~36 cols)   │
     │  colorida por grupo │
     │  ■ NLP/policy       │
     │  ■ supplier         │
     │  ■ tabular          │
     └─────────────────────┘
```

**Por que LightGBM gain em vez de Random Forest MDI?**

| Critério | RF MDI | LightGBM gain |
|---|---|---|
| Viés de cardinalidade | Alto (favorece features com mais valores únicos) | Baixo (pondera por amostras no split) |
| Consistência com o modelo final | Não — modelo diferente do que será deployado | Sim — mesma família de modelo |
| Velocidade | Lento (RF 200 árvores) | Rápido (LightGBM treinado de qualquer forma) |
| Interpretabilidade | MDI não indica direção | Gain indica quais features mais reduzem a perda |

`gain` mede a redução média ponderada de perda (binary cross-entropy) em cada split que usa a feature — features com gain alto efetivamente melhoram as predições, não apenas dividem os dados.

![*Fig. 2 — Feature importance por LightGBM gain. Vermelho = NLP/policy · Azul = Supplier · Verde = Tabular.*](figures/01_feature_importance.png)

*Fig. 2 — Feature importance por LightGBM gain. Vermelho = NLP/policy · Azul = Supplier · Verde = Tabular.*


---

## 6. Seleção e Justificativa do Modelo

### Candidatos avaliados

| Modelo | Vantagem | Limitação |
|---|---|---|
| Logistic Regression | Baseline interpretável, rápido | Não captura interações não-lineares |
| Random Forest | Robusto, sem scaling | Lento em produção, menos calibrado |
| **LightGBM** | Melhor AUC, rápido, handles imbalance | Caixa-preta (mitigado com SHAP) |
| XGBoost | Similar ao LightGBM | Mais lento, menos eficiente em memória |

### Por que LightGBM?

1. **Melhor AUC-ROC** em TimeSeriesSplit cross-validation vs. todos os concorrentes
2. **Gradient-based one-side sampling (GOSS)** — eficiente com dados desbalanceados
3. **Leaf-wise growth** — captura interações não-lineares das features de política
4. **Suporte nativo a `class_weight`** — trata o desbalanceamento 7.6:1 sem oversampling
5. **Velocidade de inferência** — < 1ms por predição em CPU, viável para Cloud Run
6. **Interpretabilidade via SHAP** — valores SHAP permitem explicar cada predição ao analista

---

## 7. Métricas de Avaliação

### 7.1 Mapa Completo de Métricas

```
╔══════════════════════════════════════════════════════════════════╗
║                  MÉTRICAS DE AVALIAÇÃO                          ║
╠══════════════════════════╦═══════════════════════════════════════╣
║  DISCRIMINAÇÃO           ║  PONTO DE OPERAÇÃO                   ║
║  (threshold-free)        ║  (threshold tuned via PR curve)      ║
║                          ║                                       ║
║  AUC-ROC                 ║  F1-score (threshold ótimo)          ║
║  └─ mede separação geral ║  ├─ Precision: evita falsos alarmes   ║
║     entre classes        ║  └─ Recall: captura casos reais       ║
║                          ║                                       ║
║  Average Precision       ║  Threshold: argmax(F1) na PR curve   ║
║  └─ área sob PR curve    ║                                       ║
║     mais informativa com ║                                       ║
║     imbalance (7.6:1)    ║                                       ║
╠══════════════════════════╬═══════════════════════════════════════╣
║  NEGÓCIO                 ║  PRODUÇÃO (drift monitoring)         ║
║                          ║                                       ║
║  Taxa de captura (recall)║  PSI por feature     alerta > 0.20   ║
║  └─ % casos prioritários ║  AUC semanal         alerta −0.05    ║
║     que entram na fila   ║  Score médio         alerta 2σ       ║
║                          ║  Calibration error   alerta > 0.10   ║
║  Workload da fila        ║                                       ║
║  └─ % fila priorizada    ║  Frequência: semanal (ground truth   ║
║     (custo operacional)  ║  disponível T+7d após cada revisão)  ║
╚══════════════════════════╩═══════════════════════════════════════╝
```

### 7.2 Métricas de Desenvolvimento

![*Fig. 3 — Comparação LR vs LightGBM em TimeSeriesSplit CV (5 folds, AUC-ROC e Average Precision).*](figures/02_model_comparison.png)

*Fig. 3 — Comparação LR vs LightGBM em TimeSeriesSplit CV (5 folds, AUC-ROC e Average Precision).*


| Métrica | Valor alvo | Por que esta métrica? |
|---|---|---|
| **AUC-ROC** | > 0.80 | Threshold-free; mede se o modelo ranqueia positivos acima de negativos. Benchmark natural para sistemas de triagem onde a ordem importa mais que o corte. |
| **Average Precision** | > 0.50 | Área sob a curva Precision-Recall — muito mais informativa que AUC-ROC com imbalance 7.6:1. Penaliza pesadamente falsos positivos em datasets desbalanceados. |
| **F1 no threshold ótimo** | > 0.55 | Média harmônica de Precision e Recall no ponto de máximo F1 da curva PR. Captura o trade-off operacional real: quantos analistas são mobilizados vs. quantos casos reais são capturados. |
| **Recall @ Precision=0.70** | > 0.50 | Restrição de negócio explícita: no máximo 30% de falsos positivos na fila de revisão. |

>**Por que não acurácia?**
> Com 86.8% de negativos, um modelo que sempre prevê "não prioritário" tem acurácia de 86.8% sem utilidade alguma. AUC-ROC e Average Precision são invariantes ao threshold e ao desbalanceamento.

### 7.3 Métricas de Negócio

![*Fig. 4 — Curvas ROC, Precision-Recall e Confusion Matrix no conjunto de teste (threshold ótimo).*](figures/03_evaluation_curves.png)

*Fig. 4 — Curvas ROC, Precision-Recall e Confusion Matrix no conjunto de teste (threshold ótimo).*


| Métrica | Definição | Frequência |
|---|---|---|
| **Taxa de captura** | `recall` — % de revisões prioritárias que entram na fila | Semanal |
| **Workload de revisão** | % da fila total que recebe score ≥ threshold | Diário |
| **Precision na fila** | % das solicitações priorizadas que são de fato prioritárias | Semanal |

### 7.4 Métricas de Produção (Drift Monitoring)

| Métrica | Threshold de alerta | Ação |
|---|---|---|
| **PSI** (Population Stability Index) | > 0.20 por feature | Investigar feature; considerar retraining |
| **Score médio semanal** | Desvio > 2σ da média histórica | Alerta de data drift |
| **AUC semanal** (ground truth T+7d) | < AUC_baseline − 0.05 | Trigger de retraining emergencial |
| **Calibration error** | > 0.10 | Recalibrar probabilidades (Platt scaling) |

**PSI — interpretação:**
```
PSI < 0.10  → distribuição estável, sem ação
PSI 0.10–0.20 → mudança moderada, monitorar
PSI > 0.20  → mudança significativa, investigar / retreinar
```

---

## 8. Estratégia de Deploy — GCP + MLflow

### 8.1 Arquitetura GCP

```
╔══════════════════════════════════════════════════════════════════════╗
║                    GCP DEPLOYMENT ARCHITECTURE                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐   ║
║  │                    DATA LAYER                               │   ║
║  │  ┌─────────────┐     ┌──────────────┐    ┌──────────────┐  │   ║
║  │  │  BigQuery   │     │  GCS Bucket  │    │  Cloud SQL   │  │   ║
║  │  │ (raw data + │     │  (models +   │    │  (MLflow DB) │  │   ║
║  │  │  features)  │     │   artifacts) │    │              │  │   ║
║  │  └──────┬──────┘     └──────┬───────┘    └──────┬───────┘  │   ║
║  └─────────┼─────────────────────────────────────── ┼──────────┘  ║
║            │                   │                    │              ║
║  ┌─────────▼───────────────────▼────────────────────▼──────────┐   ║
║  │                   MLFLOW TRACKING SERVER                     │   ║
║  │  (Cloud Run) Backend: Cloud SQL | Artifacts: GCS             │   ║
║  │                                                              │   ║
║  │  Experiments ──► Runs ──► Metrics/Params ──► Model Registry │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │                TRAINING PIPELINE                             │   ║
║  │  Cloud Composer (Airflow) / Cloud Scheduler                  │   ║
║  │                                                              │   ║
║  │  trigger ──► Vertex AI Pipeline ──► evaluate ──► register   │   ║
║  │     │          (feature eng +          │            │        │   ║
║  │  Pub/Sub        LightGBM fit)       promote?    MLflow       │   ║
║  │  (drift alert)                      champion    Registry     │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │                SERVING LAYER                                 │   ║
║  │                                                              │   ║
║  │  Solicitação nova ──► Pub/Sub ──► Cloud Run (API)            │   ║
║  │                                       │                      │   ║
║  │                                  load model from             │   ║
║  │                                  MLflow Registry             │   ║
║  │                                       │                      │   ║
║  │                          POST /v1/predict                    │   ║
║  │                          { "score": 0.82,                    │   ║
║  │                            "decision": "PRIORITÁRIA",        │   ║
║  │                            "top_reasons": [...] }            │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │                MONITORING LAYER                              │   ║
║  │  Cloud Monitoring + Looker Studio                            │   ║
║  │                                                              │   ║
║  │  • Latência de inferência (p50/p95/p99)                      │   ║
║  │  • Score distribution drift (PSI)                            │   ║
║  │  • AUC semanal (ground truth T+7d)                           │   ║
║  │  • Workload de revisão (% fila priorizada)                   │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 8.2 Componentes GCP

| Componente | Serviço GCP | Função |
|---|---|---|
| Dados brutos | **BigQuery** | Armazenamento e queries de features |
| Artefatos de modelo | **Cloud Storage** | `.pkl`, feature schemas, thresholds |
| Tracking server | **Cloud Run** (MLflow) | UI de experimentos e model registry |
| Metadados MLflow | **Cloud SQL** (PostgreSQL) | Banco do MLflow |
| Treino | **Vertex AI Pipelines** | Pipeline reproduzível e auditável |
| Agendamento | **Cloud Scheduler** | Trigger de retraining (semanal) |
| Alertas de drift | **Pub/Sub** | Trigger de retraining emergencial |
| Serving | **Cloud Run** | API REST de inferência (auto-scaling) |
| Monitoramento | **Cloud Monitoring** | Métricas operacionais e de ML |

### 8.3 MLflow — Configuração

```python
# mlflow_config.py
import mlflow

TRACKING_URI = "postgresql://user:pass@cloud-sql-host/mlflow"
ARTIFACT_URI = "gs://acol-mlflow-artifacts"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("acol-prioritization")

# Estrutura de tags para rastreabilidade
with mlflow.start_run(tags={
    "team"        : "data-science",
    "version"     : "v1.2",
    "trigger"     : "scheduled_weekly",
    "data_start"  : "2025-01-01",
    "data_end"    : "2025-12-31",
}) as run:
    mlflow.log_params({...})
    mlflow.log_metrics({...})
    mlflow.lightgbm.log_model(
        model,
        artifact_path="model",
        registered_model_name="acol-prioritizer",
        signature=infer_signature(X_train, y_pred_proba)
    )
    # Promover para produção se AUC > champion
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="acol-prioritizer",
        version=run.info.run_id,
        stage="Production"
    )
```

---

## 9. Loop de Melhoria Contínua

```
╔═══════════════════════════════════════════════════════════════╗
║              CONTINUOUS IMPROVEMENT LOOP                      ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   ┌──────────────────────────────────────────────────────┐   ║
║   │                                                      │   ║
║   │   PRODUÇÃO                                           │   ║
║   │   Modelo serve predições ──────────────────────┐    │   ║
║   │                                                │    │   ║
║   │   Analistas revisam solicitações               │    │   ║
║   │   priorizadas pelo modelo                      │    │   ║
║   │                │                               │    │   ║
║   │                ▼                               ▼    │   ║
║   │   Ground truth coletado (T+7d)     Score dist  │   │   ║
║   │   (confirmação da revisão)         armazenado  │   │   ║
║   └──────────────────────────────────────────────────────┘   ║
║                    │                               │          ║
║                    ▼                               ▼          ║
║   ┌──────────────────────┐    ┌──────────────────────────┐   ║
║   │  PERFORMANCE         │    │  DRIFT MONITOR           │   ║
║   │  MONITORING          │    │                          │   ║
║   │                      │    │  PSI por feature         │   ║
║   │  AUC semanal         │    │  Score distribution      │   ║
║   │  F1 semanal          │    │  Covariate shift         │   ║
║   │  Calibration error   │    │  Concept drift           │   ║
║   └──────────┬───────────┘    └──────────────┬───────────┘   ║
║              │                               │                ║
║              ▼                               ▼                ║
║   ┌──────────────────────────────────────────────────────┐   ║
║   │              RETRAINING TRIGGER                      │   ║
║   │                                                      │   ║
║   │  Scheduled: toda segunda-feira (semanal)             │   ║
║   │  Emergency: AUC < baseline-0.05 OU PSI > 0.20       │   ║
║   │                                                      │   ║
║   │  → Vertex AI Pipeline executa                        │   ║
║   │    feature_engineering + train + evaluate            │   ║
║   └──────────────────────┬───────────────────────────────┘   ║
║                          │                                     ║
║                          ▼                                     ║
║   ┌──────────────────────────────────────────────────────┐   ║
║   │           CHAMPION / CHALLENGER                      │   ║
║   │                                                      │   ║
║   │  Novo modelo treina em shadow mode                   │   ║
║   │  Avaliação: AUC, F1, calibração vs. champion         │   ║
║   │                                                      │   ║
║   │  IF novo_AUC > champion_AUC + 0.01:                 │   ║
║   │    → promover para Production no MLflow Registry     │   ║
║   │    → deprecar versão anterior                        │   ║
║   │  ELSE:                                               │   ║
║   │    → manter champion, logar challenger               │   ║
║   └──────────────────────┬───────────────────────────────┘   ║
║                          │                                     ║
║                          └──────► volta para PRODUÇÃO         ║
╚═══════════════════════════════════════════════════════════════╝
```

### 9.1 Estratégia de Dados para Retraining

| Estratégia | Descrição | Quando usar |
|---|---|---|
| **Rolling window** | Últimos N meses de dados | Concept drift detectado |
| **Expanding window** | Todo o histórico disponível | Dataset pequeno, sem drift |
| **Weighted recent** | Amostras recentes com peso maior | Drift gradual |

Recomendação inicial: **rolling window de 12 meses**, revisado trimestralmente.

![*Fig. 5 — F1 / Precision / Recall em função do threshold de decisão (LightGBM, split temporal).*](figures/04_threshold_analysis.png)

*Fig. 5 — F1 / Precision / Recall em função do threshold de decisão (LightGBM, split temporal).*


---

## 10. Monitoramento em Produção — MLflow

### 10.1 Dashboard de Métricas

```python
# monitoring/weekly_eval.py
import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score

def log_weekly_performance(week_start, predictions_df, ground_truth_df):
    """
    Executado semanalmente após coleta do ground truth (T+7d).
    Loga métricas de performance no MLflow e dispara alerta se necessário.
    """
    merged = predictions_df.merge(ground_truth_df, on='id_solicitacao')

    auc  = roc_auc_score(merged['actual'], merged['score'])
    psi  = compute_psi(merged['score'], baseline_scores)

    with mlflow.start_run(run_name=f"monitoring-{week_start}",
                          tags={"type": "monitoring", "week": week_start}):
        mlflow.log_metrics({
            "monitor_auc"          : round(auc, 4),
            "monitor_psi_score"    : round(psi, 4),
            "monitor_workload_pct" : round(merged['decision'].mean(), 4),
            "monitor_precision"    : round(precision_score(
                                        merged['actual'],
                                        merged['decision']), 4),
        })

    if auc < CHAMPION_AUC - 0.05 or psi > 0.20:
        trigger_retraining_pipeline(reason=f"AUC={auc:.3f} PSI={psi:.3f}")
```

### 10.2 Estrutura do Model Registry

```
MLflow Model Registry — acol-prioritizer
│
├── Version 1  [Archived]   train_date=2025-01-15  AUC=0.77
├── Version 2  [Archived]   train_date=2025-04-01  AUC=0.80
├── Version 3  [Production] train_date=2025-07-15  AUC=0.83  ← current
└── Version 4  [Staging]    train_date=2025-10-01  AUC=0.84  ← challenger
    tags:
      trigger   = scheduled_weekly
      data_end  = 2025-10-01
      cv_auc    = 0.836 ± 0.012
      threshold = 0.42
```

---

## 11. API de Serving — Schema

### Request
```json
POST /v1/predict
{
  "id_solicitacao"      : "SOL-99999",
  "fornecedor_id"       : "FORN_042",
  "categoria_item"      : "equipamento",
  "tipo_solicitacao"    : "emergencial",
  "valor_total"         : 4500.00,
  "quantidade_solicitada": 12,
  "possui_anexo"        : 0,
  "canal_entrada"       : "manual",
  "criticidade_declarada": "alta",
  "porte_fornecedor"    : "novo",
  "regiao"              : "Sul",
  "data_solicitacao"    : "2025-11-15"
}
```

### Response
```json
{
  "id_solicitacao" : "SOL-99999",
  "score"          : 0.84,
  "decision"       : "PRIORITÁRIA",
  "threshold_used" : 0.42,
  "model_version"  : 3,
  "top_reasons"    : [
    "flag_pol001: equipamento com quantidade > 10",
    "flag_pol004: emergencial sem anexo",
    "porte_fornecedor_novo: alta taxa histórica de revisão",
    "nlp_pol_001: alta similaridade com POL-001"
  ],
  "latency_ms"     : 8
}
```

---

## 12. Resumo das Decisões

| Decisão | Escolha | Justificativa |
|---|---|---|
| Leakage crítico | Excluir `decisao_pos_analise`, `motivo_fechamento` | AUC isolado = 1.0 |
| Leakage temporal | Expanding window para `taxa_revisao_historica` | Correlação 1.0 com target |
| NLP approach | TF-IDF cosine (políticas) + rule flags | Lift empírico 1.26–2.89×; vocabulário `descricao_item` insuficiente para LLM |
| Modelo | LightGBM | Melhor AUC em TimeSeriesSplit, rápido, SHAP-interpretável |
| Feature selection | LightGBM gain + correlação filter | Consistente com modelo final; gain menos enviesado que RF MDI |
| Validação | TimeSeriesSplit (5 folds) | Evita look-ahead bias |
| Imbalance | `class_weight='balanced'` | 7.6:1; oversampling evitado para preservar distribuição temporal |
| Threshold | Tuned via PR curve (max F1) | Ponto de operação alinhado ao trade-off de negócio |
| Deploy | GCP Cloud Run + Vertex AI | Serverless, escalável, integrado ao ecossistema GCP |
| Tracking | MLflow (Cloud SQL + GCS) | Open-source, portável, suporta model registry e versioning |
| Retraining | Semanal agendado + emergencial (drift) | Equilíbrio entre custo e freshness do modelo |
