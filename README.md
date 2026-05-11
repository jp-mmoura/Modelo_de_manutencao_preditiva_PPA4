# Modelo_de_manutencao_preditiva_PPA4
# Manutenção Preditiva com Machine Learning e RAG

Atividade prática de classificação de falhas em equipamentos industriais usando o dataset **AI4I 2020 Predictive Maintenance**, com análise dos resultados via **RAG (Retrieval-Augmented Generation)** usando LangChain e Google Gemini.

## Sobre o projeto

O notebook treina um modelo de Random Forest para prever falhas em máquinas industriais e, em seguida, monta uma pipeline de RAG que responde perguntas técnicas e de negócio com base nas métricas geradas.

```
Dataset AI4I 2020 → EDA → Pré-processamento → Modelo → Métricas → Relatório → RAG → Análise com IA
```

## Estrutura do repositório

```
├── manutencao_preditiva_RAG.ipynb   # Notebook principal (resolvido)
├── ai4i2020.csv                     # Dataset (baixar separadamente — ver abaixo)
└── README.md
```
## Como executar no Google Colab

### 1. Abrir o notebook
Faça upload do `.ipynb` no [Google Colab](https://colab.research.google.com/) ou abra direto do GitHub via `File → Open notebook → GitHub`.

### 2. Configurar a chave do Google AI Studio
A parte de RAG usa a API do Gemini. Para obter sua chave:

1. Acesse [Google AI Studio](https://aistudio.google.com/) e faça login
2. Clique em **Get API key → Create API key**
3. Copie a chave gerada

Para configurar no Colab **sem expor a chave no código**:

1. Clique no ícone 🔑 **Secrets** no menu lateral esquerdo
2. Clique em **Add new secret**
3. Nome: `GOOGLE_API_KEY` — Valor: sua chave (`AIza...`)
4. Ative o toggle **Notebook access**

### 3. Fazer upload do dataset
Coloque o `ai4i2020.csv` no painel **Files** do Colab (ícone de pasta no menu lateral). O notebook localiza o arquivo automaticamente.

### 4. Executar
`Runtime → Run all` — ou execute célula por célula para acompanhar cada etapa.

## Tecnologias utilizadas

- **Python 3.10+**
- `scikit-learn` — pré-processamento e modelo (Random Forest)
- `pandas` / `numpy` — manipulação de dados
- `matplotlib` / `seaborn` — visualizações
- `LangChain` — pipeline RAG
- `langchain-google-genai` — integração com Gemini
- `FAISS` — base vetorial local
- `Google Gemini 2.5 Flash` — modelo de linguagem para as respostas

## O que o notebook cobre

- Análise exploratória com gráficos de distribuição e correlação
- Remoção de colunas com data leakage (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`)
- Baseline com `DummyClassifier` para comparação
- Treinamento com `class_weight='balanced'` para lidar com desbalanceamento (~97% sem falha)
- Avaliação com Accuracy, Precision, Recall, F1, ROC-AUC e PR-AUC
- Análise de threshold (0.05 a 0.95) com impacto operacional
- Importância das features
- RAG que responde 12 perguntas técnicas e de negócio com base no relatório gerado
- Desafios extras: comparação de modelos, ajuste de threshold e demonstração de data leakage

## Arquivos gerados (pasta `outputs/`)

| Arquivo | Descrição |
|---|---|
| `relatorio_modelo_manutencao_preditiva.md` | Relatório completo com todas as métricas |
| `metricas_modelo.csv` | Métricas do baseline e do modelo em CSV |
| `respostas_rag.csv` | Perguntas e respostas geradas pela RAG |
