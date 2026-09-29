# Datathon Passos Mágicos — Predição de Risco Educacional

Projeto desenvolvido para o **Datathon da Pós-Tech**, utilizando dados da **Associação Passos Mágicos**.

O objetivo do trabalho é explorar os indicadores educacionais dos alunos, analisar sua evolução ao longo do tempo e desenvolver um modelo de Machine Learning capaz de estimar a probabilidade de um aluno entrar em situação de defasagem no ciclo seguinte.

---

## Contexto

A Associação Passos Mágicos atua na transformação da vida de crianças e jovens por meio da educação.

O Datathon propõe uma análise dos indicadores educacionais e psicossociais dos alunos, com foco em:

- adequação ao nível;
- desempenho acadêmico;
- engajamento;
- autoavaliação;
- aspectos psicossociais;
- aspectos psicopedagógicos;
- ponto de virada;
- evolução do INDE;
- risco de defasagem;
- efetividade do programa.

Além da análise exploratória, o projeto inclui um **modelo preditivo de risco** e uma aplicação em **Streamlit**.

---

## Objetivos do projeto

O projeto foi estruturado para responder três perguntas principais:

1. **Como os indicadores educacionais evoluíram entre 2022 e 2024?**
2. **Quais indicadores estão mais associados ao desenvolvimento educacional e à defasagem?**
3. **É possível identificar antecipadamente alunos com maior risco de entrar em defasagem?**

---

## Indicadores analisados

| Indicador | Descrição |
|---|---|
| **INDE** | Índice de Desenvolvimento Educacional |
| **IAN** | Indicador de Adequação ao Nível |
| **IDA** | Indicador de Aprendizagem |
| **IEG** | Indicador de Engajamento |
| **IAA** | Indicador de Autoavaliação |
| **IPS** | Indicador Psicossocial |
| **IPP** | Indicador Psicopedagógico |
| **IPV** | Indicador de Ponto de Virada |

---

## Estrutura do projeto

```text
DATATHON 5/
│
├── dashboard/
│   └── arquivos do Power BI
│
├── data/
│   ├── raw/
│   │   └── base original
│   │
│   └── processed/
│       ├── base_tratada.csv
│       ├── base_modelo_temporal.csv
│       ├── relatorio_qualidade.csv
│       ├── metricas_modelo.csv
│       ├── importancia_features.csv
│       └── predicoes_teste_modelo.csv
│
├── docs/
│   └── documentação e dicionário de dados
│
├── images/
│   └── imagens utilizadas na documentação
│
├── models/
│   ├── modelo_risco.joblib
│   └── modelo_risco_metadata.json
│
├── notebooks/
│   ├── 01_exploracao.ipynb
│   ├── 02_tratamento.ipynb
│   ├── 03_analise_exploratoria.ipynb
│   └── 04_modelo_ml.ipynb
│
├── src/
│   └── códigos auxiliares
│
├── streamlit/
│   └── app.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Pipeline do projeto

```text
Base original
     ↓
Exploração dos dados
     ↓
Limpeza e padronização
     ↓
Base longitudinal 2022–2024
     ↓
Análise exploratória
     ↓
Feature Engineering
     ↓
Modelo preditivo
     ↓
Streamlit
     ↓
Dashboard / Storytelling
```

---

# Etapas desenvolvidas

## 1. Exploração dos dados

Notebook:

```text
notebooks/01_exploracao.ipynb
```

Nesta etapa foram realizadas:

- identificação das abas da base;
- análise de dimensões;
- verificação dos tipos de dados;
- análise de valores ausentes;
- identificação de duplicidades;
- análise dos alunos presentes em múltiplos anos;
- exploração dos indicadores educacionais;
- análise inicial da variável de defasagem;
- análise de correlações.

---

## 2. Tratamento e padronização

Notebook:

```text
notebooks/02_tratamento.ipynb
```

Principais tratamentos:

- padronização dos nomes das colunas;
- correção de tipos;
- tratamento de inconsistências;
- padronização entre 2022, 2023 e 2024;
- criação de uma base longitudinal;
- criação de uma base temporal específica para Machine Learning;
- validações de qualidade dos dados.

Arquivos gerados:

```text
data/processed/base_tratada.csv
data/processed/base_modelo_temporal.csv
data/processed/relatorio_qualidade.csv
```

---

## 3. Análise exploratória

Notebook:

```text
notebooks/03_analise_exploratoria.ipynb
```

A análise foi estruturada para responder às principais perguntas do Datathon.

Entre os resultados observados:

- redução da proporção de alunos em defasagem ao longo do período analisado;
- evolução positiva do INDE médio;
- redução dos casos mais severos de inadequação ao nível;
- relações relevantes entre engajamento, aprendizagem, ponto de virada e desenvolvimento geral;
- identificação de alunos que saíram da defasagem e alunos que entraram em defasagem entre 2023 e 2024.

### Alguns resultados

| Indicador | 2022 | 2023 | 2024 |
|---|---:|---:|---:|
| Alunos em defasagem | 69,9% | 54,4% | 46,2% |
| INDE médio | 7,04 | 7,34 | 7,40 |
| Casos com IAN = 2,5 | 28 | 14 | 3 |

Entre os alunos acompanhados entre 2023 e 2024:

- **43,3%** dos alunos que estavam em defasagem em 2023 saíram dessa condição em 2024;
- **22,7%** dos alunos que estavam adequados em 2023 entraram em defasagem em 2024.

Esse segundo grupo foi utilizado como base do problema preditivo.

---

# Modelo de Machine Learning

Notebook:

```text
notebooks/04_modelo_ml.ipynb
```

## Problema preditivo

O objetivo do modelo é estimar:

> A probabilidade de um aluno que estava sem defasagem em 2023 entrar em situação de defasagem em 2024.

A abordagem temporal foi escolhida para evitar **data leakage** e aproximar o modelo de uma situação real de prevenção.

---

## População utilizada

Foram considerados:

- **370 alunos elegíveis**
- **286 alunos permaneceram sem defasagem**
- **84 alunos entraram em defasagem**
- taxa positiva: aproximadamente **22,7%**

---

## Modelos avaliados

Foram comparados:

- Regressão Logística;
- Random Forest;
- HistGradientBoosting.

A comparação foi realizada com validação cruzada estratificada.

---

## Modelo selecionado

O modelo final selecionado foi:

```text
Random Forest
```

Como o objetivo é identificar o máximo possível de alunos em risco, a seleção deu maior importância ao **Recall** e ao **F2 Score**.

---

## Resultados do modelo

Resultados no conjunto de teste:

| Métrica | Resultado |
|---|---:|
| Recall | **90,5%** |
| Precision | **65,5%** |
| F2 Score | **84,1%** |
| ROC-AUC | **95,3%** |
| PR-AUC | **85,8%** |

O limiar padrão de 50% foi ajustado para aproximadamente:

```text
41%
```

Essa decisão prioriza a redução de falsos negativos.

No conjunto de teste:

- 21 alunos entraram em defasagem;
- o modelo identificou 19 deles;
- 2 casos não foram identificados;
- foram gerados 10 falsos positivos.

O modelo foi pensado como uma ferramenta de **triagem preventiva**, e não como um sistema de decisão automática.

---

## Principais variáveis preditivas

A análise de `Permutation Importance` indicou relevância preditiva em variáveis como:

- idade;
- fase;
- IPV;
- tempo / ano de ingresso;
- IPP;
- indicadores agregados criados durante o Feature Engineering.

> Importância preditiva não significa relação causal.

---

# Aplicação Streamlit

A aplicação permite inserir os dados de um aluno e obter:

- probabilidade estimada de entrada em defasagem;
- comparação com o limiar do modelo;
- classificação visual para priorização;
- informações sobre o desempenho do modelo;
- visualização das principais variáveis preditivas.

Arquivo principal:

```text
streamlit/app.py
```

---

## Executando o projeto localmente

### 1. Clone o repositório

```bash
git clone URL_DO_REPOSITORIO
```

### 2. Entre na pasta

```bash
cd datathon-passos-magicos
```

### 3. Crie um ambiente virtual

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

### 5. Execute o Streamlit

```bash
python -m streamlit run streamlit/app.py
```

O aplicativo ficará disponível normalmente em:

```text
http://localhost:8501
```

---

# Deploy

O projeto foi preparado para publicação no **Streamlit Community Cloud**.

Arquivo principal:

```text
streamlit/app.py
```

Branch:

```text
main
```

Após o deploy, adicionar o link abaixo:

```text
https://SEU-APP.streamlit.app
```

---

# Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Altair
- Streamlit
- Jupyter Notebook
- Power BI
- Git
- GitHub

---

# Arquivos do modelo

O modelo treinado é armazenado em:

```text
models/modelo_risco.joblib
```

As informações auxiliares estão em:

```text
models/modelo_risco_metadata.json
```

---

# Limitações

Algumas limitações importantes devem ser consideradas:

- a amostra temporal disponível para o problema preditivo é relativamente pequena;
- existe apenas uma transição temporal completa utilizada para o modelo: 2023 → 2024;
- a avaliação foi realizada sobre dados históricos internos;
- alguns indicadores possuíam valores ausentes;
- as probabilidades representam estimativas estatísticas;
- o modelo não deve substituir a avaliação pedagógica, psicológica ou psicopedagógica.

---

# Uso responsável

A aplicação foi desenvolvida como uma ferramenta de **apoio à identificação preventiva de risco educacional**.

O resultado do modelo deve ser utilizado em conjunto com a análise das equipes responsáveis pelo acompanhamento dos alunos.

A previsão não deve ser utilizada como decisão automática sobre:

- permanência do aluno no programa;
- concessão de benefícios;
- acompanhamento psicológico;
- avaliação acadêmica;
- qualquer decisão que possa afetar diretamente a trajetória do aluno.

---

# Próximas etapas

- [x] Exploração dos dados
- [x] Tratamento e padronização
- [x] Análise exploratória
- [x] Machine Learning
- [x] Aplicação Streamlit
- [ ] Deploy no Streamlit Community Cloud
- [ ] Dashboard Power BI
- [ ] Apresentação gerencial
- [ ] Vídeo final do Datathon

---

## Autores

Projeto desenvolvido como parte do **Datathon da Pós-Tech**.

> Adicione aqui os nomes dos integrantes do grupo antes da entrega final.
