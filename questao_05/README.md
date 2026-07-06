# RAG para Dados Acadêmicos (Departamento de Computação - docentesDC)

Este projeto implementa uma solução de **Standard RAG (Retrieval-Augmented Generation)** focada em busca semântica por embeddings para o dataset acadêmico `docentesDC`. A arquitetura foi estruturada de forma modular, permitindo que cada etapa possa ser facilmente convertida em nós de um grafo de decisão com **LangGraph** no futuro.

---

## 📂 Estrutura do Projeto

*   **[RAG_DocentesDC.ipynb](file:///c:/Users/hever/Desktop/Q5/RAG_DocentesDC.ipynb)**: Jupyter Notebook principal contendo a implementação passo a passo dividida em 5 etapas (Configuração, Baseline sem RAG, Pipeline com RAG, Análise Prática e Benchmark de Avaliação).
*   **[generate_notebook.py](file:///c:/Users/hever/Desktop/Q5/generate_notebook.py)**: Script Python auxiliar utilizado para gerar/atualizar o notebook `.ipynb` programaticamente, incluindo a etapa de avaliação por Ragas.
*   **docentesDC.parquet / docentesDC.jsonl**: Os datasets com os perfis e produções docentes locais carregados no início da pipeline.
*   **.env**: Arquivo de configuração contendo as credenciais de API necessárias (ex: `OPENAI_API_KEY`).
*   **.venv/**: Ambiente virtual contendo todas as dependências pré-instaladas.

---

## 🛠️ Configuração e Instalação

### Pré-requisitos
*   [Python 3.10+](https://www.python.org/) instalado.
*   [uv](https://github.com/astral-sh/uv) (ferramenta rápida de empacotamento Python) instalada.

### Instalação Rápida com `uv`

Para recriar o ambiente virtual e instalar todas as dependências necessárias, execute o seguinte comando no terminal da raiz do projeto:

```bash
# Cria o ambiente virtual
uv venv

# Instala todas as dependências requeridas no ambiente virtual
uv pip install ipykernel langchain langchain-openai langchain-community langchain-ollama chromadb pandas pyarrow fastparquet python-dotenv ragas datasets
```

---

## 🚀 Como Executar

### Opção 1: Via VS Code (Recomendado)
1. Certifique-se de que a extensão **Jupyter** do VS Code está ativa.
2. Abra o arquivo **[RAG_DocentesDC.ipynb](file:///c:/Users/hever/Desktop/Q5/RAG_DocentesDC.ipynb)** no VS Code.
3. Clique em **Select Kernel** no canto superior direito e selecione a opção **Python Environments...** -> **`.venv`** (que aponta para a pasta do ambiente virtual local `./env`).
4. Crie ou configure seu arquivo `.env` na raiz do projeto contendo a sua chave da OpenAI:
   ```env
   OPENAI_API_KEY="sua-chave-aqui"
   ```
5. Execute as células sequencialmente.

### Opção 2: Via Jupyter Classic no Navegador
Se preferir rodar a interface clássica do Jupyter, execute no terminal:

```bash
# Inicia o servidor do Jupyter Notebook
uv run jupyter notebook
```
Acesse o link gerado no seu navegador e abra o arquivo `RAG_DocentesDC.ipynb`.

---

## 🧠 Arquitetura RAG Implementada

A pipeline é dividida em 5 etapas detalhadas:

1.  **Etapa 1: Carga e Análise Exploratória**: Lê o dataset `.parquet` (com fallback robusto linha a linha para o formato `.jsonl` em caso de falha de driver) e exibe estatísticas básicas sobre os docentes indexados.
2.  **Etapa 2: Baseline sem RAG (LLM Pura)**: Cria uma chamada direta à LLM `gpt-4o-mini` sem contexto adicional, exemplificando a taxa de alucinação de modelos gerais em dados privados.
3.  **Etapa 3: Engenharia de Dados RAG**:
    *   **Document Loading**: Converte o DataFrame pandas em objetos `Document` da classe LangChain preservando metadados dos professores.
    *   **Text Splitter**: Utiliza o `RecursiveCharacterTextSplitter` com tamanho de chunk de 1000 caracteres e overlap de 200 para preservar a estrutura semântica dos slides/códigos acadêmicos.
    *   **Embeddings & Vector Store**: Utiliza `OpenAIEmbeddings` (ou localmente `HuggingFaceEmbeddings`) e indexa os chunks em uma base vetorial local do **ChromaDB** persistida no diretório `./chroma_db`. **Otimização de Carregamento**: O inicializador detecta se o banco local já está criado e populado para carregá-lo instantaneamente, evitando duplicações desnecessárias.
    *   **Filtro por Metadados (Metadata Filtering)**: Inclui uma função helper `obter_filtro_professor(query)` que mapeia menções a nomes de docentes nas perguntas do usuário, ativando uma busca filtrada no ChromaDB. Isso eleva a precisão do contexto a níveis máximos, eliminando interferências semânticas e vazamento de dados de outros docentes.
    *   **Retrieval Chain**: Um Prompt Template restritivo que instrui a LLM a responder **apenas** com base no contexto fornecido, prevenindo a alucinação acadêmica.
4.  **Etapa 4: Comparação Prática**: Executa o mesmo questionamento sobre disciplinas/algoritmos ministrados pelos docentes na LLM Pura e no pipeline RAG, exibindo a comparação das respostas e suas respectivas fontes documentais.
5.  **Etapa 5: Criação do Benchmark de Avaliação (Ragas)**:
    *   **Subetapa 1: Definição do Golden Dataset**: Estabelece o conceito de *Ground Truth* e configura um conjunto fixo de **30 perguntas e respostas gabarito** cobrendo o corpo docente do DC/UFPI.
    *   **Subetapa 2: Execução em Massa**: Varre as 30 perguntas chamando a baseline e a pipeline RAG em um loop robusto (com try/except e sleep), consolidando as respostas e os contextos em um Pandas DataFrame.
    *   **Subetapa 3: Avaliação Ragas (LLM-as-a-Judge)**: Converte os resultados para Hugging Face Dataset e executa a avaliação utilizando as métricas de `faithfulness` (fidelidade), `answer_relevancy` (relevância da resposta) e `context_precision` (precisão da busca vetorial).
    *   **Subetapa 4: Análise Comparativa**: Gera tabelas comparativas de performance média e discute qualitativamente como o enriquecimento de contexto RAG reduz as alucinações da LLM.

---

## 📊 Resultados da Avaliação (Ragas)

Abaixo estão os resultados consolidados da última avaliação do benchmark de 30 perguntas (Golden Dataset):

| Métrica de Avaliação | Pontuação Média (0 a 1) | Diagnóstico Geral |
| :--- | :---: | :--- |
| **Fidelidade (Faithfulness)** | **0.6737** | Penalizado pelo Ragas devido à ausência do nome do professor no texto cru do chunk (embora presente nos metadados da busca). |
| **Relevância (Answer Relevancy)** | **0.5609** | Penalizado quando a resposta diz "não encontrei a informação no contexto" (comportamento correto do prompt). |
| **Precisão de Busca (Context Precision)** | **0.4296** | Baixo devido a lacunas no dataset (perguntas do benchmark que não possuem respostas nos documentos). |

Para uma análise de gargalos aprofundada, diagnósticos técnicos e sugestões de melhorias arquiteturais (como Parent Document Retrieval e refinamento de metadados), consulte o relatório completo em:
* [analise_resultado_rag.md](analise_resultado_rag.md)

