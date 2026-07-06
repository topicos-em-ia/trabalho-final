# Análise Detalhada dos Resultados do RAG (Ragas)

Esta análise investiga a avaliação do pipeline de **Standard RAG** realizada sobre o dataset `docentesDC`, utilizando o framework **Ragas** como juiz (*LLM-as-a-Judge* com GPT-4o-Mini).

---

## 📊 Estatísticas Gerais (Médias de Execução)

A tabela abaixo consolida as pontuações médias obtidas no conjunto de dados de validação de 30 perguntas (Golden Dataset):

| Métrica de Avaliação | Pontuação Média (0 a 1) | Significado e Status |
| :--- | :---: | :--- |
| **Fidelidade (Faithfulness)** | **0.6737** | Mede se a resposta é construída *apenas* com base no contexto recuperado. (Abaixo do ideal > 0.85). |
| **Relevância (Answer Relevancy)** | **0.5609** | Mede se a resposta responde de forma direta e completa à pergunta do usuário. (Médio/Baixo). |
| **Precisão de Busca (Context Precision)** | **0.4296** | Mede a qualidade do Vector Store em ranquear trechos úteis nas primeiras posições. (Baixo). |

---

## 🔍 Diagnóstico de Gargalos (Por que as notas estão baixas?)

Uma análise linha a linha dos logs de execução e do banco de dados revelou três problemas principais na arquitetura e na definição do benchmark:

### 1. Desalinhamento entre o "Golden Dataset" e os Dados Existentes (Lacunas no Dataset)
O conjunto de 30 perguntas do *Golden Dataset* assume a existência de informações que **não estão presentes no dataset local** `docentesDC.parquet`/`docentesDC.jsonl`.
* **Exemplo 1 (Prof. Luiz Claudio Demes):** O benchmark pergunta sobre conceitos de *Compiladores* e *Análise Léxica* ministrados por ele. A inspeção do banco revelou que o dataset dele possui **0** menções a esses temas. Seus 275 documentos são compostos apenas por slides de *Gestão de Projetos* e *PROFNIT*.
* **Exemplo 2 (Prof. Glauber Dias Gonçalves):** O benchmark pergunta sobre *Normalização de Bancos de Dados*. A busca no dataset de Glauber por termos de normalização retornou apenas **1** documento, que trata de normalização estatística em mineração de dados (*min-max* e *z-score*), e não de formas normais de banco de dados.
* **Exemplo 3 (Prof. Rodrigo Veras):** O benchmark pergunta por sua área de atuação principal (Visão Computacional, etc.). Os 212 documentos dele tratam de *Heapsort* e *Lógica Proposicional*. Não há nenhum resumo curricular ou descrição de sua linha de pesquisa no dataset.

> [!IMPORTANT]
> Quando a informação não existe no contexto, o prompt restritivo força o modelo a responder **"Não encontrei essa informação no contexto fornecido."** factual e correto. No entanto, o Ragas avalia essa resposta curta com **Answer Relevancy = 0** e a precisão do contexto como **0**, derrubando as médias artificiais.

### 2. Discrepância de Contexto na Avaliação de Fidelidade (Bug de Metadados)
Durante a execução do RAG, a função `formatar_contexto` injeta o nome do professor no prompt enviada para a LLM, por exemplo: `--- Bloco 1 (Docente Associado: LAURINDO DE SOUSA BRITTO NETO) --- [conteúdo do slide]`.
No entanto, no momento de salvar o relatório de avaliação para o Ragas:
* O script salva apenas o texto cru (`doc.page_content`) na lista de `contexts` avaliada pelo Ragas.
* Sem a associação com o nome do professor no texto cru, o LLM Juiz do Ragas lê a pergunta *"Qual a linha de atuação do professor Laurindo?"* e compara com um texto contendo projetos como *"AAREM - Avaliação Automática..."* sem nenhuma menção explícita ao nome "Laurindo".
* O juiz conclui que a resposta cometeu uma alucinação de vínculo e penaliza a **Fidelidade (Faithfulness)** para valores muito baixos (**0.33** ou **0.40**), mesmo quando o modelo seguiu perfeitamente o contexto fornecido.

### 3. Colisão Semântica de Alta Relevância (Planos de Ensino)
Em perguntas gerais (ex: *"Quais algoritmos Raimundo Moura ensina?"*), o Vector Store traz pedaços do **Plano de Ensino** (que contém termos gerais como "algoritmos de ordenação" e o nome do docente).
* Como esses termos coincidem perfeitamente com a query, o algoritmo de busca por similaridade dá a pontuação máxima a esses chunks.
* No entanto, o Plano de Ensino não detalha quais são os algoritmos específicos (Bubblesort, Mergesort, etc.). O conteúdo detalhado está nos slides de aula, que acabam sendo omitidos do contexto (crowded out) por terem menor relevância matemática imediata do que o cabeçalho do plano de ensino.

---

## 🛠️ Recomendações e Plano de Ação

Para elevar as pontuações e garantir um RAG de nível profissional, propomos as seguintes melhorias:

### A. Correção da Avaliação (Curto Prazo)
* **Alinhamento do Contexto do Ragas:** Modificar o pipeline de exportação para que o vetor de contextos enviado ao Ragas inclua as mesmas informações de metadados de docentes injetadas na LLM, evitando a penalização injusta de *Faithfulness*.
  ```python
  # Alterar no loop de benchmark:
  chunks_recuperados = [
      f"Docente: {doc.metadata.get('nome_professor', 'Desconhecido')}\n{doc.page_content}" 
      for doc in resultado_rag["fontes"]
  ]
  ```

### B. Enriquecimento dos Dados (Médio Prazo)
* **Inclusão de Perfil Curricular:** Adicionar um documento de perfil curto (resumo do Lattes) para cada docente com metadados estruturados. Isso fornecerá as respostas para perguntas sobre linhas de atuação e áreas gerais de pesquisa.
* **Saneamento do Golden Dataset:** Ajustar as perguntas do benchmark para cobrir apenas os materiais acadêmicos que de fato estão indexados nos arquivos `.parquet` e `.jsonl`.

### C. Evolução da Arquitetura (Longo Prazo)
* **Parent Document Retriever / Hierarchical Retrieval:** Indexar chunks menores para busca semântica, mas retornar documentos ou seções completas para a LLM, garantindo a retenção dos algoritmos específicos listados nas aulas.
* **Filtragem Estruturada por Agente (Self-RAG/LangGraph):** Implementar um nó de decisão para remover chunks redundantes (como múltiplos planos de ensino idênticos) antes de enviar à LLM.
