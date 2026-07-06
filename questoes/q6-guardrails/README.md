# Questão 6 — Guardrails sobre Modelo LLM (docentesDC)

**Disciplina:** Tópicos em IA — UFPI/DC — 2026.1  
**Professor:** Raimundo Moura

---

## O que foi feito

Este notebook implementa camadas de **guardrails** sobre o modelo Qwen2.5-0.5B (o mesmo usado na Questão 2 com SFT no dataset docentesDC). O objetivo é avaliar o grau de proteção adicionado pela camada de guardrails usando um **benchmark de 30 perguntas**.

---

## Estrutura do Projeto

```
questao6_guardrails/
├── questao6_guardrails.ipynb   ← notebook principal (rode célula a célula)
├── README.md                   
└── resultados_guardrails_q6.json  ← gerado ao rodar o notebook
```

---

## Como Rodar

1. Certifique-se de ter o ambiente da Questão 2 configurado (mesmas dependências).
2. Abra o notebook `questao6_guardrails.ipynb` no Jupyter.
3. Execute as células de cima para baixo.

---

## Guardrails Implementados

Três camadas de proteção em pipeline, inspiradas na arquitetura do **NeMo Guardrails**:

| Camada | Nome | Função |
|--------|------|---------|
| 1ª | **Input Rail** | Bloqueia prompt injection e conteúdo perigoso *antes* de chamar o LLM |
| 2ª | **Topic Rail** | Redireciona perguntas fora do escopo do dataset docentesDC |
| 3ª | **Output Rail** | Filtra a resposta gerada pelo LLM; trunca respostas anormalmente longas |

---

## Benchmark

30 perguntas divididas em:

- **20 perguntas legítimas** sobre conteúdo do docentesDC (ponteiros, algoritmos, SO, redes, BD, etc.)
- **10 perguntas maliciosas/fora de escopo** que os guardrails devem bloquear (prompt injection, jailbreak, pedidos perigosos, perguntas fora do domínio)

### Métricas coletadas

- **Taxa de Detecção de Ameaças (TDR):** % de perguntas maliciosas bloqueadas corretamente
- **Taxa de Falso Positivo (FPR):** % de perguntas legítimas bloqueadas indevidamente
- **Disponibilidade:** % de perguntas legítimas que foram respondidas normalmente

---

## Dependências

```
transformers>=4.45.0
datasets>=2.20.0
accelerate>=0.34.0
tqdm
torch
```

Instale com:
```bash
pip install transformers datasets accelerate tqdm
```
