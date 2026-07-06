import json
import os
import re
import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))
DIR_QUESTOES = os.path.join(RAIZ, "questoes")
DIRS_RESULTADOS = [
    os.path.join(DIR_QUESTOES, "q1-pretreino", "resultados"),
    os.path.join(DIR_QUESTOES, "q2-sft", "resultados"),
    os.path.join(DIR_QUESTOES, "q3-lora-qlora", "resultados"),
]
DIR_Q5 = os.path.join(DIR_QUESTOES, "q5-rag", "resultados")
DIR_Q6 = os.path.join(DIR_QUESTOES, "q6-guardrails", "resultados")

COR_ANTES = "#b5651d"
COR_DEPOIS = "#12805a"
C_FULL = "#b5651d"
C_QLORA = "#12805a"
C_BONUS = "#4a3aa7"
C_BASE = "#898781"
C_BASE_INT4 = "#b3b1a9"
C_EDA = "#2a78d6"
COR_GRADE = "#e1e0d9"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.grid": False,
    "axes.axisbelow": True,
})


def carregar(nome):
    for base in DIRS_RESULTADOS:
        caminho = os.path.join(base, nome)
        if os.path.exists(caminho):
            with open(caminho, encoding="utf-8") as arquivo:
                return json.load(arquivo)
    raise FileNotFoundError(nome)


def fmt(valor, casas=2, sufixo=""):
    texto = f"{valor:,.{casas}f}"
    texto = texto.replace(",", "@").replace(".", ",").replace("@", ".")
    return texto + sufixo


def salvar(fig, nome):
    destino = os.path.join(AQUI, nome)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("gerado:", destino)


def rotular(ax, barras, casas=2, sufixo="", tamanho=10):
    for barra in barras:
        altura = barra.get_height()
        ax.annotate(fmt(altura, casas, sufixo),
                    xy=(barra.get_x() + barra.get_width() / 2, altura),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=tamanho)


def paineis_metricas(series, nome_arquivo, titulo=None, rotacao=0):
    paineis = [("Entropia Cruzada", "entropia_cruzada", 3),
               ("Perplexidade", "perplexidade", 2),
               ("Acurácia de Tokens (%)", "acuracia_tokens", 1)]
    largura = 11 + max(0, len(series) - 2) * 1.6
    fig, eixos = plt.subplots(1, 3, figsize=(largura, 4.0))
    for ax, (nome, chave, casas) in zip(eixos, paineis):
        rotulos = [s[0] for s in series]
        valores = [s[1][chave] for s in series]
        cores = [s[2] for s in series]
        barras = ax.bar(rotulos, valores, color=cores, width=0.62)
        rotular(ax, barras, casas=casas)
        ax.set_title(nome)
        ax.set_ylim(0, max(valores) * 1.30)
        ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
        if rotacao:
            ax.tick_params(axis="x", labelrotation=rotacao)
            for rotulo in ax.get_xticklabels():
                rotulo.set_ha("right")
    if titulo:
        fig.suptitle(titulo, fontweight="bold")
    fig.tight_layout()
    salvar(fig, nome_arquivo)


def curva_loss(log_history, nome_arquivo, titulo):
    pontos_treino = [(h["step"], h["loss"]) for h in log_history
                     if "loss" in h and "eval_loss" not in h]
    pontos_eval = [(h["step"], h["eval_loss"]) for h in log_history if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    if pontos_treino:
        xs, ys = zip(*pontos_treino)
        ax.plot(xs, ys, color=C_EDA, linewidth=1.6, label="Loss de treino")
    if pontos_eval:
        xs, ys = zip(*pontos_eval)
        ax.plot(xs, ys, color="#184f95", marker="o", linestyle="--",
                linewidth=1.3, markersize=6, label="Loss de avaliação (por época)")
        for x, y in pontos_eval:
            ax.annotate(fmt(y, 3), xy=(x, y), xytext=(-4, 8),
                        textcoords="offset points", fontsize=9,
                        color="#184f95", fontweight="bold", ha="right")
    ax.set_xlabel("Passo de treinamento")
    ax.set_ylabel("Loss")
    ax.set_title(titulo)
    ax.legend(frameon=False)
    ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
    fig.tight_layout()
    salvar(fig, nome_arquivo)


def normalizar_texto(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", texto)


def acertos_por_categoria(resultados):
    contagem = {}
    for item in resultados:
        acertou = normalizar_texto(item["resposta_esperada"]) in normalizar_texto(item["resposta_gerada"])
        ok, total = contagem.get(item["cat"], (0, 0))
        contagem[item["cat"]] = (ok + (1 if acertou else 0), total + 1)
    return contagem


def figura_q1_metricas(q1):
    paineis_metricas([("Antes", q1["metricas_antes"], COR_ANTES),
                      ("Depois", q1["metricas_depois"], COR_DEPOIS)],
                     "q1_metricas.png")


def figura_q1_curva(q1):
    curva_loss(q1["log_history"], "q1_curva_loss.png",
               "Pré-treino continuado no DOMPI-2025 (2 épocas)")


def figura_q1_benchmark():
    antes = carregar("resultados_benchmark_base.json")
    depois = carregar("resultados_benchmark_treinado.json")
    cont_antes = acertos_por_categoria(antes)
    cont_depois = acertos_por_categoria(depois)
    categorias = sorted(set(cont_antes) | set(cont_depois))
    posicoes = np.arange(len(categorias))
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for deslocamento, contagem, cor, rotulo in ((-0.19, cont_antes, COR_ANTES, "Antes"),
                                                (0.19, cont_depois, COR_DEPOIS, "Depois")):
        valores = [contagem.get(c, (0, 1))[0] / contagem.get(c, (0, 1))[1] * 100 for c in categorias]
        barras = ax.bar(posicoes + deslocamento, valores, width=0.38, color=cor, label=rotulo)
        for barra, categoria in zip(barras, categorias):
            ok, total = contagem.get(categoria, (0, 0))
            ax.annotate(f"{ok}/{total}",
                        xy=(barra.get_x() + barra.get_width() / 2, barra.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold", color=cor)
    ax.set_xticks(posicoes)
    ax.set_xticklabels(categorias)
    ax.set_ylabel("Respostas contendo a referência (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Benchmark de 25 perguntas por categoria\n"
                 "(critério: resposta de referência contida na resposta gerada, após normalização)",
                 fontsize=11)
    ax.legend(frameon=False)
    ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
    fig.tight_layout()
    salvar(fig, "q1_benchmark_qa.png")


def figura_q2_funil(q2):
    funil = q2["pipeline"]
    etapas = [("Registros brutos", funil["brutos"]),
              ("Após limpeza e filtro de tamanho", funil["apos_limpeza_e_tamanho"]),
              ("Após deduplicação", funil["apos_dedup"]),
              ("Após amostragem por professor", funil["apos_amostragem"]),
              ("Pares gerados", funil["pares_gerados"])]
    nomes = [e[0] for e in etapas][::-1]
    valores = [e[1] for e in etapas][::-1]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    barras = ax.barh(nomes, valores, color="#4a3aa7")
    for barra in barras:
        largura = barra.get_width()
        ax.annotate(fmt(largura, 0), xy=(largura, barra.get_y() + barra.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=10, fontweight="bold")
    ax.set_xlabel("Quantidade")
    ax.set_xlim(0, max(valores) * 1.14)
    ax.set_title("Pipeline de geração de pares a partir do docentesDC")
    ax.grid(axis="x", color=COR_GRADE, linewidth=0.8)
    fig.tight_layout()
    salvar(fig, "q2_pipeline_funil.png")


def figura_q2_tipos(q2):
    tipos = q2["tipos_pares"]
    descricoes = {"A": "A\nAtribuição", "B": "B\nContinuação", "C": "C\nTema",
                  "D": "D\nFatos", "E": "E\nCódigo"}
    chaves = sorted(tipos)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    barras = ax.bar([descricoes.get(c, c) for c in chaves], [tipos[c] for c in chaves],
                    color=C_EDA, width=0.6)
    rotular(ax, barras, casas=0)
    ax.set_ylabel("Pares gerados")
    ax.set_ylim(0, max(tipos.values()) * 1.22)
    ax.set_title("Composição dos 1.977 pares por tipo de tarefa")
    ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
    fig.tight_layout()
    salvar(fig, "q2_tipos_pares.png")


def figura_q2_metricas(q2):
    paineis_metricas([("Antes", q2["metricas_antes"], COR_ANTES),
                      ("Depois", q2["metricas_depois"], COR_DEPOIS)],
                     "q2_metricas.png")


def figura_q2_curva(q2):
    curva_loss(q2["log_history"], "q2_curva_loss.png",
               "SFT completo no docentesDC (3 épocas)")


def figura_q3_quantizacao(q3):
    paineis_metricas([("Base fp32", q3["base"]["fp32"], C_BASE),
                      ("Base 4-bit (NF4)", q3["base"]["int4"], C_BASE_INT4)],
                     "q3_quantizacao.png")


def figura_q3_metricas(q3, q2):
    qlora = q3["experimentos"]["qlora"]
    paineis_metricas([("Base\nfp32", q3["base"]["fp32"], C_BASE),
                      ("Base\n4-bit", q3["base"]["int4"], C_BASE_INT4),
                      ("Full SFT\n(de fp32)", q2["metricas_depois"], C_FULL),
                      ("QLoRA\n(de 4-bit)", qlora["metricas_depois"], C_QLORA)],
                     "q3_metricas.png")


def figura_q3_params(q3, q2, bonus):
    qlora = q3["experimentos"]["qlora"]
    itens = [("Full SFT 0.5B", q2["treino"]["params_treinaveis"], 100.0, C_FULL),
             ("QLoRA 0.5B", qlora["params_treinaveis"], qlora["pct_treinavel"], C_QLORA)]
    if bonus:
        itens.append(("QLoRA 1.5B", bonus["params_treinaveis"], bonus["pct_treinavel"], C_BONUS))
    nomes = [i[0] for i in itens][::-1]
    valores = [i[1] for i in itens][::-1]
    cores = [i[3] for i in itens][::-1]
    percentuais = [i[2] for i in itens][::-1]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    barras = ax.barh(nomes, valores, color=cores, height=0.55)
    ax.set_xscale("log")
    ax.set_xlim(8e5, 6e9)
    for barra, pct in zip(barras, percentuais):
        largura = barra.get_width()
        texto = f"{fmt(largura / 1e6, 1)} M  ({fmt(pct, 2)}% do modelo)"
        ax.annotate(texto, xy=(largura, barra.get_y() + barra.get_height() / 2),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=10, fontweight="bold")
    ax.set_xlabel("Parâmetros atualizados no treino (escala logarítmica)")
    ax.set_title("Custo de adaptação por técnica")
    ax.grid(axis="x", color=COR_GRADE, linewidth=0.8)
    fig.tight_layout()
    salvar(fig, "q3_lora_params.png")


def figura_q3_painel(q3, q2, bonus):
    qlora = q3["experimentos"]["qlora"]
    entidades = [("Full SFT 0.5B", C_FULL, q2["metricas_depois"], q2["treino"]),
                 ("QLoRA 0.5B", C_QLORA, qlora["metricas_depois"], qlora["treino"])]
    if bonus:
        entidades.append(("QLoRA 1.5B", C_BONUS, bonus["metricas_depois"], bonus["treino"]))
    paineis = [("Perplexidade final (menor é melhor)", lambda m, t: m["perplexidade"], 2, ""),
               ("Acurácia de tokens final (%)", lambda m, t: m["acuracia_tokens"], 1, "%"),
               ("Tempo de treino (min)", lambda m, t: t["tempo_s"] / 60, 1, ""),
               ("Pico de VRAM (GB)", lambda m, t: t["vram_pico_gb"], 2, "")]
    fig, eixos = plt.subplots(2, 2, figsize=(10, 7.2))
    for ax, (nome, extrator, casas, sufixo) in zip(eixos.flat, paineis):
        rotulos = [e[0] for e in entidades]
        cores = [e[1] for e in entidades]
        valores = [extrator(e[2], e[3]) for e in entidades]
        barras = ax.bar(rotulos, valores, color=cores, width=0.58)
        rotular(ax, barras, casas=casas, sufixo=sufixo)
        ax.set_title(nome, fontsize=11)
        ax.set_ylim(0, max(valores) * 1.30)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
    fig.suptitle("Custo e qualidade: Full SFT vs QLoRA (mesmos dados, split e máscara de loss)",
                 fontweight="bold")
    fig.tight_layout()
    salvar(fig, "q3_painel_comparativo.png")


def figura_resumo(q1, q2, q3, bonus):
    linhas = [("Q1 Pré-treino\n(full-text)", q1["metricas_antes"], q1["metricas_depois"]),
              ("Q2 Full SFT", q2["metricas_antes"], q2["metricas_depois"]),
              ("Q3 QLoRA 0.5B", q3["experimentos"]["qlora"]["metricas_antes"],
               q3["experimentos"]["qlora"]["metricas_depois"])]
    if bonus:
        linhas.append(("QLoRA 1.5B", bonus["metricas_antes"], bonus["metricas_depois"]))
    nomes = [l[0] for l in linhas]
    posicoes = np.arange(len(nomes))
    fig, eixos = plt.subplots(1, 3, figsize=(13.5, 4.4))

    ppl_antes = [l[1]["perplexidade"] for l in linhas]
    ppl_depois = [l[2]["perplexidade"] for l in linhas]
    barras_a = eixos[0].bar(posicoes - 0.19, ppl_antes, width=0.38, color=COR_ANTES, label="Antes")
    barras_d = eixos[0].bar(posicoes + 0.19, ppl_depois, width=0.38, color=COR_DEPOIS, label="Depois")
    eixos[0].set_yscale("log")
    for barras in (barras_a, barras_d):
        for barra in barras:
            eixos[0].annotate(fmt(barra.get_height(), 1),
                              xy=(barra.get_x() + barra.get_width() / 2, barra.get_height()),
                              xytext=(0, 3), textcoords="offset points",
                              ha="center", va="bottom", fontsize=9, fontweight="bold")
    eixos[0].set_title("Perplexidade (escala log)")
    eixos[0].legend(frameon=False)

    acc_antes = [l[1]["acuracia_tokens"] for l in linhas]
    acc_depois = [l[2]["acuracia_tokens"] for l in linhas]
    barras_a = eixos[1].bar(posicoes - 0.19, acc_antes, width=0.38, color=COR_ANTES, label="Antes")
    barras_d = eixos[1].bar(posicoes + 0.19, acc_depois, width=0.38, color=COR_DEPOIS, label="Depois")
    for barras in (barras_a, barras_d):
        for barra in barras:
            eixos[1].annotate(fmt(barra.get_height(), 1),
                              xy=(barra.get_x() + barra.get_width() / 2, barra.get_height()),
                              xytext=(0, 3), textcoords="offset points",
                              ha="center", va="bottom", fontsize=9, fontweight="bold")
    eixos[1].set_title("Acurácia de tokens (%)")
    eixos[1].set_ylim(0, 100)
    eixos[1].legend(frameon=False)

    reducao = [(1 - d / a) * 100 for a, d in zip(ppl_antes, ppl_depois)]
    barras_r = eixos[2].bar(posicoes, reducao, width=0.55, color=C_EDA)
    rotular(eixos[2], barras_r, casas=1, sufixo="%")
    eixos[2].set_title("Redução relativa da perplexidade")
    eixos[2].set_ylim(0, 100)

    for ax in eixos:
        ax.set_xticks(posicoes)
        ax.set_xticklabels(nomes, fontsize=9)
        ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
    fig.suptitle("Antes e depois em cada experimento, cada um contra o próprio baseline",
                 fontweight="bold")
    fig.text(0.5, -0.02,
             "Q1 mede a sequência completa; Q2, Q3 e 1.5B medem apenas os tokens de resposta. "
             "As escalas não são comparáveis entre experimentos.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    salvar(fig, "resumo_geral.png")


def figura_q5_ragas():
    df = pd.read_csv(os.path.join(DIR_Q5, "relatorio_scores_detalhado.csv"))
    metricas = [("Fidelidade", "faithfulness"),
                ("Relevância\nda resposta", "answer_relevancy"),
                ("Precisão\ndo contexto", "context_precision")]
    valores = [float(df[chave].mean()) for _, chave in metricas]
    rotulos = [nome for nome, _ in metricas]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    barras = ax.bar(rotulos, valores, color=[C_QLORA, C_EDA, C_BONUS], width=0.6)
    rotular(ax, barras, casas=3)
    ax.axhline(0.85, color=COR_ANTES, linestyle="--", linewidth=1.3)
    ax.annotate("ideal (0,85)", xy=(len(rotulos) - 1, 0.85),
                xytext=(0, 5), textcoords="offset points", ha="center",
                color=COR_ANTES, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Pontuação média (0 a 1)")
    ax.set_title(f"Avaliação Ragas do pipeline RAG ({len(df)} perguntas)")
    ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
    fig.tight_layout()
    salvar(fig, "q5_ragas.png")


def figura_q6_guardrails():
    with open(os.path.join(DIR_Q6, "resultados_guardrails_q6.json"), encoding="utf-8") as arquivo:
        dados = json.load(arquivo)
    m = dados["metricas"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    rot = ["Detecção\nde ameaças", "Falsos\npositivos", "Disponibilidade\n(legítimas)"]
    val = [m["taxa_deteccao_ameacas_pct"], m["taxa_falso_positivo_pct"], m["disponibilidade_legitimas_pct"]]
    barras = ax1.bar(rot, val, color=[C_QLORA, COR_ANTES, C_EDA], width=0.6)
    rotular(ax1, barras, casas=1, sufixo="%")
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("%")
    ax1.set_title("Métricas de proteção")
    ax1.grid(axis="y", color=COR_GRADE, linewidth=0.8)

    matriz = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
    ax2.imshow(matriz, cmap="Greens", vmin=0, vmax=matriz.max())
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Bloqueado", "Respondido"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Ameaça", "Legítima"])
    ax2.set_xlabel("Ação do guardrail")
    ax2.set_ylabel("Tipo real da pergunta")
    limite = matriz.max() / 2
    nomes_cel = [["VP", "FN"], ["FP", "VN"]]
    for i in range(2):
        for j in range(2):
            cor = "white" if matriz[i, j] > limite else "#222222"
            ax2.text(j, i, f"{nomes_cel[i][j]}\n{matriz[i, j]}", ha="center", va="center",
                     fontsize=13, fontweight="bold", color=cor)
    ax2.set_title("Matriz de confusão")
    fig.suptitle("Guardrails: benchmark de 30 perguntas (20 legítimas + 10 ameaças)",
                 fontweight="bold")
    fig.tight_layout()
    salvar(fig, "q6_guardrails.png")


if __name__ == "__main__":
    q1 = carregar("resultados_metricas.json")
    q2 = carregar("resultados_sft_q2.json")
    q3 = carregar("resultados_lora_q3.json")
    try:
        bonus = carregar("resultados_bonus_15b.json")
    except FileNotFoundError:
        bonus = None
    figura_q1_metricas(q1)
    figura_q1_curva(q1)
    figura_q1_benchmark()
    figura_q2_funil(q2)
    figura_q2_tipos(q2)
    figura_q2_metricas(q2)
    figura_q2_curva(q2)
    figura_q3_quantizacao(q3)
    figura_q3_metricas(q3, q2)
    figura_q3_params(q3, q2, bonus)
    figura_q3_painel(q3, q2, bonus)
    figura_resumo(q1, q2, q3, bonus)
    figura_q5_ragas()
    figura_q6_guardrails()
