import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))

COR_ANTES = "#b5651d"
COR_DEPOIS = "#1f6f54"
plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})


def rotular(ax, barras, sufixo="", casas=2):
    for b in barras:
        altura = b.get_height()
        ax.annotate(
            f"{altura:.{casas}f}{sufixo}",
            xy=(b.get_x() + b.get_width() / 2, altura),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )


def figura_q1():
    paineis = [
        ("Entropia Cruzada", 3.0812, 0.7390, "", 2),
        ("Perplexidade", 21.78, 2.09, "", 2),
        ("Acuracia de Tokens", 42.25, 82.00, "%", 1),
    ]
    fig, eixos = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, (titulo, antes, depois, suf, casas) in zip(eixos, paineis):
        barras = ax.bar(
            ["Antes", "Depois"],
            [antes, depois],
            color=[COR_ANTES, COR_DEPOIS],
            width=0.62,
        )
        rotular(ax, barras, sufixo=suf, casas=casas)
        ax.set_title(titulo)
        ax.set_ylim(0, max(antes, depois) * 1.22)
        ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    destino = os.path.join(AQUI, "q1_metricas.png")
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("gerado:", destino)


def figura_q3_lora():
    fig, ax = plt.subplots(figsize=(8, 3.4))
    rotulos = ["Full Fine-Tuning\n(Questao 2)", "LoRA / QLoRA\n(Questao 3)"]
    valores = [494.0, 2.16]
    barras = ax.barh(rotulos, valores, color=[COR_ANTES, COR_DEPOIS], height=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("Parametros treinaveis (milhoes, escala log)")
    ax.set_xlim(1, 1500)
    textos = ["494,0 M  (100%)", "2,16 M  (0,44%)"]
    for b, t in zip(barras, textos):
        ax.annotate(
            t,
            xy=(b.get_width(), b.get_y() + b.get_height() / 2),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=11,
        )
    ax.set_title("Custo de adaptacao: parametros atualizados no treino")
    fig.tight_layout()
    destino = os.path.join(AQUI, "q3_lora_params.png")
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("gerado:", destino)


if __name__ == "__main__":
    figura_q1()
    figura_q3_lora()
