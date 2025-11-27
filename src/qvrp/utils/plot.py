import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(samples, shots):
     # dict: bitstring -> count

    # 1. Mantém a ordem original das bitstrings
    items = list(samples.items())
    bitstrings = [bs for bs, cnt in items]
    counts = np.array([cnt for bs, cnt in items], dtype=float)

    percentages = counts / shots

    # 2. Descobre os índices das 10 com maior contagem (sem mudar a ordem original)
    top_k = 10
    # argsort crescente, depois inverte para decrescente e pega top_k
    top_indices = np.argsort(counts)[::-1][:top_k]

    # 3. Preparar eixo x na ordem original
    x = np.arange(len(bitstrings))

    plt.figure(figsize=(16, 6))

    # 4. Plotar TODAS as barras na ordem original
    plt.bar(x, percentages, color="lightgray")

    # 5. Destacar apenas as top_k (na posição original)
    for idx in top_indices:
        plt.bar(x[idx], percentages[idx], color="tab:blue")

    # 6. Rótulos de eixo X somente para as top_k
    xticks_positions = top_indices
    xticks_labels = [bitstrings[i] for i in top_indices]

    plt.xticks(xticks_positions, xticks_labels, rotation=45, ha="right")

    plt.ylabel("Percentage (%)")
    plt.xlabel("Bitstrings (only top 10 labeled)")
    plt.title("QAOA outcome histogram\nAll states plotted, only top 10 labeled")

    plt.tight_layout()
    plt.show()
