import sys
import os
import csv
import matplotlib.pyplot as plt
from collections import Counter

# adicionar projeto ao path
sys.path.append(os.path.join(os.path.dirname(__file__), "eavrp_contents"))

from encoder import QUBOEncoder
from solver import QAOASolver
from classical_solver import ClassicalVRPSolver
from grouping import GeoGrouper

# =============================
# Parâmetros fixos
# =============================
P = 2                 # camadas do QAOA
LAMBDA = 10.0         # penalidade λ
MU = 5.0              # penalidade μ

# =============================
# Definição da instância fixa
# (mantida como no código anterior)
# =============================
n_passengers = 3
n_vehicles = 3
n_depots = 1
n_recharge_points = 1 
seed = 42
diameter = 0.004

generator = Generator(
    n_passengers,
    n_vehicles,
    n_depots,
    n_recharge_points,
    seed=seed
)

VEHICLES_BASE, PASSENGERS, DEPOTS, RPS = generator.build()

grouper = GeoGrouper(delta=diameter)
groups = grouper.fit(PASSENGERS)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# Funções auxiliares
# =============================

def run_classical(encoder: QUBOEncoder):
    """Resolve com o solver clássico do projeto e retorna
    (best_cost, best_bits, feasible_solutions_list).

    feasible_solutions_list = [(bits, cost), ...] apenas para as viáveis.
    """
    solver = ClassicalVRPSolver(encoder)
    solver.solve()                             # usualmente retorna None e preenche solver.response
    best = solver.best()                        # dicionário com melhor solução viável

    best_bits = None
    best_cost = float("nan")
    if isinstance(best, dict):
        best_bits = best.get("bits")
        # se o dicionário já trouxer o custo, usa; senão calcula
        if "cost" in best and best["cost"] is not None:
            best_cost = float(best["cost"])
        elif best_bits is not None:
            best_cost = float(encoder.cost(best_bits))

    # Varre todas as amostras da resposta clássica para contar/exportar viáveis
    feasible = []
    resp = getattr(solver, "response", None)
    if resp is not None:
        # resp.variables define a ordem dos índices
        for rec in resp.record:
            sample_map = {v: rec.sample[j] for j, v in enumerate(resp.variables)}
            bits = [sample_map[i] for i in range(len(resp.variables))]
            if encoder.is_feasible(bits):
                feasible.append((bits, encoder.cost(bits)))

    return best_cost, best_bits, feasible


def run_quantum(encoder: QUBOEncoder):
    """Resolve com QAOA do projeto e retorna (best_cost, best_bits).
    Respeita a API do QAOASolver do projeto: solve() povoa atributos e/ou retorna namedtuple.
    """
    qsolver = QAOASolver(encoder, p=P, shots=2000)
    res = qsolver.solve()  # pode ser None ou um namedtuple com best_bits

    # tenta pegar os bits pelo atributo; se não, pelo retorno
    best_bits = getattr(qsolver, "best_bits", None)
    if best_bits is None and res is not None:
        best_bits = getattr(res, "best_bits", None)

    if best_bits is None:
        return float("nan"), None

    # custo limpo só se for viável; se não for, marcamos como NaN
    if encoder.is_feasible(best_bits):
        return float(encoder.cost(best_bits)), best_bits
    else:
        return float("nan"), best_bits


def decode_route(encoder: QUBOEncoder, sample):
    # Alguns encoders oferecem decode(); outros, interpret().
    if hasattr(encoder, "decode"):
        return encoder.decode(sample)
    elif hasattr(encoder, "interpret"):
        return encoder.interpret(sample)
    return {}

# =============================
# Impressão detalhada da solução
# =============================

def print_solution(title: str, encoder, sample, cost, feasible: bool):
    print(f"\n== {title} ==")
    if sample is None:
        print("  (nenhuma solução viável encontrada)")
        return
    bitstr = "".join(str(int(b)) for b in sample)
    print(f"Bits: {bitstr}")
    print(f"Feasible: {feasible} | Custo: {cost:.6f}")
    try:
        mapping = decode_route(encoder, sample)
    except Exception:
        mapping = {}
    print("Decodificado:")
    print(mapping)

# =============================
# Experimentos agrupado/sem agrupamento
# =============================

def experiment(grouping: bool):
    encoder = QUBOEncoder(
        vehicles=VEHICLES_BASE,
        groups=groups,
        stations=RPS,
        penalty_lambda=LAMBDA,
        penalty_mu=MU,
        depot=DEPOTS,
    )

    if encoder.num_qubits > 12:
        print(f"Instância excede limite de qubits ({encoder.num_qubits}) → ignorada.")
        return None

    classical_cost, classical_sample, feasible_solutions = run_classical(encoder)
    quantum_cost, quantum_sample = run_quantum(encoder)

    print("====================")
    print("Agrupamento" if grouping else "Sem Agrupamento")
    print(f"Qubits usados: {encoder.num_qubits}")

    # bitstrings
    try:
        classical_bits = "".join(str(int(b)) for b in (classical_sample or []))
    except Exception:
        classical_bits = str(classical_sample)
    try:
        quantum_bits = "".join(str(int(b)) for b in (quantum_sample or []))
    except Exception:
        quantum_bits = str(quantum_sample)

    # factibilidade do estado quântico
    quantum_feasible = (quantum_sample is not None) and encoder.is_feasible(quantum_sample)

    print(f"Clássico (ótimo): {classical_cost:.6f}")
    print(f"  Bits (A): {classical_bits}")
    print("  Rota decodificada (A):", decode_route(encoder, classical_sample) if classical_sample is not None else {})

    print(f"Quântico (QAOA): {quantum_cost:.6f}  | viável: {'sim' if quantum_feasible else 'não'}")
    print(f"  Bits (B): {quantum_bits}")
    print("  Rota decodificada (B):", decode_route(encoder, quantum_sample) if quantum_sample is not None else {})

    if classical_cost == classical_cost and quantum_cost == quantum_cost:  # ambos não-NaN
        delta = quantum_cost - classical_cost
        print(f"Δ (Q − C): {delta:+.6f}")

    # Contagem de soluções clássicas viáveis (limites do QAOA)
    print(f"Nº soluções clássicas viáveis: {len(feasible_solutions)}")
    if len(feasible_solutions) > 0:
        # mostra até 3 exemplos de soluções viáveis clássicas
        print("Algumas soluções clássicas viáveis (bits, custo):")
        for i, (samp, cst) in enumerate(feasible_solutions[:3], start=1):
            try:
                bits_str = "".join(str(int(b)) for b in samp)
            except Exception:
                bits_str = str(samp)
            print(f"  {i:>2}. {bits_str}  →  {cst:.6f}")

    # salvar todas as soluções viáveis em CSV (compatível com pedido anterior)
    scen_slug = "agrupado" if grouping else "sem_agrupamento"

    csv_path1 = os.path.join(OUTPUT_DIR, f"solucoes_classicas_{scen_slug}.csv")
    with open(csv_path1, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "index", "bits", "cost"])
        for i, (samp, cst) in enumerate(feasible_solutions, start=1):
            try:
                bits_str = "".join(str(int(b)) for b in samp)
            except Exception:
                bits_str = str(samp)
            w.writerow([scen_slug, i, bits_str, f"{cst:.6f}"])
    print(f"CSV salvo: {csv_path1}")

    csv_path2 = os.path.join(OUTPUT_DIR, f"comparacao_{scen_slug}.csv")
    with open(csv_path2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "method", "bits", "cost", "feasible"])
        w.writerow([scen_slug, "classical", classical_bits, f"{classical_cost:.6f}", True])
        w.writerow([scen_slug, "quantum",   quantum_bits,   f"{quantum_cost:.6f}", quantum_feasible])
    print(f"CSV salvo: {csv_path2}")

    # gráfico comparativo
    plt.bar(["Clássico", "Quântico"], [classical_cost, quantum_cost])
    plt.ylabel("Custo Limpo")
    plt.title("Agrupado" if grouping else "Sem Agrupamento")
    outpath = os.path.join(OUTPUT_DIR, "agrupado_barras.png" if grouping else "sem_agrupamento_barras.png")
    plt.savefig(outpath)
    plt.close()


if __name__ == "__main__":
    experiment(grouping=True)
    experiment(grouping=False)
