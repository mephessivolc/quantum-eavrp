from pathlib import Path

# exemplo_minimo.py
from shrink import Shrink
from graph import Graph
from grouping import GeoGrouper

from feasible_instance_generator import InstanceGenerator

OUTDIR = Path("outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

n_passengers=100
n_vehicles=80
n_depots=4
n_recharges=75

filename_graph = OUTDIR / "teste_shrink_graph.png"
# 1) Gera uma instância pequena sob limite de qubits
diameter= 0.004
gen = InstanceGenerator(n_passengers=n_passengers,
                         n_vehicles=n_vehicles,
                         n_depots=n_depots, 
                         n_recharges=n_recharges, 
                         seed=42
                        )

v, p, d, rp = gen.build()

# agrupa cada passageiro sozinho (teste viabilidade)
grouper = GeoGrouper(delta=diameter)
groups = grouper.fit(p)


graph = Graph(p, groups, rp, d, filename_graph).build_graph()

# 3) Instancia o pipeline Shrink → Executing → Refining
shr = Shrink(
    passengers=p,
    depots=d,
    recharge_points=rp,
    G=graph,
    compression_ratio=0.6,   # alvo de “compressão” via agrupamento
)

# 4) Coarsening: agrupamento geográfico → “supernós”
groups = shr.coarsening()
print(f"#grupos: {len(groups)}")

# 5) Executing: resolve no grafo comprimido (clássico ou quântico)
# result = shr.executing(method="classical")   # use "quantum" para QAOA
# print("Rotas (coarse):", shr.routes_coarse)
# print("Resultado bruto:", result)

# 6) Refining: projeta para grafo original e recalcula caminhos
refined = shr.refining()
print("Rotas refinadas:", refined)
