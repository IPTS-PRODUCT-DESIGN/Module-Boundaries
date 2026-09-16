"""
PROOF OF CONCEPT WITH REAL DATA
Voron 2.4r2 - 1428 parts from the official STEP assembly

Not a synthetic example. Both data sources are external:
GEOMETRY      official STEP assembly (VoronDesign/Voron-2, GitHub)
GROUND TRUTH  the module structure recorded by the designer

CENTRAL QUESTION
Does a geometry-based module partition recover the module structure
that real engineers created?
"""
import json
import re
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi

TOL = 1.0   # mm tolerance for contact between bounding boxes
SEED = 42

# ---------------------------------------------------------------
# Standard-part recognition from real part names (name patterns)
# ---------------------------------------------------------------
NORMTEILE = [
    (re.compile(r"\bF?6\d{2}\s*(ZZ|2RS)?\b|\bMR\d+\b|\bbearing\b", re.I),
     "Rolling bearing"),
    (re.compile(r"\bGT2\b.*\b(belt|loop)\b|\bbelt\b", re.I),
     "Timing belt"),
    (re.compile(r"\bGT2\s*\d+T\b|\bpulley\b|\bidler\b", re.I),
     "Pulley"),
    (re.compile(r"\bNEMA\s*\d+\b|\bstepper\b|\bmotor\b", re.I),
     "Stepper motor"),
    (re.compile(r"\bM\d+(\.\d+)?x\d+\b|\bBHCS\b|\bSHCS\b|\bFHCS\b|"
                r"\bnut\b|\bwasher\b|\bshim\b|\bheatset\b|\binsert\b", re.I),
     "Fastener"),
    (re.compile(r"\blead\s*screw\b|\bTR8\b|\bT8\b|\bball\s*screw\b", re.I),
     "Lead screw"),
    (re.compile(r"\bfan\b|\bblower\b|\b\d{4}\s*fan\b", re.I),
     "Fan"),
    (re.compile(r"\bnozzle\b|\bhotend\b|\bheater\b|\bthermistor\b", re.I),
     "Hotend wear part"),
    (re.compile(r"\bextrusion\b|\b20\d0\b.*\bextrusion\b", re.I),
     "Aluminium extrusion"),
    (re.compile(r"\bpanel\b|\bacrylic\b|\bpolycarbonate\b|\bglass\b", re.I),
     "Panel"),
    (re.compile(r"\bPCB\b|\bboard\b|\bRaspberry\b|\bOctopus\b|\bMCU\b|"
                r"\bpower\s*supply\b|\bPSU\b|\bSSR\b", re.I),
     "Electronics"),
    (re.compile(r"\brail\b|\bMGN\d+\b|\blinear\b", re.I),
     "Linear guide"),
]

def klassifiziere(name):
    for rx, art in NORMTEILE:
        if rx.search(name):
            return art
    return "Design part"

# ---------------------------------------------------------------

def lade():
    teile = json.load(open("teile.json"))
    for t in teile:
        t["art"] = klassifiziere(t["name"])
        b = t["bbox"]
        t["zentrum"] = [(b[0]+b[3])/2, (b[1]+b[4])/2, (b[2]+b[5])/2]
        t["volumen"] = max((b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2]), 1e-9)
    return teile

def kontaktgraph(teile, tol=TOL):
    """Edges between parts whose bounding boxes touch.
    Purely geometric - independent of the assembly hierarchy."""
    B = np.array([t["bbox"] for t in teile])
    n = len(teile)
    G = nx.Graph()
    for i, t in enumerate(teile):
        G.add_node(i, name=t["name"], modul=t["modul"],
                   art=t["art"], volumen=t["volumen"])

    lo, hi = B[:, :3], B[:, 3:]
    for i in range(n):
        # vectorised overlap test against all following parts
        j = np.arange(i + 1, n)
        if len(j) == 0:
            break
        ueberlappt = np.all(
            (lo[i] - tol <= hi[j]) & (hi[i] + tol >= lo[j]), axis=1)
        for jj in j[ueberlappt]:
            # weight: smaller parts bind more strongly (screws connect)
            v = min(teile[i]["volumen"], teile[int(jj)]["volumen"])
            G.add_edge(i, int(jj), weight=1.0 / (1.0 + np.log10(max(v, 1))))
    return G

# ---------------------------------------------------------------

if __name__ == "__main__":
    teile = lade()
    print("=" * 74)
    print("PROOF OF CONCEPT WITH REAL DATA - Voron 2.4r2")
    print("=" * 74)
    print("Source: official STEP assembly, VoronDesign/Voron-2 (GitHub)")
    print(f"Part instances: {len(teile)}")

    print("\n--- Part kinds (recognised from real names) ---")
    for art, n in Counter(t["art"] for t in teile).most_common():
        print(f"  {art:26s} {n:5d}")

    print("\n--- Designer module structure (ground truth) ---")
    for m, n in Counter(t["modul"] for t in teile).most_common():
        print(f"  {m:32s} {n:5d}")

    print("\nBuilding contact graph from geometry ...")
    G = kontaktgraph(teile)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} contacts")
    komp = list(nx.connected_components(G))
    print(f"  Connected components: {len(komp)} "
          f"(largest: {max(len(c) for c in komp)})")

    H = G.subgraph(max(komp, key=len)).copy()
    knoten = list(H.nodes())
    truth = [H.nodes[n]["modul"] for n in knoten]

    print("\n" + "=" * 74)
    print("EXPERIMENT: Does geometry recover the engineers' modules?")
    print("=" * 74)

    ergebnisse = []
    for res in [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0]:
        comms = nx.community.louvain_communities(
            H, weight="weight", resolution=res, seed=SEED)
        lab = {n: i for i, c in enumerate(comms) for n in c}
        pred = [lab[n] for n in knoten]
        a, nm = ari(truth, pred), nmi(truth, pred)
        ergebnisse.append((res, len(comms), a, nm))
        print(f"  resolution={res:<4} -> {len(comms):3d} clusters | "
              f"ARI={a:+.3f} | NMI={nm:.3f}")

    aris = [r[2] for r in ergebnisse]
    print(f"\n ARI range across parameters: {max(aris)-min(aris):.3f}")
    best = max(ergebnisse, key=lambda r: r[2])
    print(f" Best result: ARI={best[2]:+.3f} at resolution={best[0]}")

    # ---------- Detailed analysis of the best partition ----------
    comms = nx.community.louvain_communities(
        H, weight="weight", resolution=best[0], seed=SEED)
    lab = {n: i for i, c in enumerate(comms) for n in c}

    print("\n" + "=" * 74)
    print("WHERE DOES GEOMETRY DEVIATE FROM THE ENGINEERS?")
    print("=" * 74)
    verteilung = defaultdict(Counter)
    for n in knoten:
        verteilung[H.nodes[n]["modul"]][lab[n]] += 1
    for modul, c in sorted(verteilung.items(),
                           key=lambda x: -sum(x[1].values())):
        ges = sum(c.values())
        groesster = c.most_common(1)[0]
        rein = groesster[1] / ges
        print(f"  {modul:30s} {ges:4d} parts -> {len(c):2d} clusters, "
              f"largest holds {rein*100:4.1f} %")

    json.dump({"ergebnisse": ergebnisse}, open("poc_real_ergebnisse.json", "w"))
    print("\nSaved: poc_real_ergebnisse.json")
