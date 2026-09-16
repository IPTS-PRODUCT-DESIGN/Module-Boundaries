"""
WEIGHTING SENSITIVITY (Table 3 in the CIRP paper)

Repeats the core measurement under four edge weightings, to check
whether the finding depends on the chosen weighting scheme:

  unweighted   all edges equal
  invlogvol    1 / (1 + log10 max(min(Vi, Vj), 1)) - used in the paper
  overlap      cube root of the bounding-box overlap volume
               (approximation of the contact size)
  invvol       1 / (1 + cube root of the volume)

For each scheme it reports the self-consistency (Louvain against itself
over 12 random seeds), the agreement with the CAD structure and their
difference.

Usage:
    python3 src/gewichtung.py <teile.json> [module level]
    python3 src/gewichtung.py <jubilee_exakt_graph.json> 2 --exakt
"""
import json
import sys
from itertools import combinations

import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score as ari

TOL = 1.0
SEEDS = 12
RESOLUTION = 1.0

SCHEMATA = [("unweighted", "unweighted"),
            ("invlogvol", "inverse log volume"),
            ("overlap", "contact extent proxy"),
            ("invvol", "inverse cube-root volume")]

def volumen(t):
    b = t["bbox"]
    return max((b[3]-b[0]) * (b[4]-b[1]) * (b[5]-b[2]), 1e-9)

def gewicht(modus, ti, tj, lo=None, hi=None, i=None, j=None):
    if modus == "unweighted":
        return 1.0
    if modus == "invlogvol":
        v = min(volumen(ti), volumen(tj))
        return 1.0 / (1.0 + np.log10(max(v, 1)))
    if modus == "invvol":
        v = min(volumen(ti), volumen(tj))
        return 1.0 / (1.0 + v ** (1/3))
    if modus == "overlap":
        d = np.minimum(hi[i], hi[j]) - np.maximum(lo[i], lo[j])
        return float(np.prod(np.clip(d, 0.01, None))) ** (1/3)
    raise ValueError(modus)

def graph_aabb(teile, modus):
    """contact graph via bounding-box touch"""
    B = np.array([t["bbox"] for t in teile])
    lo, hi = B[:, :3], B[:, 3:]
    G = nx.Graph()
    G.add_nodes_from(range(len(teile)))
    for i in range(len(teile)):
        j = np.arange(i + 1, len(teile))
        if not len(j):
            break
        ov = np.all((lo[i] - TOL <= hi[j]) & (hi[i] + TOL >= lo[j]), axis=1)
        for jj in j[ov]:
            jj = int(jj)
            G.add_edge(i, jj,
                       weight=gewicht(modus, teile[i], teile[jj], lo, hi, i, jj))
    return G

def graph_exakt(teile, kanten, modus):
    """contact graph from precomputed exact surface distances"""
    B = np.array([t["bbox"] for t in teile])
    lo, hi = B[:, :3], B[:, 3:]
    G = nx.Graph()
    G.add_nodes_from(range(len(teile)))
    for i, j, _ in kanten:
        G.add_edge(i, j, weight=gewicht(modus, teile[i], teile[j], lo, hi, i, j))
    return G

def labels(comms, knoten):
    m = {n: k for k, c in enumerate(comms) for n in c}
    return [m[n] for n in knoten]

def messe(G, teile, ebene):
    H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    kn = list(H.nodes())
    truth = [teile[n]["pfad"][ebene] if len(teile[n]["pfad"]) > ebene else "ROOT"
             for n in kn]
    parts = [labels(nx.community.louvain_communities(
        H, weight="weight", resolution=RESOLUTION, seed=s), kn)
        for s in range(SEEDS)]
    selbst = np.mean([ari(a, b) for a, b in combinations(parts, 2)])
    gegen = np.mean([ari(truth, p) for p in parts])
    return selbst, gegen

if __name__ == "__main__":
    pfad = sys.argv[1]
    ebene = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1
    exakt = "--exakt" in sys.argv

    daten = json.load(open(pfad))
    if exakt:
        teile, kanten = daten["teile"], daten["kanten_exakt"]
    else:
        teile, kanten = daten, None

    print(f"{pfad} ({len(teile)} parts, module level {ebene},"
          f" {'exact contacts' if exakt else 'bounding box'})\n")
    print(f"{'Weighting':26s}{'self':>9s}{'vs. designer':>14s}{'gap':>9s}")
    print("-" * 58)
    for modus, label in SCHEMATA:
        G = graph_exakt(teile, kanten, modus) if exakt else graph_aabb(teile, modus)
        s, g = messe(G, teile, ebene)
        print(f"{label:26s}{s:9.3f}{g:14.3f}{s-g:9.3f}")
