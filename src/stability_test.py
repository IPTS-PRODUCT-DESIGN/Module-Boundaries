"""
DECISION TEST: Is the low agreement between methods a property of the
assembly - or an artefact of unstable algorithms?

Four checks on the real Voron data:

T1 SEED STABILITY
   Same method, same parameters, only a different random seed.
   If Louvain is unstable, that explains the divergence without any
   statement about module boundaries.

T2 RESOLUTION STABILITY
   Same method across the resolution parameter.

T3 STRUCTURE-PRESERVING NULL MODEL (configuration model)
   Graph with identical degree sequence but randomly rewired.
   How strongly do two runs on it agree? That is the lower bound
   that follows from the degree distribution alone.

T4 METHODS WITH THE SAME OBJECTIVE FUNCTION
   Louvain and Greedy both maximise modularity. If they diverge,
   it is not because of different objective functions.
"""
import json
import os
import sys
import numpy as np
import networkx as nx
from itertools import combinations
from sklearn.metrics import adjusted_rand_score as ari

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poc_real import klassifiziere, kontaktgraph

DATA = os.environ.get("MODULGRENZEN_DATA", "data")
GERAETE = [
    ("Voron 2.4r2",      os.path.join(DATA, "voron_step",      "teile.json")),
    ("Voron Trident",    os.path.join(DATA, "trident_step",    "teile.json")),
    ("Voron Switchwire", os.path.join(DATA, "switchwire_step", "teile.json")),
]
N_SEEDS = 12
RES = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

def lade(pfad):
    teile = json.load(open(pfad))
    for t in teile:
        t["art"] = klassifiziere(t["name"])
        b = t["bbox"]
        t["volumen"] = max((b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2]), 1e-9)
    return teile

def labels(comms, knoten):
    m = {n: i for i, c in enumerate(comms) for n in c}
    return [m[n] for n in knoten]

def paarweise(parts):
    return [ari(a, b) for a, b in combinations(parts, 2)]

if __name__ == "__main__":
    print("=" * 78)
    print("DECISION TEST: method instability or property of the assembly?")
    print("=" * 78)

    ergebnis = {}
    for name, pfad in GERAETE:
        teile = lade(pfad)
        G = kontaktgraph(teile)
        H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        kn = list(H.nodes())
        truth = [H.nodes[n]["modul"] for n in kn]

        print(f"\n{'='*78}\n{name} ({H.number_of_nodes()} nodes, "
              f"{H.number_of_edges()} edges)\n{'='*78}")

        # ---------- T1 seed stability ----------
        parts = [labels(nx.community.louvain_communities(
            H, weight="weight", resolution=1.0, seed=s), kn)
            for s in range(N_SEEDS)]
        t1 = paarweise(parts)
        n_cl = [len(set(p)) for p in parts]
        gegen_gt = [ari(truth, p) for p in parts]
        print(f"\nT1 SEED STABILITY (Louvain, res=1.0, {N_SEEDS} seeds)")
        print(f"  Louvain against itself   : ARI {np.mean(t1):.3f} "
              f"(min {min(t1):.3f}, max {max(t1):.3f})")
        print(f"  cluster count            : {min(n_cl)}\u2013{max(n_cl)}")
        print(f"  against CAD structure    : ARI {np.mean(gegen_gt):.3f} "
              f"(\u00b1{np.std(gegen_gt):.3f})")

        # ---------- T2 resolution stability ----------
        pr = [labels(nx.community.louvain_communities(
            H, weight="weight", resolution=r, seed=42), kn) for r in RES]
        t2 = paarweise(pr)
        print(f"\nT2 RESOLUTION STABILITY (res {RES[0]}\u2013{RES[-1]})")
        print(f"  Louvain against itself   : ARI {np.mean(t2):.3f} "
              f"(min {min(t2):.3f}, max {max(t2):.3f})")

        # ---------- T3 configuration model ----------
        grade = [d for _, d in H.degree()]
        null_paare = []
        for rep in range(4):
            CM = nx.configuration_model(grade, seed=100 + rep)
            CM = nx.Graph(CM)
            CM.remove_edges_from(nx.selfloop_edges(CM))
            kn_cm = list(CM.nodes())
            ps = [labels(nx.community.louvain_communities(
                CM, resolution=1.0, seed=s), kn_cm) for s in range(4)]
            null_paare += paarweise(ps)
        print(f"\nT3 NULL MODEL (configuration model, same degree sequence)")
        print(f"  Louvain against itself   : ARI {np.mean(null_paare):.3f} "
              f"(min {min(null_paare):.3f}, max {max(null_paare):.3f})")
        print(f"  \u2192 lower bound from the degree distribution alone")

        # ---------- T4 same objective function ----------
        lou = labels(nx.community.louvain_communities(
            H, weight="weight", resolution=1.0, seed=42), kn)
        gre = labels(nx.community.greedy_modularity_communities(
            H, weight="weight"), kn)
        t4 = ari(lou, gre)
        q_lou = nx.community.modularity(
            H, nx.community.louvain_communities(H, weight="weight",
                                                resolution=1.0, seed=42),
            weight="weight")
        q_gre = nx.community.modularity(
            H, nx.community.greedy_modularity_communities(H, weight="weight"),
            weight="weight")
        print(f"\nT4 SAME OBJECTIVE FUNCTION (both maximise modularity)")
        print(f"  Louvain vs Greedy        : ARI {t4:.3f}")
        print(f"  modularity Q             : Louvain {q_lou:.3f} | "
              f"Greedy {q_gre:.3f}")

        ergebnis[name] = dict(
            t1=float(np.mean(t1)), t1_min=float(min(t1)),
            t2=float(np.mean(t2)), t3=float(np.mean(null_paare)),
            t4=float(t4), gt=float(np.mean(gegen_gt)),
            q_lou=float(q_lou), q_gre=float(q_gre))

    # ---------------- Evaluation ----------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'Machine':20s}{'T1 seed':>10s}{'T2 res.':>10s}"
          f"{'T3 null':>10s}{'T4 same obj':>12s}{'vs CAD':>12s}")
    print("-" * 78)
    for n, d in ergebnis.items():
        print(f"{n:20s}{d['t1']:10.3f}{d['t2']:10.3f}"
              f"{d['t3']:10.3f}{d['t4']:12.3f}{d['gt']:12.3f}")

    t1m = np.mean([d["t1"] for d in ergebnis.values()])
    t3m = np.mean([d["t3"] for d in ergebnis.values()])
    t4m = np.mean([d["t4"] for d in ergebnis.values()])
    gtm = np.mean([d["gt"] for d in ergebnis.values()])

    print("\n" + "=" * 78)
    print("FINDING")
    print("=" * 78)
    if t1m > 0.8:
        print(f"T1 Louvain is STABLE across seeds (ARI {t1m:.3f}).")
        print("   The divergence between methods is therefore NOT explained")
        print("   by random instability of a single method.")
    elif t1m > 0.6:
        print(f"T1 Louvain is MODERATELY stable across seeds (ARI {t1m:.3f}).")
        print("   Part of the divergence is due to method instability.")
    else:
        print(f"T1 Louvain is UNSTABLE across seeds (ARI {t1m:.3f}).")
        print("   The non-uniqueness thesis is then untenable - the")
        print("   divergence is an artefact of the algorithm.")

    print(f"\nT3 On a structureless graph of the same degree sequence,")
    print(f"   Louvain against itself reaches ARI {t3m:.3f}.")
    if t1m - t3m > 0.15:
        print(f"   The real graph lies clearly above it (+{t1m-t3m:.3f})")
        print("   \u2192 the stability comes from real structure, not from the")
        print("   degree distribution.")
    else:
        print("   The difference is small \u2192 the stability could follow from")
        print("   the degree distribution alone.")

    print(f"\nT4 Louvain and Greedy maximise the same objective function and")
    print(f"   still reach only ARI {t4m:.3f}.")
    if t4m < 0.6:
        print("   \u2192 The divergence is NOT explained by different objective")
        print("   functions alone (degeneracy of modularity).")

    print(f"\nCONTEXT OF THE MAIN FINDING")
    print(f"   method against CAD structure : {gtm:.3f}")
    print(f"   method against itself        : {t1m:.3f}")
    # Note: the agreement of the methods among each other
    # (mean ~0.316 for Voron 2.4r2) is computed in triangulation.py.
    print(f"   \u2192 clear ranking: identical method > other methods")
    print(f"   > designer structure")

    json.dump(ergebnis, open("stabilitaet.json", "w"), indent=1)
    print("\nSaved: stabilitaet.json")
