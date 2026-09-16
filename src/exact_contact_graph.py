"""
COMPLETE EXACT CONTACT GRAPH

Computes the true minimum distance for ALL bounding-box candidate pairs
and stores a cleaned contact graph.

Difference from kontaktvergleich.py: there only a sample, here everything.
Runtime for Jubilee about 100 minutes (2096 pairs at ~3 s).

Progress is checkpointed every 100 pairs. If the run aborts, it can be
resumed with the same command - already computed pairs are skipped.

Usage:
    python3 src/exakter_graph.py data/raw/jubilee.STEP ergebnisse/jubilee_exakt.json 1.0
"""
import json
import os
import sys
import time
import numpy as np
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopLoc import TopLoc_Location
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

SPEICHER_INTERVALL = 100

def label_name(l):
    n = TDataStd_Name()
    return n.Get().ToExtString() if l.FindAttribute(TDataStd_Name.GetID_s(), n) else "?"

def lade(pfad):
    """read STEP, keep solids in memory"""
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-CAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-CAF"), doc)
    r = STEPCAFControl_Reader()
    r.SetNameMode(True); r.SetColorMode(False); r.SetLayerMode(False)
    if not r.ReadFile(pfad):
        raise RuntimeError(f"Could not read {pfad}")
    r.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    teile = []

    def geh(lab, loc, pf):
        nm = label_name(lab); neu = pf + [nm]
        if st.IsAssembly_s(lab):
            k = TDF_LabelSequence(); st.GetComponents_s(lab, k)
            for i in range(1, k.Length() + 1):
                geh(k.Value(i), loc, neu)
            return
        if st.IsReference_s(lab):
            ref = TDF_Label(); st.GetReferredShape_s(lab, ref)
            rl = st.GetLocation_s(lab); rn = label_name(ref)
            geh(ref, loc.Multiplied(rl), pf + [nm if nm not in ("?", "") else rn])
            return
        sh = st.GetShape_s(lab)
        if sh.IsNull():
            return
        sh = sh.Moved(loc)
        bx = Bnd_Box()
        try:
            BRepBndLib.Add_s(sh, bx, False)
            if bx.IsVoid():
                return
        except Exception:
            return
        teile.append({"name": neu[-1], "pfad": neu,
                      "bbox": [round(v, 3) for v in bx.Get()], "shape": sh})

    frei = TDF_LabelSequence(); st.GetFreeShapes(frei)
    for i in range(1, frei.Length() + 1):
        geh(frei.Value(i), TopLoc_Location(), [])
    return teile

def bb_kandidaten(teile, tol):
    B = np.array([t["bbox"] for t in teile]); lo, hi = B[:, :3], B[:, 3:]
    paare = []
    for i in range(len(teile)):
        j = np.arange(i + 1, len(teile))
        if not len(j):
            break
        ov = np.all((lo[i] - tol <= hi[j]) & (hi[i] + tol >= lo[j]), axis=1)
        paare += [(i, int(x)) for x in j[ov]]
    return paare

def speichern(aus, teile, paare, abstaende, tol):
    json.dump({
        "teile": [{k: v for k, v in t.items() if k != "shape"} for t in teile],
        "aabb": paare,
        "abstaende": abstaende,   # "i,j" -> distance
        "toleranz": tol,
        "fertig": len(abstaende) == len(paare),
    }, open(aus, "w"))

if __name__ == "__main__":
    pfad = sys.argv[1]
    aus = sys.argv[2]
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    os.makedirs(os.path.dirname(os.path.abspath(aus)), exist_ok=True)

    print(f"Reading {pfad} ...", flush=True)
    teile = lade(pfad)
    print(f" parts: {len(teile)}", flush=True)

    paare = bb_kandidaten(teile, tol)
    print(f" candidate pairs: {len(paare)}", flush=True)

    # ---- load existing progress ----
    abstaende = {}
    if os.path.exists(aus):
        try:
            alt = json.load(open(aus))
            abstaende = alt.get("abstaende", {})
            print(f" resuming: {len(abstaende)} pairs already computed",
                  flush=True)
        except Exception:
            print(" existing file unreadable, starting over", flush=True)

    offen = [(i, j) for (i, j) in paare if f"{i},{j}" not in abstaende]
    print(f" still to compute: {len(offen)}\n", flush=True)

    t0 = time.time()
    for k, (i, j) in enumerate(offen):
        try:
            d = BRepExtrema_DistShapeShape(teile[i]["shape"], teile[j]["shape"])
            abstaende[f"{i},{j}"] = round(d.Value(), 4) if d.IsDone() else -1.0
        except Exception:
            abstaende[f"{i},{j}"] = -1.0

        if (k + 1) % SPEICHER_INTERVALL == 0:
            v = (time.time() - t0) / (k + 1)
            rest = v * (len(offen) - k - 1) / 60
            speichern(aus, teile, paare, abstaende, tol)
            print(f" {k+1}/{len(offen)} ({v:.2f} s/pair, "
                  f"{rest:.0f} min left) [saved]", flush=True)

    speichern(aus, teile, paare, abstaende, tol)

    # ---- evaluation ----
    gueltig = {k: v for k, v in abstaende.items() if v >= 0}
    echt = {k: v for k, v in gueltig.items() if v <= tol}
    d = np.array(list(gueltig.values()))

    print("\n" + "=" * 62)
    print("RESULT")
    print("=" * 62)
    print(f" candidate pairs (AABB) : {len(paare)}")
    print(f" evaluable              : {len(gueltig)}")
    print(f" real contacts          : {len(echt)} "
          f"({100*len(echt)/max(len(gueltig),1):.1f} %)")
    print(f" pseudo contacts        : {len(gueltig)-len(echt)}")
    print()
    print(" distance distribution:")
    for lo, hi, lab in [(0, 0.001, "touch (0 mm)"), (0.001, 0.5, "0-0.5 mm"),
                        (0.5, 1.0, "0.5-1 mm"), (1.0, 5, "1-5 mm"),
                        (5, 20, "5-20 mm"), (20, 1e9, "over 20 mm")]:
        n = int(((d >= lo) & (d < hi)).sum())
        print(f"   {lab:20s} {n:5d} ({100*n/max(len(d),1):5.1f} %)")

    kanten = [(int(k.split(",")[0]), int(k.split(",")[1]), v)
              for k, v in echt.items()]
    json.dump({
        "teile": [{kk: vv for kk, vv in t.items() if kk != "shape"} for t in teile],
        "kanten_exakt": kanten,
        "toleranz": tol,
    }, open(aus.replace(".json", "_graph.json"), "w"))

    print(f"\n raw data : {aus}")
    print(f" graph    : {aus.replace('.json', '_graph.json')}")
    print("\n Next step: repeat the evaluation with this graph")
    print(" (gewichtung.py --exakt) and compare the gap against the")
    print(" bounding-box variant.")
