"""
CONTACT MEASUREMENT: AABB versus OBB versus exact distance

Criticism of the previous approach: axis-aligned bounding boxes (AABB)
overestimate contacts for oblique and long parts - i.e. for aluminium
extrusions, belts and struts. The measured gap could be an artefact of
this approximation.

Three stages:
  1. AABB   - previous approach, axis-aligned
  2. OBB    - oriented bounding box, rotates with the part
              (fixes the oblique-orientation error)
  3. exact  - BRepExtrema_DistShapeShape on a sample,
              to quantify the error rate of both approximations

For the complete exact evaluation of all pairs see exakter_graph.py.

Usage:
    python3 src/kontaktvergleich.py <step> <output.json> [tol] [sample]
"""
import json
import os
import random
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
from OCP.Bnd import Bnd_Box, Bnd_OBB
from OCP.BRepBndLib import BRepBndLib
from OCP.TopLoc import TopLoc_Location
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.gp import gp_Pnt, gp_Dir

def label_name(l):
    n = TDataStd_Name()
    return n.Get().ToExtString() if l.FindAttribute(TDataStd_Name.GetID_s(), n) else "?"

def lade(pfad):
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-CAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-CAF"), doc)
    r = STEPCAFControl_Reader()
    r.SetNameMode(True); r.SetColorMode(False); r.SetLayerMode(False)
    r.ReadFile(pfad); r.Transfer(doc)
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
        ob = Bnd_OBB()
        try:
            BRepBndLib.AddOBB_s(sh, ob, True, True, False)
        except Exception:
            ob = None
        teile.append({"name": neu[-1], "pfad": neu,
                      "bbox": [round(v, 3) for v in bx.Get()],
                      "obb": ob, "shape": sh})

    frei = TDF_LabelSequence(); st.GetFreeShapes(frei)
    for i in range(1, frei.Length() + 1):
        geh(frei.Value(i), TopLoc_Location(), [])
    return teile

if __name__ == "__main__":
    pfad, aus = sys.argv[1], sys.argv[2]
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    n_stich = int(sys.argv[4]) if len(sys.argv) > 4 else 250

    print(f"Reading {pfad} ...", flush=True)
    teile = lade(pfad)
    print(f" parts: {len(teile)}", flush=True)

    # ---------- stage 1: AABB ----------
    B = np.array([t["bbox"] for t in teile]); lo, hi = B[:, :3], B[:, 3:]
    aabb = []
    for i in range(len(teile)):
        j = np.arange(i + 1, len(teile))
        if not len(j):
            break
        ov = np.all((lo[i] - tol <= hi[j]) & (hi[i] + tol >= lo[j]), axis=1)
        aabb += [(i, int(x)) for x in j[ov]]
    print(f" AABB contacts: {len(aabb)}", flush=True)

    # ---------- stage 2: OBB ----------
    obb = []
    fehlend = 0
    for (i, j) in aabb:
        oi, oj = teile[i]["obb"], teile[j]["obb"]
        if oi is None or oj is None:
            fehlend += 1
            obb.append((i, j))   # keep when in doubt
            continue
        def erw(o, d):
            return Bnd_OBB(gp_Pnt(o.Center()), gp_Dir(o.XDirection()),
                           gp_Dir(o.YDirection()), gp_Dir(o.ZDirection()),
                           o.XHSize()+d, o.YHSize()+d, o.ZHSize()+d)
        a, b = erw(oi, tol/2), erw(oj, tol/2)
        if not a.IsOut(b):
            obb.append((i, j))
    print(f" OBB contacts : {len(obb)} "
          f"({100*len(obb)/max(len(aabb),1):.1f} % of AABB contacts, "
          f"{fehlend} without OBB)", flush=True)

    # ---------- stage 3: exact sample ----------
    random.seed(42)
    stich = random.sample(aabb, min(n_stich, len(aabb)))
    obb_set = set(obb)
    ergeb = []
    t0 = time.time()
    for k, (i, j) in enumerate(stich):
        if k % 50 == 0 and k:
            print(f" exact {k}/{len(stich)} "
                  f"({(time.time()-t0)/k:.2f} s/pair)", flush=True)
        try:
            d = BRepExtrema_DistShapeShape(teile[i]["shape"], teile[j]["shape"])
            if not d.IsDone():
                continue
            ergeb.append({"i": i, "j": j, "dist": round(d.Value(), 4),
                          "aabb": True, "obb": (i, j) in obb_set})
        except Exception:
            continue

    echt = [e for e in ergeb if e["dist"] <= tol]
    aabb_tp = len(echt)
    obb_tp = len([e for e in echt if e["obb"]])
    obb_fp = len([e for e in ergeb if e["obb"] and e["dist"] > tol])

    print(f"\n === sample {len(ergeb)} pairs ===", flush=True)
    print(f" actual contact (<= {tol} mm): {aabb_tp} "
          f"({100*aabb_tp/max(len(ergeb),1):.1f} %)")
    print(f" AABB hit rate (precision) : "
          f"{100*aabb_tp/max(len(ergeb),1):.1f} %")
    print(f" OBB hit rate (precision)  : "
          f"{100*obb_tp/max(obb_tp+obb_fp,1):.1f} %")
    print(f" OBB completeness (recall) : "
          f"{100*obb_tp/max(aabb_tp,1):.1f} %")

    json.dump({
        "teile": [{k: v for k, v in t.items() if k not in ("shape", "obb")}
                  for t in teile],
        "aabb": aabb, "obb": obb, "stichprobe": ergeb, "toleranz": tol,
    }, open(aus, "w"))
    print(f"\nSaved: {aus}")
