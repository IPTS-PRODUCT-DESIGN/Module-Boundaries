"""
Extracts from the real Voron STEP assembly:
- every part instance with its name
- its world-coordinate bounding box
- the assembly membership recorded by the designer (ground truth)

Output: teile.json -> basis for the independent contact graph
"""
import json
import sys
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

def label_name(label):
    name = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), name):
        return name.Get().ToExtString()
    return "?"

def main(pfad):
    print(f"Reading {pfad} with OpenCASCADE ...")
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-CAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-CAF"), doc)

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(False)
    reader.SetLayerMode(False)
    if not reader.ReadFile(pfad):
        print("Reading failed"); return
    print(" File read, transferring ...")
    reader.Transfer(doc)
    print(" Transfer complete")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    teile = []

    def durchlaufe(label, loc, pfad_namen):
        """recurse through the assembly tree"""
        name = label_name(label)
        neu_pfad = pfad_namen + [name]

        if shape_tool.IsAssembly_s(label):
            kinder = TDF_LabelSequence()
            shape_tool.GetComponents_s(label, kinder)
            for i in range(1, kinder.Length() + 1):
                # placement is applied in the reference branch, not here
                durchlaufe(kinder.Value(i), loc, neu_pfad)
            return

        if shape_tool.IsReference_s(label):
            ref = TDF_Label()
            shape_tool.GetReferredShape_s(label, ref)
            rloc = shape_tool.GetLocation_s(label)
            refname = label_name(ref)
            nm = name if name not in ("?", "") else refname
            durchlaufe(ref, loc.Multiplied(rloc), pfad_namen + [nm])
            return

        # single part
        shape = shape_tool.GetShape_s(label)
        if shape.IsNull():
            return
        shape = shape.Moved(loc)
        box = Bnd_Box()
        try:
            BRepBndLib.Add_s(shape, box, False)
            if box.IsVoid():
                return
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        except Exception:
            return

        teile.append({
            "name": neu_pfad[-1],
            "pfad": neu_pfad,
            "modul": neu_pfad[1] if len(neu_pfad) > 1 else "ROOT",
            "bbox": [round(v, 2) for v in (xmin, ymin, zmin, xmax, ymax, zmax)],
        })

    frei = TDF_LabelSequence()
    shape_tool.GetFreeShapes(frei)
    print(f" Root objects: {frei.Length()}")
    for i in range(1, frei.Length() + 1):
        durchlaufe(frei.Value(i), TopLoc_Location(), [])

    print(f" Part instances with geometry: {len(teile)}")
    from collections import Counter
    c = Counter(t["modul"] for t in teile)
    print("\n Designer's real module structure:")
    for m, n in c.most_common():
        print(f"  {m:32s} {n:4d} parts")

    with open("teile.json", "w") as f:
        json.dump(teile, f)
    print("\nSaved: teile.json")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Voron_2.4r2_Assembly.step")
