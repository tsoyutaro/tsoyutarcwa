"""Extract independent vector-path reference data from Vallius/Honkanen (2002).

Reads the PDF drawing operators, never computed solver data. pdfplumber exposes
the transformed path vertices and dash styles; pypdf audits original operator
counts. Coordinates are calibrated only against the printed axes/tick labels.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
import pdfplumber
from pypdf import PdfReader
from pypdf.generic import ContentStream

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pdf", type=Path, required=True, help="Path to the original Vallius/Honkanen PDF")
parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "reference")
args = parser.parse_args()
SOURCE = args.pdf
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)

# Tick positions are PDF points with origin at upper left, measured directly
# from the original vector horizontal tick segments. y values are printed ticks.
PANELS = [
    dict(figure=6, panel="a", page=8, method="fmm", modes=5,
         box=[246.6213724,201.5331676,379.6714936,261.7034331],
         wavelengths=[.95,1.05], ypoints=[248.3817235,233.2712784,218.1608333,203.0503882], yvalues=[.20,.22,.24,.26],
         solid=[12], dashed=[63], diamond_modes=10, dashed_modes=240),
    dict(figure=6, panel="b", page=8, method="asr", modes=5,
         box=[246.6213724,283.9641590,379.6714936,344.1017961],
         wavelengths=[.95,1.05], ypoints=[334.9273892,318.9068448,302.9189288,286.9321781], yvalues=[.20,.22,.24,.26],
         solid=[64], dashed=[115], diamond_modes=10, dashed_modes=240),
    dict(figure=7, panel="a", page=8, method="fmm", modes=12,
         box=[246.6213724,432.3790976,379.6714936,492.5843221],
         wavelengths=[.95,1.05], ypoints=[481.6899324,470.7617490,459.8335656,448.8727538,437.9445704], yvalues=[.1,.2,.3,.4,.5],
         solid=[117], dashed=[168,169], diamond_modes=24, dashed_modes=240),
    dict(figure=7, panel="b", page=8, method="asr", modes=12,
         box=[246.6213724,514.8438827,379.6714936,574.9815198],
         wavelengths=[.95,1.05], ypoints=[564.0871301,553.1601120,542.2319286,531.3037452,520.3755618], yvalues=[.1,.2,.3,.4,.5],
         solid=[170], dashed=[221], diamond_modes=24, dashed_modes=240),
    dict(figure=9, panel="a", page=9, method="fmm", modes=19,
         box=[226.2946996,322.15104,403.7206042,402.3912674],
         wavelengths=[1.16,1.18], ypoints=[402.3912674,322.15104], yvalues=[0.,1.],
         solid=[5], dashed=[6], diamond_modes=38, dashed_modes=38),
    dict(figure=9, panel="b", page=9, method="asr", modes=19,
         box=[226.2946996,432.0761196,403.7206042,512.2709003],
         wavelengths=[1.16,1.18], ypoints=[512.2709003,432.0761196], yvalues=[0.,1.],
         solid=[57], dashed=[58], diamond_modes=38, dashed_modes=38),
    dict(figure=11, panel="a", page=10, method="fmm", modes=7,
         box=[234.3250515,353.6099892,383.7125145,422.8794678],
         wavelengths=[.90,.96], ypoints=[422.8794678,392.071527,361.3025328], yvalues=[0.,.2,.4],
         solid=[15], dashed=[16], diamond_modes=14, dashed_modes=14),
    dict(figure=11, panel="b", page=10, method="asr", modes=7,
         box=[234.3250515,448.4815464,383.7125145,517.751025],
         wavelengths=[.90,.96], ypoints=[517.751025,486.9430842,456.17409], yvalues=[0.,.2,.4],
         solid=[77], dashed=[78], diamond_modes=14, dashed_modes=14),
]

metadata = {
    "source_file": SOURCE.name,
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "source_title": "Reformulation of the Fourier modal method with adaptive spatial resolution: application to multilevel profiles",
    "source_authors": "T. Vallius and M. Honkanen",
    "source_year": 2002,
    "extraction": "Original PDF vector path vertices via pdfplumber; no raster digitization, no calculated results, no curve fitting. Linear axis calibration only.",
    "limitations": [
        "Coordinates have the rounding and polyline sampling resolution of the published PDF; they are not original numerical solver outputs.",
        "Curves clipped by a plot border remain clipped. No missing peak points are reconstructed.",
        "Digitized marker locations are bounding-box centers of original vector diamonds.",
        "Figure 9b solid/dashed paths have the same extrema and vertex counts but different interior vertices; the high-mode dashed trace agrees with the diamond centers.",
        "Figures 6/7 dashed reference curves are FMM 240 modes in both panels, as stated in the caption.",
    ],
    "geometry_notes": {
        "figure_5": {
            "documented_parameters": {"period": 1, "z_interfaces": [0, .125, .25, .375, .5], "n_a": [0.1217,3.2966], "n_b": 1, "n_incident": 1, "n_exit": 1},
            "horizontal_transitions_documented_numerically": False,
            "source_pdf_page": 8,
            "vector_domain_x_pt": [248.579,355.464],
            "vector_metal_polygon_xy_from_top_pt": [[248.579,86.568],[248.682,115.417],[251.977,115.417],[252.081,129.841],[259.449,129.841],[259.579,144.291],[290.970,144.291],[291.073,129.841],[298.441,129.841],[298.545,115.417],[305.264,115.417],[305.394,100.992],[312.762,100.992],[312.866,86.568],[344.464,86.568],[344.568,100.992],[351.936,100.992],[352.066,115.417],[355.360,115.417],[355.464,86.568]],
            "measured_metal_intervals_lower_to_upper_periodic": [[.102,.397],[.032,.467],[-.031,.532],[-.101,.601]],
            "suggested_rational_approximation_lower_to_upper_periodic": [[.1,.4],[1/30,7/15],[-1/30,8/15],[-.1,.6]],
            "suggested_center": .25,
            "suggested_widths": [.3,13/30,17/30,.7],
            "caution": "The figure is schematic. The suggested rational approximation is an explicit modeling assumption inferred from the PDF geometry, not a parameter quoted in the paper and not a fit to calculated efficiencies."
        },
        "figure_8": {"source_pdf_page": 9, "period": 1, "z_interfaces": [0,10,20], "n_a": 5, "n_b": 1.5, "lower_a_interval": [0,.5], "upper_a_interval": [.5,1], "n_incident": 1, "n_exit": 1},
        "figure_10": {"source_pdf_page": 10, "period": 1, "total_thickness": 1, "circle_radius": .25, "circle_centers": [[.25,.25],[.75,.75]], "circle_n_a": 1, "background_n_b": 5, "equal_z_slices": 120, "n_incident": 1, "n_exit": 1},
        "all_examples": {"resolution_G": .001, "G_source_pdf_page": 6, "incidence_degrees": 0}
    },
    "panels": [],
}

reader=PdfReader(SOURCE)
for page in [8,9,10]:
    ops=ContentStream(reader.pages[page-1].get_contents(), reader).operations
    metadata.setdefault("original_content_operator_counts", {})[str(page)] = {
        op.decode("latin1"):sum(o==op for _,o in ops) for op in [b"m",b"l",b"c",b"S",b"d",b"cm"]
    }

with pdfplumber.open(SOURCE) as doc:
    for panel in PANELS:
        page=doc.pages[panel["page"]-1]
        left,top,right,bottom=panel["box"]
        xlo,xhi=panel["wavelengths"]
        slope,intercept=np.polyfit(panel["ypoints"],panel["yvalues"],1)
        def convert(x,y):
            return xlo+(x-left)*(xhi-xlo)/(right-left), slope*y+intercept
        records=[]
        for style in ["solid","dashed"]:
            nm=panel["modes"] if style=="solid" else panel["dashed_modes"]
            for curve_id in panel[style]:
                curve=page.curves[curve_id]
                assert curve["stroke"] and len(curve["pts"])>8
                assert bool(curve["dash"][0])==(style=="dashed")
                for vertex,(x,y) in enumerate(curve["pts"]):
                    wavelength,eta0=convert(x,y)
                    records.append([style,nm,curve_id,vertex,wavelength,eta0,x,y])
        diamonds=[]
        for curve_id,curve in enumerate(page.curves):
            pts=curve["pts"]
            if len(pts)!=5 or pts[0]!=pts[-1] or not curve["stroke"]:
                continue
            x=(curve["x0"]+curve["x1"])/2
            y=(curve["top"]+curve["bottom"])/2
            # Plot border markers extend slightly outside the axes.
            if left-.1<=x<=right+.1 and top-2<=y<=bottom+2:
                if abs(pts[0][0]-pts[2][0])<.002 and abs(pts[1][1]-pts[3][1])<.002:
                    diamonds.append((curve_id,x,y))
        for vertex,(curve_id,x,y) in enumerate(sorted(diamonds,key=lambda t:t[1])):
            wavelength,eta0=convert(x,y)
            records.append(["diamond",panel["diamond_modes"],curve_id,vertex,wavelength,eta0,x,y])
        name=f"figure_{panel['figure']:02}_{panel['panel']}_{panel['method']}.csv"
        with (OUT/name).open("w",encoding="utf-8",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["style","caption_modes","pdf_curve_index","vertex","wavelength","eta0","pdf_x_pt","pdf_y_from_top_pt"])
            writer.writerows(records)
        for style in ["solid", "dashed", "diamond"]:
            split_name=f"fig{panel['figure']}_{panel['panel']}_{style}.csv"
            with (OUT/split_name).open("w",encoding="utf-8",newline="") as f:
                writer=csv.writer(f)
                writer.writerow(["wavelength","eta0"])
                writer.writerows([r[4],r[5]] for r in records if r[0]==style)
        entry=dict(panel)
        entry.update(filename=name,y_slope=float(slope),y_intercept=float(intercept),
                     tick_max_residual=float(np.max(np.abs(slope*np.array(panel["ypoints"])+intercept-np.array(panel["yvalues"])))),
                     extracted_rows={style:sum(r[0]==style for r in records) for style in ["solid","dashed","diamond"]})
        metadata["panels"].append(entry)

(OUT/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
print(json.dumps([{k:e[k] for k in ["filename","extracted_rows","tick_max_residual"]} for e in metadata["panels"]],indent=2))
