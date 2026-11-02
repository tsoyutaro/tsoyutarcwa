"""Read the solid NV-radial lines from the embedded Fig.10 JPEG (not raw data).

Uses axis tick calibration and linear fits to the interiors of successive
order-to-order line segments. Circular electrostatic markers are excluded.
Every extracted point remains an estimate with a conservative pixel bound.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def digitize(image, output):
    gray=np.asarray(Image.open(image).convert('L'))
    if gray.shape != (1523,850):
        raise ValueError('Requires original 850x1523 embedded Im11.jpg, without resizing.')
    x=np.linspace(150.,842.,10)
    panels=[('diagonal',50.,613.,[500,438,429,406,399,388,387,382,380,374]),
            ('x',835.,1398.,[1285,1223,1215,1192,1186,1174,1171,1167,1165,1160])]
    rows=[]; diagnostics=[]
    for pol,top,bottom,rough in panels:
        fits=[]
        for i in range(9):
            xx=np.arange(int(np.ceil(x[i]+14)),int(np.floor(x[i+1]-14)))
            points=[]
            for xi in xx:
                expected=np.interp(xi,x,rough)
                ya,yb=int(np.floor(expected-9)),int(np.ceil(expected+9))
                ink=np.where(gray[ya:yb+1,xi]<110)[0]+ya
                groups=[g for g in np.split(ink,np.where(np.diff(ink)>1)[0]+1) if len(g)>0]
                if not groups:raise ValueError(f'Missing radial line at {pol} x={xi}')
                selected=min(groups,key=lambda g:abs(g.mean()-expected))
                points.append((xi,float(selected.mean())))
            px,py=np.array(points).T
            fit=np.polyfit(px,py,1)
            residual=float(np.max(np.abs(py-np.polyval(fit,px))))
            if residual>2.:raise ValueError(f'Unexpected line curvature/raster contamination: {pol} interval={i}, residual={residual}')
            fits.append(fit)
            diagnostics.append(dict(polarization=pol,from_order=i+4,to_order=i+5,
                pixel_line_fit=fit.tolist(),max_residual_px=residual))
        for i,xi in enumerate(x):
            estimates=[]
            if i>0:estimates.append(float(np.polyval(fits[i-1],xi)))
            if i<9:estimates.append(float(np.polyval(fits[i],xi)))
            y=float(np.mean(estimates))
            r=.25+(y-top)*(.24-.25)/(bottom-top)
            rows.append(dict(polarization=pol,order=i+4,R00_paper_digitized=r,
                uncertainty_abs_R=2.5*.01/(bottom-top),pixel_x=float(xi),pixel_y=y,
                adjacent_fit_spread_px=max(estimates)-min(estimates)))
    output.mkdir(parents=True,exist_ok=True)
    with (output/'paper_radial_digitized.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    info=dict(source_image=str(image),source_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        original_pixel_size=[850,1523],curve='solid NV radial only; not NV electrostatic circles',
        raw_numerical_data_available=False,uncertainty_convention='conservative +/-2.5 pixels, not a statistical confidence interval',
        calibration=dict(x_pixels=[150,842],order=[4,13],y_pixels_diagonal=[50,613],
                         y_pixels_x=[835,1398],R00=[.25,.24]),segment_fits=diagnostics)
    (output/'digitization_provenance.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    # Visual overlay makes curve selection and endpoint calibration inspectable.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,15.23))
    ax.imshow(Image.open(image).convert('RGB'))
    for pol,_,_,_ in panels:
        rr=[v for v in rows if v['polarization']==pol]
        ax.plot([v['pixel_x'] for v in rr],[v['pixel_y'] for v in rr],'-+',color='#d62828',lw=1.,ms=10,label=pol)
    ax.set_xlim(0,850);ax.set_ylim(1523,0);ax.axis('off')
    fig.subplots_adjust(0,0,1,1)
    fig.savefig(output/'digitization_overlay.png',dpi=150)
    plt.close(fig)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();digitize(a.image,a.output)
