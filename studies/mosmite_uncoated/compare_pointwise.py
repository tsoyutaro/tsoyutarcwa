"""Compare each computed wavelength directly with a measurement; no spectral averaging."""
from pathlib import Path
import argparse,csv,hashlib,io,json,math
import numpy as np

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def read_measurement(path,column):
    raw=Path(path).read_bytes()
    try:
        rows=read_csv(path)
        if rows and 'wavelength_nm' in rows[0]:
            if column not in rows[0]:raise ValueError('Specify a measurement column from: '+', '.join(rows[0]))
            return np.array([[float(r['wavelength_nm']),float(r[column])] for r in rows])
    except UnicodeDecodeError:pass
    rows=list(csv.reader(io.StringIO(raw.decode('cp932'))))
    start=next((i for i,r in enumerate(rows) if r and r[0].strip()=='XYDATA'),None)
    if start is None:raise ValueError('Expected a wavelength_nm CSV or JASCO XYDATA CSV.')
    header={r[0].strip():r[1].strip() for r in rows[:start] if len(r)>1}
    if header.get('XUNITS')!='NANOMETERS' or header.get('YUNITS')!='TRANSMITTANCE':raise ValueError('Expected JASCO wavelength and transmittance.')
    values=[]
    for row in rows[start+1:]:
        try:values.append([float(row[0]),float(row[1])])
        except (ValueError,IndexError):break
    footer={r[0].strip():r[1].strip() for r in rows[start+1+len(values):] if len(r)>1}
    if footer.get('縦軸')!='%T':raise ValueError('Expected measured %T.')
    return np.array(values)

def compare(spectrum,measurement,column):
    rows=read_csv(spectrum);data=read_measurement(measurement,column)
    if not np.isfinite(data).all() or len(np.unique(data[:,0]))!=len(data):raise ValueError('Nonfinite or duplicate measurement data.')
    lookup={round(float(w),8):float(t) for w,t in data}
    output=[]
    for row in rows:
        wl=float(row['wavelength_nm']);key=round(wl,8)
        if key not in lookup:raise ValueError(f'{wl}nm is absent from the measurement; spectral interpolation/averaging is not applied.')
        T=100*float(row['T']);observed=lookup[key]
        output.append(dict(wavelength_nm=wl,simulated_T_percent=T,measured_T_percent=observed,simulated_minus_measured_pp=T-observed))
    if not output:raise ValueError('No simulation wavelengths.')
    d=np.array([r['simulated_minus_measured_pp'] for r in output])
    metrics=dict(points=len(output),RMSE_pp=float(np.sqrt(np.sum(d*d)/len(d))),MAE_pp=float(np.sum(abs(d))/len(d)),bias_pp=float(np.sum(d)/len(d)),max_abs_pp=float(np.max(abs(d))))
    return output,metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spectrum',type=Path,required=True)
    p.add_argument('--measurement',type=Path,required=True)
    p.add_argument('--measurement-column',default='T_grand_mean_percent')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();rows,metrics=compare(a.spectrum,a.measurement,a.measurement_column)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'comparison_pointwise.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(metrics=metrics,comparison='exact wavelength match; no moving windows, wavelength interpolation, phase samples or spectral smoothing',measurement_column=a.measurement_column,
                sources={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.spectrum,a.measurement]})
    (a.output_dir/'comparison_pointwise.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
