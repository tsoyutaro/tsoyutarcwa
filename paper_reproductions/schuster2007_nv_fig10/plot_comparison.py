"""Plot saved production results against raster-digitized Schuster Fig.10."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def read_csv(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))


def main(root):
    names={x.name for x in font_manager.fontManager.ttflist}
    for name in ['Meiryo','Yu Gothic','Noto Sans CJK JP','DejaVu Sans']:
        if name in names:
            plt.rcParams['font.family']=name;break
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.unicode_minus':False,'svg.fonttype':'none'})
    data=read_csv(root/'results/production_ON_OFF.csv')
    ref=read_csv(root/'reference/paper_radial_digitized.csv')
    colors={'ON':'#226bc0','OFF':'#e36b06'}
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True,
                          gridspec_kw={'height_ratios':[1.25,1]})
    metrics=[];paired=[]
    for column,(pol,title) in enumerate([('diagonal','図10(a)：[1,1]偏光'),('x','図10(b)：[1,0]偏光')]):
        paper=sorted([r for r in ref if r['polarization']==pol],key=lambda r:int(r['order']))
        orders=np.array([int(r['order']) for r in paper])
        pr=np.array([float(r['R00_paper_digitized']) for r in paper])
        uncertainty=np.array([float(r['uncertainty_abs_R']) for r in paper])
        ax=axes[0,column]; err_ax=axes[1,column]
        ax.fill_between(orders,pr-uncertainty,pr+uncertainty,color='#737b83',alpha=.16,
                        label='図の読取幅（±2.5 pixel）')
        ax.plot(orders,pr,'D-',color='#292d33',ms=4,lw=1.4,label='論文 NV radial（図から読取）')
        for mode,marker,style in [('ON','o','-'),('OFF','s','--')]:
            values=np.array([float(next(r for r in data if r['mode']==mode and
                int(r['grid'])==1024 and int(r['order'])==m)['R00']) for m in orders])
            error=values-pr
            ax.plot(orders,values,marker=marker,linestyle=style,color=colors[mode],ms=4,
                    lw=1.7,mfc='white',label='正則化'+mode+'（現実装）')
            err_ax.plot(orders,error*1e5,marker=marker,linestyle=style,color=colors[mode],
                        ms=4,lw=1.6,mfc='white',label=mode)
            metrics.append(dict(polarization=pol,mode=mode,grid=1024,orders=orders.tolist(),
                mean_abs_error=float(np.mean(abs(error))),rmse=float(np.sqrt(np.mean(error**2))),
                max_abs_error=float(np.max(abs(error))),digitization_bound_abs_R=float(uncertainty[0]),
                points_inside_readout_bound=int(np.sum(abs(error)<=uncertainty))))
            for m,pp,rr,ee,u in zip(orders,pr,values,error,uncertainty):
                paired.append(dict(polarization=pol,order=int(m),grid=1024,mode=mode,
                                   R00_paper_digitized=pp,R00=rr,error=ee,paper_readout_bound=u,
                                   calculation_source='x-source sector; diagonal equivalent by square symmetry' if pol=='diagonal' else 'x-source sector'))
        err_ax.axhspan(-uncertainty[0]*1e5,uncertainty[0]*1e5,color='#737b83',alpha=.16)
        err_ax.axhline(0.,color='#444',lw=.9)
        ax.set_title(title,pad=12)
        ax.set_ylim(.2417,.2445)
        ax.set_ylabel('ゼロ次反射効率 '+r'$R_{00}$')
        err_ax.set_ylabel(r'計算値 − 論文読取値（$\times 10^{-5}$）')
        err_ax.set_xlabel('回折次数 M')
        for part in [ax,err_ax]:
            part.grid(axis='y',alpha=.25);part.set_xticks(orders)
            part.set_xlim(3.7,13.3)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.073),ncol=2,frameon=False)
    fig.suptitle('円形空気孔：論文図10と正則化ON・OFFの比較',fontsize=20,y=.98)
    fig.text(.5,.925,'正方格子 P=1000 nm・孔径500 nm・膜厚50 nm・λ=500 nm・grid 1024',ha='center',fontsize=11)
    fig.text(.5,.029,'ON/OFFは同じ対称化NVM式。正方格子対称性により両偏光のR₀₀は同値（M4/13で直接確認）。',ha='center',fontsize=10,color='#555')
    fig.text(.5,.007,'論文曲線は図からの推定値。灰色の幅は読取誤差の目安であり、厳密解の誤差ではありません。',ha='center',fontsize=10,color='#555')
    fig.subplots_adjust(left=.085,right=.98,top=.855,bottom=.22,hspace=.16,wspace=.25)
    for ext in ['png','svg']:
        fig.savefig(root/f'figure10_ON_OFF_comparison.{ext}',dpi=200,facecolor='white')
    plt.close(fig)
    with (root/'comparison.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(paired[0]));w.writeheader();w.writerows(paired)
    uncertainty=float(ref[0]['uncertainty_abs_R'])
    ranking={}
    for pol in ['x','diagonal']:
        on=next(r for r in metrics if r['polarization']==pol and r['mode']=='ON')
        off=next(r for r in metrics if r['polarization']==pol and r['mode']=='OFF')
        ranges=[]
        for m in range(4,14):
            pp=next(r for r in paired if r['polarization']==pol and r['mode']=='ON' and r['order']==m)
            oo=next(r for r in paired if r['polarization']==pol and r['mode']=='OFF' and r['order']==m)
            lo,hi=pp['R00_paper_digitized']-uncertainty,pp['R00_paper_digitized']+uncertainty
            candidates=[lo,hi]+[v for v in [pp['R00'],oo['R00']] if lo<=v<=hi]
            values=[abs(pp['R00']-v)-abs(oo['R00']-v) for v in candidates]
            ranges.append([min(values),max(values)])
        lower,upper=np.mean(ranges,axis=0)
        ranking[pol]=dict(lower_MAE='OFF' if off['mean_abs_error']<on['mean_abs_error'] else 'ON',
            MAE_ON_minus_OFF=on['mean_abs_error']-off['mean_abs_error'],
            MAE_difference_allowed_by_readout_intervals=[float(lower),float(upper)],
            robust_winner_from_readout_intervals=bool(lower>0 or upper<0),
            interpretation='Nominal OFF MAE is smaller; difference is below raster readout uncertainty.')
    grid_checks=[]
    for mode in ['ON','OFF']:
        diffs=[]
        for m in range(4,14):
            vals={int(r['grid']):float(r['R00']) for r in data if r['mode']==mode and int(r['order'])==m}
            diffs.append(abs(vals[512]-vals[1024]))
        grid_checks.append(dict(mode=mode,coarse_grid=512,fine_grid=1024,max_abs_R00_difference=max(diffs)))
    point13={mode:float(next(r for r in data if r['mode']==mode and r['order']=='13' and r['grid']=='1024')['R00']) for mode in ['ON','OFF']}
    summary=dict(metrics=metrics,ranking=ranking,grid_checks=grid_checks,M13_Q1024=point13,
                 digitization_readout_bound=uncertainty,
                 conclusion='Both reproduce the paper trend. OFF is nominally closer, but the difference cannot establish a robust accuracy ranking.')
    (root/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    main(p.parse_args().root)
