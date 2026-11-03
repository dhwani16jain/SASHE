import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=json.load(open('res_all.json')); n=len(R)
def ms(v): v=np.array(v,float); return v.mean(), v.std(ddof=1) if len(v)>1 else 0.0
PS=['1.0','0.5','0.3','0.2','0.1','0.05']
out={'n_seeds':n,'M':R[0]['M']}
def agg(getter): return ms([getter(r) for r in R])
out['central']={k:agg(lambda r,k=k:r['central'][k]) for k in ['acc','f1','auc']}
for name in ['fedavg','scaffold','fedavg_dp1','fedavg_dp4']:
    out[name]={k:agg(lambda r,k=k,name=name:r[name][k]) for k in ['acc','f1','auc','up_per_round']}

for p in PS:
    out['he_'+p]={k:agg(lambda r,k=k,p=p:r['he'][p][k]) for k in ['acc','f1','auc','up_per_round','down_per_round','t_enc','t_agg','t_dec','t_local','maxerr','n_enc']}
    out['he_'+p]['t_he']=agg(lambda r,p=p:r['he'][p]['t_enc']+r['he'][p]['t_agg']+r['he'][p]['t_dec'])
out['setup']={k:agg(lambda r,k=k:r['setup'][k]) for k in ['t_enc','t_agg','t_dec','bytes_up','err']}
# rounds to reach >=0.95
def r2t(c,t=0.95):
    for i,a in enumerate(c):
        if a>=t: return i+1
    return np.nan
for name in ['fedavg','scaffold']:
    out[name]['r95']=ms([r2t(r[name]['curve']) for r in R])
    out[name]['r90']=ms([r2t(r[name]['curve'],0.90) for r in R])
# privacy
keys=list(R[0]['privacy'].keys())
out['privacy']={k:agg(lambda r,k=k:r['privacy'][k]) for k in keys}
out['dy_norm_dp']=agg(lambda r:r['fedavg_dp1']['dy_norm'])
out['sizes']=[r['sizes'] for r in R]; out['pos']=[r['pos_frac'] for r in R]
json.dump(out,open('summary.json','w'),indent=1)
print(json.dumps({k:v for k,v in out.items() if k in['central','fedavg','scaffold','fedavg_dp1','fedavg_dp4','setup','dy_norm_dp']},indent=1))
for p in PS: print(p,{k:(round(v[0],4),round(v[1],4)) for k,v in out['he_'+p].items()})
for k,v in out['privacy'].items(): print(k,round(v[0],4),round(v[1],4))
