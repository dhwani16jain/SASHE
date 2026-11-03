import sim, numpy as np, json, tenseal as ts, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
sk=sim.make_ctx(); pub=ts.context_from(sk.serialize()); pub.make_context_public()
out={'guard':[], 'noniid':[]}
# (a) guard mask (all of W1,b1 + layer-wise top-p elsewhere), real CKKS cost
for seed in range(5):
    data=sim.prepare(seed); Xtr,ytr,Xte,yte,parts=data; d=Xtr.shape[1]
    x0=sim.init(np.random.default_rng(seed),d); S=sum(sim.sensitivity(x0,Xtr[p],d) for p in parts)/len(parts)
    W1n=d*sim.H1
    for p in (0.05,0.1):
        m=sim.build_mask(S,p,'layer',d); m[:W1n+sim.H1]=True
        o=sim.run_fl('he',seed,data,mask=m,sk_ctx=sk,pub_ctx=pub)
        out['guard'].append(dict(seed=seed,p=p,frac=float(m.mean()),acc=o['acc'],f1=o['f1'],up=o['up_per_round'],down=o['down_per_round'],t=o['t_enc']+o['t_agg']+o['t_dec']))
# (b) stronger heterogeneity: Dirichlet alpha 0.1, plaintext FedAvg vs SCAFFOLD
def prep(seed,alpha):
    X,y=load_breast_cancer(return_X_y=True)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,stratify=y,random_state=seed)
    sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
    return Xtr,ytr,Xte,yte,sim.partition(ytr,np.random.default_rng(seed),alpha=alpha,min_size=15)
for alpha in (0.5,0.1):
    for seed in range(10):
        data=prep(seed,alpha); r={'alpha':alpha,'seed':seed}
        for m in ('fedavg','scaffold'):
            o=sim.run_fl(m,seed,data); c=o['curve']
            r[m]=dict(acc=o['acc'],f1=o['f1'],auc=o['auc'],mean_last5=float(np.mean(c[-5:])),r90=next((i+1 for i,a in enumerate(c) if a>=0.90),None))
        out['noniid'].append(r)
json.dump(out,open('extra.json','w'))
print('done')
