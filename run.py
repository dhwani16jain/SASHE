import sim, numpy as np, time, json, sys, tenseal as ts, pickle
seeds=[int(s) for s in sys.argv[1].split(',')]; out=sys.argv[2]
PS=[1.0,0.5,0.3,0.2,0.1,0.05]
res=[]
sk=sim.make_ctx(); pub=ts.context_from(sk.serialize()); pub.make_context_public()
for seed in seeds:
    t0=time.time(); data=sim.prepare(seed); Xtr,ytr,Xte,yte,parts=data; d=Xtr.shape[1]
    r=dict(seed=seed,sizes=[len(p) for p in parts],pos_frac=[float(ytr[p].mean()) for p in parts])
    a,f,u=sim.central(seed,data); r['central']=dict(acc=a,f1=f,auc=u)
    for name,kw in [('fedavg',dict(mode='fedavg')),('scaffold',dict(mode='scaffold')),
                    ('fedavg_dp1',dict(mode='fedavg',dp_z=1.0)),('fedavg_dp4',dict(mode='fedavg',dp_z=4.0))]:
        o=sim.run_fl(seed=seed,data=data,**kw); r[name]={k:v for k,v in o.items() if k!='model'}
        if name=='scaffold': model=o['model']
    # real setup phase: encrypted sensitivity aggregation
    x0=sim.init(np.random.default_rng(seed),d); S_loc=[sim.sensitivity(x0,Xtr[p],d) for p in parts]
    t=time.perf_counter(); cts=[sim.enc_chunks(pub,s) for s in S_loc]; t_enc=(time.perf_counter()-t)/len(parts)
    t=time.perf_counter(); agg=[]
    for j in range(len(cts[0])):
        acc=cts[0][j]
        for i in range(1,len(parts)): acc=acc+cts[i][j]
        agg.append(acc)
    t_agg=time.perf_counter()-t
    t=time.perf_counter(); S=np.concatenate([np.array(a.decrypt(sk.secret_key())) for a in agg])/len(parts); t_dec=time.perf_counter()-t
    r['setup']=dict(t_enc=t_enc,t_agg=t_agg,t_dec=t_dec,bytes_up=sim.ct_bytes(cts[0]),err=float(np.abs(S-sum(S_loc)/len(parts)).max()))
    r['he']={}
    for p in PS:
        mask=sim.build_mask(S,p,'layer',d)
        o=sim.run_fl('he',seed,data,mask=mask,sk_ctx=sk,pub_ctx=pub); o={k:v for k,v in o.items() if k!='model'}; o['n_enc']=int(mask.sum()); r['he'][str(p)]=o
    S_fin=sum(sim.sensitivity(model,Xtr[p],d) for p in parts)/len(parts)
    r['privacy']=sim.privacy_proxy(model,S,data,d,[0.05,0.1,0.3],np.random.default_rng(seed+5),S_final=S_fin)
    r['M']=int(len(model)); res.append(r)
    print('seed',seed,'done',round(time.time()-t0,1),'s; acc fedavg/scaffold/he1.0/he0.1:',r['fedavg']['acc'],r['scaffold']['acc'],r['he']['1.0']['acc'],r['he']['0.1']['acc'],flush=True)
    json.dump(res,open(out,'w'))
