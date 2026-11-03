"""SASHE-FL prototype: real CKKS (TenSEAL) selective encryption + SCAFFOLD on a
non-IID partition of the Wisconsin breast-cancer diagnostic dataset (scikit-learn copy)."""
import numpy as np, time, json, sys, tenseal as ts
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

H1, H2 = 256, 128
N_CLIENTS, ROUNDS, LOCAL_STEPS, BATCH, LR = 5, 30, 10, 32, 0.05
SLOTS = 4096

def shapes(d): return [(d,H1),(H1,),(H1,H2),(H2,),(H2,1),(1,)]
def sizes(d): return [int(np.prod(s)) for s in shapes(d)]
def flat(ts_): return np.concatenate([t.ravel() for t in ts_])
def unflat(v,d):
    out=[];i=0
    for s,n in zip(shapes(d),sizes(d)): out.append(v[i:i+n].reshape(s)); i+=n
    return out
def init(rng,d):
    W1=rng.normal(0,np.sqrt(2/d),(d,H1));W2=rng.normal(0,np.sqrt(2/H1),(H1,H2));W3=rng.normal(0,np.sqrt(1/H2),(H2,1))
    return flat([W1,np.zeros(H1),W2,np.zeros(H2),W3,np.zeros(1)])

def fwd(v,X,d):
    W1,b1,W2,b2,W3,b3=unflat(v,d)
    p1=X@W1+b1;a1=np.maximum(p1,0);p2=a1@W2+b2;a2=np.maximum(p2,0);z=(a2@W3+b3).ravel()
    return z,(X,p1,a1,p2,a2)
def sig(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
def grad(v,X,y,d):
    W1,b1,W2,b2,W3,b3=unflat(v,d); z,(X_,p1,a1,p2,a2)=fwd(v,X,d)
    dz=((sig(z)-y)/len(y))[:,None]
    gW3=a2.T@dz; gb3=dz.sum(0); d2=(dz@W3.T)*(p2>0); gW2=a1.T@d2; gb2=d2.sum(0)
    d1=(d2@W2.T)*(p1>0); gW1=X.T@d1; gb1=d1.sum(0)
    return flat([gW1,gb1,gW2,gb2,gW3,gb3])
def sensitivity(v,X,d):
    """S(w_m)=mean_k |d/dy_k (dl_k/dw_m)| ; for sigmoid+BCE this equals mean_k |dz_k/dw_m| (exact)."""
    W1,b1,W2,b2,W3,b3=unflat(v,d); z,(X_,p1,a1,p2,a2)=fwd(v,X,d); K=len(X)
    D2=np.abs(W3.T*(p2>0)); D1=np.abs((W3.T*(p2>0))@W2.T*(p1>0)) if False else np.abs(((W3.T*(p2>0))@W2.T)*(p1>0))
    S=[np.abs(X).T@D1/K, D1.mean(0), np.abs(a1).T@D2/K, D2.mean(0), np.abs(a2).mean(0)[:,None], np.ones(1)]
    return flat(S)

def metrics(v,X,y,d):
    z,_=fwd(v,X,d); pr=(z>0).astype(int)
    return float((pr==y).mean()), float(f1_score(y,pr)), float(roc_auc_score(y,z))

def partition(y,rng,alpha=0.5,min_size=25):
    while True:
        idx=[[] for _ in range(N_CLIENTS)]
        for c in (0,1):
            ids=rng.permutation(np.where(y==c)[0]); pr=rng.dirichlet(alpha*np.ones(N_CLIENTS))
            cuts=(np.cumsum(pr)*len(ids)).astype(int)[:-1]
            for k,part in enumerate(np.split(ids,cuts)): idx[k]+=part.tolist()
        if min(len(i) for i in idx)>=min_size and all(0<y[i].mean()<1 for i in idx): return [np.array(i) for i in idx]

def local_train(x,c_glob,c_loc,X,y,d,rng,scaffold):
    yv=x.copy()
    for _ in range(LOCAL_STEPS):
        b=rng.choice(len(X),size=min(BATCH,len(X)),replace=False)
        g=grad(yv,X[b],y[b],d)
        if scaffold: g=g-c_loc+c_glob
        yv=yv-LR*g
    dy=yv-x
    if scaffold:
        c_new=c_loc-c_glob+(x-yv)/(LOCAL_STEPS*LR); return dy,c_new-c_loc,c_new
    return dy,None,None

def make_ctx():
    c=ts.context(ts.SCHEME_TYPE.CKKS,poly_modulus_degree=8192,coeff_mod_bit_sizes=[60,40,40,60])
    c.global_scale=2**40; c.generate_galois_keys(); return c

def enc_chunks(ctx,vec):
    return [ts.ckks_vector(ctx,vec[i:i+SLOTS].tolist()) for i in range(0,len(vec),SLOTS)]
def ct_bytes(cts): return sum(len(c.serialize()) for c in cts)

def build_mask(S,p,scope,d):
    M=len(S)
    if p>=1: return np.ones(M,bool)
    m=np.zeros(M,bool)
    if scope=='global':
        k=int(np.ceil(p*M)); m[np.argsort(-S)[:k]]=True
    else:
        i=0
        for n in sizes(d):
            k=int(np.ceil(p*n)); seg=S[i:i+n]; m[i:i+n][np.argsort(-seg)[:k]]=True; i+=n
    return m

def run_fl(mode,seed,data,p=None,scope='layer',mask=None,dp_z=0.0,sk_ctx=None,pub_ctx=None,log=None):
    """mode: fedavg | scaffold | he  (he = SCAFFOLD + CKKS on masked coordinates)"""
    Xtr,ytr,Xte,yte,parts=data; d=Xtr.shape[1]; M=len(init(np.random.default_rng(0),d))
    rng=np.random.default_rng(seed+100); x=init(np.random.default_rng(seed),d)
    scaffold=mode in('scaffold','he'); c=np.zeros(M); cl=[np.zeros(M) for _ in range(N_CLIENTS)]
    n=np.array([len(p_) for p_ in parts]); alpha=n/n.sum()
    curve=[]; norms=[]; t_enc=t_agg=t_dec=t_loc=0.; up=down=0; maxerr=0.
    if mode=='he': K=np.where(mask)[0]; U=np.where(~mask)[0]; nk=len(K)
    for r in range(ROUNDS):
        dys=[];dcs=[]
        t0=time.perf_counter()
        for i,idx in enumerate(parts):
            dy,dc,cn=local_train(x,c,cl[i],Xtr[idx],ytr[idx],d,rng,scaffold)
            if dp_z>0:
                nrm=np.linalg.norm(dy); norms.append(nrm); dy=dy*min(1.0,DP_CLIP/nrm)
            dys.append(dy); dcs.append(dc)
            if scaffold: cl[i]=cn
        t_loc+=(time.perf_counter()-t0)/N_CLIENTS
        agg_dy=sum(a*dy for a,dy in zip(alpha,dys))
        if dp_z>0: agg_dy=agg_dy+rng.normal(0,dp_z*DP_CLIP/N_CLIENTS,M)
        if mode=='he':
            agg_dc_true=sum(dc for dc in dcs)/N_CLIENTS
            # client side: encrypt [dy_K, dc_K] with public context
            cts=[];t0=time.perf_counter()
            for i in range(N_CLIENTS): cts.append(enc_chunks(pub_ctx,np.concatenate([dys[i][K],dcs[i][K]])))
            t_enc+=(time.perf_counter()-t0)/N_CLIENTS
            up+=ct_bytes(cts[0])+8*len(U)   # per-client upload: ciphertexts + plaintext dy_U,dc_U (float32)
            # server side: weighted sum on ciphertexts (server context has no secret key)
            t0=time.perf_counter(); aggct=[]
            for j in range(len(cts[0])):
                lo=j*SLOTS; ln=min(SLOTS,2*nk-lo)
                acc=None
                for i in range(N_CLIENTS):
                    w=np.concatenate([np.full(nk,alpha[i]),np.full(nk,1.0/N_CLIENTS)])[lo:lo+ln].tolist()
                    term=cts[i][j]*w
                    acc=term if acc is None else acc+term
                aggct.append(acc)
            agg_U_dy=agg_dy[U]; agg_U_dc=sum(dc[U] for dc in dcs)/N_CLIENTS
            t_agg+=time.perf_counter()-t0
            down+=ct_bytes(aggct)+8*len(U)
            t0=time.perf_counter(); dec=np.concatenate([np.array(a.decrypt(sk_ctx.secret_key())) for a in aggct]); t_dec+=time.perf_counter()-t0
            dyK,dcK=dec[:nk],dec[nk:2*nk]
            agg_dy2=np.zeros(M);agg_dy2[K]=dyK;agg_dy2[U]=agg_U_dy
            agg_dc2=np.zeros(M);agg_dc2[K]=dcK;agg_dc2[U]=agg_U_dc
            maxerr=max(maxerr,np.abs(agg_dy2-agg_dy).max(),np.abs(agg_dc2-agg_dc_true).max())
            agg_dy,agg_dc=agg_dy2,agg_dc2
        else:
            if scaffold: agg_dc=sum(dcs)/N_CLIENTS
            up+=(8 if scaffold else 4)*M; down+=(8 if scaffold else 4)*M
        x=x+agg_dy
        if scaffold: c=c+agg_dc
        curve.append(metrics(x,Xte,yte,d)[0])
    acc,f1,auc=metrics(x,Xte,yte,d)
    return dict(acc=acc,f1=f1,auc=auc,curve=curve,up_per_round=up/ROUNDS,down_per_round=down/ROUNDS,
                t_enc=t_enc/ROUNDS,t_agg=t_agg/ROUNDS,t_dec=t_dec/ROUNDS,t_local=t_loc/ROUNDS,maxerr=maxerr,dy_norm=(float(np.mean(norms)) if norms else None),model=x)

DP_CLIP=1.0

def prepare(seed):
    X,y=load_breast_cancer(return_X_y=True)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,stratify=y,random_state=seed)
    sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
    parts=partition(ytr,np.random.default_rng(seed))
    return Xtr,ytr,Xte,yte,parts

def central(seed,data):
    Xtr,ytr,Xte,yte,_=data; d=Xtr.shape[1]; rng=np.random.default_rng(seed+7); x=init(np.random.default_rng(seed),d)
    for _ in range(ROUNDS*LOCAL_STEPS*N_CLIENTS):
        b=rng.choice(len(Xtr),size=BATCH,replace=False); x=x-LR*grad(x,Xtr[b],ytr[b],d)
    return metrics(x,Xte,yte,d)

def single_grads(v,X,y,d):
    """per-sample gradient pieces needed for the analytic inversion attack"""
    W1,b1,W2,b2,W3,b3=unflat(v,d); z,(X_,p1,a1,p2,a2)=fwd(v,X,d)
    dz=(sig(z)-y)[:,None]; d2=(dz@W3.T)*(p2>0); d1=(d2@W2.T)*(p1>0); return d1

def attack_cos(G_vis,vis,x_true,iters=30):
    """rank-1 matrix completion of the visible entries of gW1 = x delta^T (alternating least squares)."""
    dl,H=G_vis.shape
    e=(G_vis**2).sum(0); 
    if e.max()==0: return 0.0
    j=e.argmax(); xh=G_vis[:,j].copy(); 
    for _ in range(iters):
        num=(xh[:,None]*G_vis).sum(0); den=((xh[:,None]**2)*vis).sum(0)
        dh=np.where(den>0,num/np.maximum(den,1e-30),0)
        numx=(vis*dh[None,:]*G_vis).sum(1); denx=(vis*(dh[None,:]**2)).sum(1)
        xh=np.where(denx>0,numx/np.maximum(denx,1e-30),0)
        if np.linalg.norm(xh)==0: return 0.0
    return float(abs(xh@x_true)/(np.linalg.norm(xh)*np.linalg.norm(x_true)+1e-12))

def privacy_proxy(model,S_init,data,d,ps,rng,nsamp=60,S_final=None):
    Xtr,ytr=data[0],data[1]; ids=rng.choice(len(Xtr),nsamp,replace=False); M=len(model)
    W1n=d*H1; out={}
    G=[sim_grad(model,Xtr[k:k+1],ytr[k:k+1],d) for k in ids]
    def energy(mask):
        return float(np.mean([ (g[~mask]**2).sum()/(g**2).sum() for g in G]))
    def cos_for(mask):
        vals=[]
        for k in ids:
            x=Xtr[k:k+1]; y=ytr[k:k+1]; d1=single_grads(model,x,y,d)[0]
            gW1=np.outer(x[0],d1); vis=(~mask[:W1n]).reshape(d,H1).astype(float)
            vals.append(attack_cos(gW1*vis,vis,x[0]))
        return float(np.mean(vals))
    def both(name,mask): out['cos_'+name]=cos_for(mask); out['exp_'+name]=energy(mask); out['frac_'+name]=float(mask.mean())
    both('none',np.zeros(M,bool))
    for p in ps:
        both(f'sens_global_{p}',build_mask(S_init,p,'global',d)); both(f'sens_layer_{p}',build_mask(S_init,p,'layer',d))
        both(f'random_{p}',np.isin(np.arange(M),rng.choice(M,int(np.ceil(p*M)),replace=False)))
        if S_final is not None: both(f'refresh_layer_{p}',build_mask(S_final,p,'layer',d))
        m=build_mask(S_init,p,'layer',d); m[:W1n+H1]=True; both(f'guard_{p}',m)
    return out
def sim_grad(v,X,y,d): return grad(v,X,y,d)
