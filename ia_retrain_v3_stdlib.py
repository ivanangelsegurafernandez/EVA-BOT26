#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, os, random, statistics, pickle
from datetime import datetime
from typing import List, Dict, Tuple, Any

SEED = 20260318
random.seed(SEED)

SCALPING10 = [
    "ret_1m","ret_3m","ret_5m","slope_5m","rv_20",
    "range_norm","bb_z","body_ratio","wick_imbalance","micro_trend_persist"
]
CORE13 = ["racha_actual","puntaje_estrategia","payout"] + SCALPING10
DEFAULTS = {k:0.0 for k in SCALPING10}
TARGET = "result_bin"
PRIMARY_DATASET = "dataset_incremental_v3.csv"
CONTRAST_DATASET = "dataset_incremental.csv"
STRICT_PRIMARY_DATASET = True


def to_float(v, default=None):
    try:
        if v is None: return default
        s=str(v).strip()
        if s=="": return default
        x=float(s)
        if math.isfinite(x): return x
        return default
    except Exception:
        return default


def load_rows(path:str)->Tuple[List[Dict[str,float]],Dict[str,Any]]:
    info={"path":path,"exists":os.path.exists(path)}
    if not info["exists"]:
        return [],info
    rows=[]
    with open(path,newline='',encoding='utf-8',errors='replace') as f:
        rdr=csv.DictReader(f)
        info["columns"]=rdr.fieldnames or []
        for r in rdr:
            y=to_float(r.get(TARGET),None)
            if y not in (0.0,1.0):
                continue
            row={}
            bad=False
            for c in CORE13:
                x=to_float(r.get(c),None)
                if x is None:
                    bad=True; break
                row[c]=x
            if bad: continue
            row[TARGET]=int(y)
            # optional time
            t=to_float(r.get("epoch"),None)
            if t is None:
                t=to_float(r.get("timestamp"),None)
            row["_t"]=t
            rows.append(row)
    info["rows_loaded"]=len(rows)
    return rows,info


def exact_duplicates(rows):
    seen=set(); d=0
    for r in rows:
        key=tuple(round(r[c],12) for c in CORE13)+(r[TARGET],)
        if key in seen: d+=1
        else: seen.add(key)
    return d


def class_balance(rows):
    n=len(rows)
    if n==0:return {}
    pos=sum(r[TARGET] for r in rows)
    neg=n-pos
    return {"n":n,"pos":pos,"neg":neg,"pos_rate":pos/n}


def var(vals):
    if len(vals)<2:return 0.0
    m=sum(vals)/len(vals)
    return sum((x-m)**2 for x in vals)/len(vals)


def scalping_health(rows):
    out={}
    ge4=0; all10=0
    for r in rows:
        real=sum(1 for c in SCALPING10 if abs(r[c]-DEFAULTS[c])>1e-12)
        if real>=4: ge4+=1
        if real==10: all10+=1
    for c in SCALPING10:
        vals=[r[c] for r in rows]
        d=DEFAULTS[c]
        dr=sum(1 for v in vals if abs(v-d)<=1e-12)/len(vals) if vals else 1.0
        out[c]={"default_ratio":dr,"var":var(vals),"min":min(vals) if vals else None,"max":max(vals) if vals else None}
    return out,{"ge4":ge4,"all10":all10,"total":len(rows)}


def rankdata(vals):
    s=sorted((v,i) for i,v in enumerate(vals))
    ranks=[0.0]*len(vals)
    i=0
    while i<len(s):
        j=i
        while j+1<len(s) and s[j+1][0]==s[i][0]: j+=1
        r=(i+j+2)/2.0
        for k in range(i,j+1): ranks[s[k][1]]=r
        i=j+1
    return ranks


def auc(y,p):
    n=len(y)
    pos=sum(y); neg=n-pos
    if pos==0 or neg==0:return None
    r=rankdata(p)
    sum_pos=sum(r[i] for i,v in enumerate(y) if v==1)
    u=sum_pos-pos*(pos+1)/2
    return u/(pos*neg)


def pearson(x,y):
    n=len(x)
    if n<2:return 0.0
    mx=sum(x)/n; my=sum(y)/n
    vx=sum((a-mx)**2 for a in x); vy=sum((b-my)**2 for b in y)
    if vx<=0 or vy<=0:return 0.0
    c=sum((x[i]-mx)*(y[i]-my) for i in range(n))
    return c/math.sqrt(vx*vy)


def quantile_bins(vals,q=4):
    s=sorted(vals)
    cuts=[]
    for i in range(1,q):
        idx=min(len(s)-1,max(0,int(i*len(s)/q)))
        cuts.append(s[idx])
    return cuts


def univariate(rows):
    y=[r[TARGET] for r in rows]
    n=len(rows)
    fold_size=max(1,n//5)
    out=[]
    for c in CORE13:
        x=[r[c] for r in rows]
        a=auc(y,x)
        corr=pearson(x,y)
        cuts=quantile_bins(x,4)
        qwr=[]
        for qi in range(4):
            lo=-1e99 if qi==0 else cuts[qi-1]
            hi=1e99 if qi==3 else cuts[qi]
            idx=[i for i,v in enumerate(x) if (v>=lo and v<=hi if qi in (0,3) else v>lo and v<=hi)]
            if not idx: qwr.append(None)
            else: qwr.append(sum(y[i] for i in idx)/len(idx))
        fold_aucs=[]
        for k in range(1,5):
            t0=k*fold_size
            if t0>=n: break
            ys=y[t0:]; xs=x[t0:]
            aa=auc(ys,xs)
            if aa is not None: fold_aucs.append(aa)
        stdev=statistics.pstdev(fold_aucs) if len(fold_aucs)>=2 else None
        strength=abs((a or 0.5)-0.5)
        out.append({"feature":c,"auc":a,"corr":corr,"quartile_winrate":qwr,"fold_auc_std":stdev,"strength":strength})
    out.sort(key=lambda d:d["strength"],reverse=True)
    return out


def corr_matrix(rows, feats):
    m={f:{} for f in feats}
    cols={f:[r[f] for r in rows] for f in feats}
    for a in feats:
        for b in feats:
            m[a][b]=pearson(cols[a],cols[b])
    return m


def select_decorrelated(rows, ranked, thr=0.70):
    feats=[r["feature"] for r in ranked]
    cm=corr_matrix(rows, feats)
    kept=[]; dropped=[]
    for f in feats:
        hit=None
        for k in kept:
            if abs(cm[f][k])>thr:
                hit=k; break
        if hit is None: kept.append(f)
        else: dropped.append({"feature":f,"dropped_by":hit,"corr":cm[f][hit]})
    return kept,dropped,cm


def standardize_fit(X):
    n=len(X); m=len(X[0])
    mu=[sum(X[i][j] for i in range(n))/n for j in range(m)]
    sd=[]
    for j in range(m):
        v=sum((X[i][j]-mu[j])**2 for i in range(n))/n
        s=math.sqrt(v) if v>1e-12 else 1.0
        sd.append(s)
    Xs=[[ (X[i][j]-mu[j])/sd[j] for j in range(m)] for i in range(n)]
    return Xs,mu,sd


def standardize_apply(X,mu,sd):
    return [[(row[j]-mu[j])/sd[j] for j in range(len(mu))] for row in X]


def sigmoid(z):
    if z>35:return 1.0
    if z<-35:return 0.0
    return 1/(1+math.exp(-z))


def train_logreg(X,y,l2=1.0,lr=0.08,epochs=120):
    Xs,mu,sd=standardize_fit(X)
    n=len(Xs); m=len(Xs[0])
    w=[0.0]*m; b=0.0
    for _ in range(epochs):
        gw=[0.0]*m; gb=0.0
        for i in range(n):
            z=b+sum(w[j]*Xs[i][j] for j in range(m))
            p=sigmoid(z)
            e=p-y[i]
            gb+=e
            for j in range(m): gw[j]+=e*Xs[i][j]
        for j in range(m):
            gw[j]=gw[j]/n + l2*w[j]/n
            w[j]-=lr*gw[j]
        b-=lr*(gb/n)
    return {"w":w,"b":b,"mu":mu,"sd":sd,"family":"logreg_stdlib"}


def pred_logreg(model,X):
    Xs=standardize_apply(X,model["mu"],model["sd"])
    out=[]
    for row in Xs:
        z=model["b"]+sum(model["w"][j]*row[j] for j in range(len(row)))
        out.append(sigmoid(z))
    return out


def train_stump(X,y):
    n=len(X); m=len(X[0])
    best={"gini":9e9}
    for j in range(m):
        vals=[row[j] for row in X]
        svals=sorted(vals)
        if len(svals)<4:
            continue
        # candidatos por cuantiles (rápido y estable)
        cand=[]
        for q in (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9):
            idx=min(len(svals)-1,max(0,int(q*(len(svals)-1))))
            cand.append(svals[idx])
        for t in sorted(set(cand)):
            l_n=l_pos=r_n=r_pos=0
            for i,v in enumerate(vals):
                if v<=t:
                    l_n+=1; l_pos+=y[i]
                else:
                    r_n+=1; r_pos+=y[i]
            if l_n==0 or r_n==0:
                continue
            p_l=l_pos/l_n; p_r=r_pos/r_n
            g_l=1-p_l*p_l-(1-p_l)*(1-p_l)
            g_r=1-p_r*p_r-(1-p_r)*(1-p_r)
            gini=(l_n/n)*g_l+(r_n/n)*g_r
            if gini<best["gini"]:
                best={"feature_idx":j,"thr":t,"pl":p_l,"pr":p_r,"gini":gini,"family":"stump_tree"}
    if "feature_idx" not in best:
        p=sum(y)/len(y)
        return {"feature_idx":0,"thr":0.0,"pl":p,"pr":p,"gini":1.0,"family":"stump_tree"}
    return best


def pred_stump(model,X):
    j=model["feature_idx"]; t=model["thr"]
    return [model["pl"] if row[j]<=t else model["pr"] for row in X]


def metrics(y,p):
    n=len(y)
    if n==0:return {}
    a=auc(y,p)
    acc=sum((1 if p[i]>=0.5 else 0)==y[i] for i in range(n))/n
    brier=sum((p[i]-y[i])**2 for i in range(n))/n
    eps=1e-12
    ll=-sum(y[i]*math.log(max(eps,min(1-eps,p[i])))+(1-y[i])*math.log(max(eps,min(1-eps,1-p[i]))) for i in range(n))/n
    return {"auc":a,"accuracy":acc,"brier":brier,"logloss":ll}


def tscv_indices(n,folds=5,test_frac=0.15):
    test=max(1,int(n*test_frac))
    out=[]
    for k in range(folds):
        train_end=max(1,int(n*(0.35+0.1*k)))
        test_end=min(n,train_end+test)
        if test_end-train_end<1: continue
        out.append((list(range(train_end)),list(range(train_end,test_end))))
    return out


def evaluate_model(rows, feats, trainer, predictor, name):
    n=len(rows)
    split=max(1,int(n*0.8))
    tr=rows[:split]; te=rows[split:]
    Xtr=[[r[f] for f in feats] for r in tr]; ytr=[r[TARGET] for r in tr]
    Xte=[[r[f] for f in feats] for r in te]; yte=[r[TARGET] for r in te]
    model=trainer(Xtr,ytr)
    pte=predictor(model,Xte)
    hold=metrics(yte,pte)
    folds=tscv_indices(n,folds=5,test_frac=0.15)
    fm=[]
    for tri,tei in folds:
        trr=[rows[i] for i in tri]; tee=[rows[i] for i in tei]
        X1=[[r[f] for f in feats] for r in trr]; y1=[r[TARGET] for r in trr]
        X2=[[r[f] for f in feats] for r in tee]; y2=[r[TARGET] for r in tee]
        m=trainer(X1,y1)
        p=predictor(m,X2)
        fm.append(metrics(y2,p))
    aucs=[d["auc"] for d in fm if d.get("auc") is not None]
    summary={
        "name":name,"features":feats,"rows_train":len(tr),"rows_test":len(te),
        "holdout":hold,
        "folds":fm,
        "auc_mean_folds":(sum(aucs)/len(aucs) if aucs else None),
        "auc_std_folds":(statistics.pstdev(aucs) if len(aucs)>=2 else None),
        "model":model,
        "holdout_probs":pte,
        "holdout_y":yte,
    }
    return summary


def high_bands(y,p):
    bands=[(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,1.01)]
    out=[]; n=len(y)
    for lo,hi in bands:
        idx=[i for i,v in enumerate(p) if v>=lo and v<hi]
        cov=len(idx)/n if n else 0
        wr=(sum(y[i] for i in idx)/len(idx)) if idx else None
        out.append({"band":f"[{lo:.2f},{hi:.2f})","n":len(idx),"coverage":cov,"win_rate":wr})
    return out


def audit_signals_log(path):
    if not os.path.exists(path): return {"exists":False}
    with open(path,newline='',encoding='utf-8',errors='replace') as f:
        rdr=csv.DictReader(f)
        rows=list(rdr)
        cols=rdr.fieldnames or []
    useful_prob=next((c for c in ["ia_prob_en_juego","prob_ia_oper","prob_ia"] if c in cols),None)
    useful_res=next((c for c in ["resultado","result_bin"] if c in cols),None)
    return {"exists":True,"rows":len(rows),"columns":len(cols),"prob_col":useful_prob,"result_col":useful_res,
            "usable":bool(useful_prob and useful_res and len(rows)>0)}


def try_load_current_model(path):
    if not os.path.exists(path): return {"exists":False}
    try:
        with open(path,'rb') as f:
            pickle.load(f)
        return {"exists":True,"loadable":True}
    except Exception as e:
        return {"exists":True,"loadable":False,"error":f"{type(e).__name__}: {e}"}


def main():
    report={"seed":SEED,"generated_at":datetime.utcnow().isoformat()+"Z"}
    rows_v3,info_v3=load_rows(PRIMARY_DATASET)
    report["dataset_v3_info"]=info_v3
    report["dataset_policy"]={
        "strict_primary": bool(STRICT_PRIMARY_DATASET),
        "primary": PRIMARY_DATASET,
        "contrast": CONTRAST_DATASET,
    }
    rows_main=[]
    primary_usable = bool(rows_v3)
    if primary_usable:
        rows_main=rows_v3
        report["dataset_used"]=PRIMARY_DATASET
    else:
        report["dataset_used"]=None
        report["primary_dataset_error"]=(
            "missing_or_unusable_primary_dataset"
            if not info_v3.get("exists", False)
            else "primary_dataset_loaded_zero_rows"
        )

    # Carga de contraste secundaria (nunca principal si strict_primary=True)
    rows_c,info_c=load_rows(CONTRAST_DATASET)
    report["contrast_dataset_info"]=info_c
    if (not primary_usable) and (not STRICT_PRIMARY_DATASET):
        rows_main=rows_c
        report["dataset_used"]=CONTRAST_DATASET

    n=len(rows_main)
    if n == 0:
        report["audit"]={"rows":0,"error":"no_usable_rows_in_primary_dataset"}
        report["univariate_rank"]=[]
        report["multicollinearity"]={"threshold":0.70,"kept":[],"dropped":[],"groups_hint":{}}
        report["models"]={}
        report["current_model_eval"]={"status":try_load_current_model("modelo_xgb_v2.pkl"),"note":"Modelo actual no evaluado por falta de dataset primario usable"}
        report["best_model_name"]=None
        report["best_model_bands"]=[]
        report["promotion_decision"]={"promote":False,"reason":"Sin dataset primario usable"}
        report["ia_signals_log"]=audit_signals_log("ia_signals_log.csv")
        os.makedirs("tmp_validation",exist_ok=True)
        out_json="tmp_validation/retrain_v3_report.json"
        report["generated_artifacts"]=[]
        with open(out_json,'w',encoding='utf-8') as f:
            json.dump(report,f,ensure_ascii=False,indent=2)
        print(out_json)
        return
    report["audit"]={
        "rows":n,
        "columns_expected":len(CORE13)+1,
        "duplicates_exact":exact_duplicates(rows_main),
        "class_balance":class_balance(rows_main),
        "nan_or_inf_rows":0,
    }
    shp,counts=scalping_health(rows_main)
    report["audit"]["scalping_defaults"]=shp
    report["audit"]["scalping_real_counts"]=counts
    report["audit"]["feature_variance"]={c:var([r[c] for r in rows_main]) for c in CORE13}

    u=univariate(rows_main)
    report["univariate_rank"]=u
    kept,dropped,cm=select_decorrelated(rows_main,u,thr=0.70)
    report["multicollinearity"]={"threshold":0.70,"kept":kept,"dropped":dropped,
                                  "groups_hint":{
                                      "momentum":[f for f in ["ret_1m","ret_3m","ret_5m","slope_5m","micro_trend_persist","bb_z"] if f in kept or any(d['feature']==f for d in dropped)]
                                  }}

    top3=kept[:3] if len(kept)>=3 else kept
    m1=evaluate_model(rows_main, kept, train_logreg, pred_logreg, "logreg_decorrelated")
    m2=evaluate_model(rows_main, kept, train_stump, pred_stump, "stump_tree_decorrelated")
    m3=evaluate_model(rows_main, top3, train_logreg, pred_logreg, "logreg_min_interpretable")

    report["models"]={k:{kk:vv for kk,vv in m.items() if kk not in ("model","holdout_probs","holdout_y")} for k,m in {
        "A_logreg":m1,"B_tree":m2,"C_min":m3}.items()}

    current=try_load_current_model("modelo_xgb_v2.pkl")
    report["current_model_eval"]={"status":current,"note":"Sin sklearn/xgboost en entorno, no evaluable de forma justa" if not current.get("loadable") else "cargable"}

    candidates=[m1,m2,m3]
    best=max(candidates,key=lambda m:(m["auc_mean_folds"] if m["auc_mean_folds"] is not None else -1))
    report["best_model_name"]=best["name"]
    report["best_model_bands"]=high_bands(best["holdout_y"],best["holdout_probs"])

    stable=(best["auc_mean_folds"] is not None and best["auc_mean_folds"]>=0.55 and (best["auc_std_folds"] or 9)<=0.03)
    report["promotion_decision"]={"promote":False,"reason":"Aún frágil" if not stable else "Mejora candidata"}

    report["ia_signals_log"]=audit_signals_log("ia_signals_log.csv")

    os.makedirs("tmp_validation",exist_ok=True)
    out_json="tmp_validation/retrain_v3_report.json"
    with open(out_json,'w',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)

    # Guardado artefactos solo si hay v3 y mejora estable
    generated=[]
    if primary_usable and stable:
        stamp=datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_path=f"modelo_logreg_v3_{stamp}.json"
        meta_path=f"model_meta_v3_{stamp}.json"
        fn_path=f"feature_names_v3_{stamp}.json"
        with open(model_path,'w',encoding='utf-8') as f:
            json.dump({"family":best["model"]["family"],"model":best["model"],"features":best["features"]},f,ensure_ascii=False,indent=2)
        with open(fn_path,'w',encoding='utf-8') as f:
            json.dump(best["features"],f,ensure_ascii=False,indent=2)
        meta={
            "rows_total":n,"rows_train":best["rows_train"],"rows_test":best["rows_test"],"n_samples":n,
            "model_family":best["model"]["family"],"auc_holdout":best["holdout"].get("auc"),
            "auc_folds_mean":best["auc_mean_folds"],"auc_folds_std":best["auc_std_folds"],
            "reliable": bool(stable),"warmup_mode": not bool(stable),"features_finales":best["features"],
            "trained_at":datetime.utcnow().isoformat()+"Z","seed":SEED,
        }
        with open(meta_path,'w',encoding='utf-8') as f:
            json.dump(meta,f,ensure_ascii=False,indent=2)
        generated=[model_path,fn_path,meta_path]
    report["generated_artifacts"]=generated
    with open(out_json,'w',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(out_json)

if __name__=='__main__':
    main()
