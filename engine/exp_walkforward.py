# -*- coding: utf-8 -*-
"""Walk-forward режимного фильтра v5. На каждый тестовый год бакеты для отреза
выбираются ТОЛЬКО по прошлым данным (expanding window), применяются вслепую.
Проверяем: стабильность выбора + OOS-итог vs база. python3 engine/exp_walkforward.py"""
import importlib.util, sys
import numpy as np
import pandas as pd

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

base = 'engine/'
sys.path.insert(0, base)
e3 = load_mod('e3', base+'engine_v3.py'); e4 = load_mod('e4', base+'engine_v4.py'); e5 = load_mod('e5', base+'engine_v5.py')
from exp_one_pos import trades_with_exits
from reference_v5 import load, labels_stream

df = load('data/gold/xau_1h_master.csv')
H, L, O, C = df['high'].values, df['low'].values, df['open'].values, df['close'].values
n = len(df)
def piv(a,b):
    out=[]
    for i in range(a,n-b):
        if H[i]>H[i-a:i].max() and H[i]>H[i+1:i+b+1].max(): out.append((i+b,i,'high',H[i]))
        if L[i]<L[i-a:i].min() and L[i]<L[i+1:i+b+1].min(): out.append((i+b,i,'low',L[i]))
    out.sort(key=lambda x:(x[0],x[1])); return out
labs = labels_stream(piv(5,5))
reg=np.zeros(n,int); lastH=lastL=''; lvl=np.nan; st=0; li=0; lHp=lLp=np.nan
for j in range(n):
    while li<len(labs) and labs[li][0]<=j:
        _,lab,side,price=labs[li]
        if side=='high': lastH=lab; lHp=price
        else: lastL=lab; lLp=price
        if st==1 and lab=='HL': lvl=price
        if st==-1 and lab=='LH': lvl=price
        if st==0:
            if (lastH,lastL)==('HH','HL'): st=1; lvl=lLp
            elif (lastH,lastL)==('LH','LL'): st=-1; lvl=lHp
        li+=1
    if st==1 and not np.isnan(lvl) and min(O[j],C[j])<lvl: st=0; lvl=np.nan
    elif st==-1 and not np.isnan(lvl) and max(O[j],C[j])>lvl: st=0; lvl=np.nan
    reg[j]=st

E = e3.build_engine('data/gold/xau_1h_master.csv')
tr = trades_with_exits(E, e4, e5)
tr['reg']=reg[tr['i'].values]
tr['is_long']=tr['dir']=='LONG'
tr['y']=pd.to_datetime(tr['dt'],utc=True).dt.year
tr['bucket']=list(zip(tr['reg'], tr['is_long']))

BUCKETS=[(1,True),(1,False),(-1,True),(-1,False),(0,True),(0,False)]
NAME={(1,True):'ВОСХ+LONG',(1,False):'ВОСХ+SHORT',(-1,True):'НИСХ+LONG',
      (-1,False):'НИСХ+SHORT',(0,True):'БОК+LONG',(0,False):'БОК+SHORT'}

print("WALK-FORWARD (expanding window; отрез = бакет с суммой R<0 и >=15 сделок в прошлом):")
print(f"{'год':4s} {'база OOS':>9s} {'фильтр OOS':>11s}  отрезанные в этом году (выбрано по прошлому)")
b_tot=f_tot=0.0
for Y in range(2019,2027):
    past=tr[tr['y']<Y]; oos=tr[tr['y']==Y]
    if len(oos)==0: continue
    cut=set()
    for bk in BUCKETS:
        s=past[past['bucket']==bk]
        if len(s)>=15 and s['R'].sum()<0:
            cut.add(bk)
    base_oos=oos['R'].sum()
    filt_oos=oos[~oos['bucket'].isin(cut)]['R'].sum()
    b_tot+=base_oos; f_tot+=filt_oos
    cutnames=', '.join(NAME[b] for b in sorted(cut, key=lambda x:BUCKETS.index(x))) or '—'
    print(f"{Y:4d} {base_oos:+8.1f}R {filt_oos:+9.1f}R  {cutnames}")
print(f"ИТОГ OOS 2019-26: база {b_tot:+.1f}R | фильтр {f_tot:+.1f}R | разница {f_tot-b_tot:+.1f}R")

# стабильность: как часто каждый бакет попадал в отрез
print("\nСтабильность выбора (в скольких из 8 лет бакет резался):")
cnt={bk:0 for bk in BUCKETS}
for Y in range(2019,2027):
    past=tr[tr['y']<Y]
    for bk in BUCKETS:
        s=past[past['bucket']==bk]
        if len(s)>=15 and s['R'].sum()<0: cnt[bk]+=1
for bk in BUCKETS:
    bar='█'*cnt[bk]
    print(f"  {NAME[bk]:11s}: {cnt[bk]}/8 {bar}")

print("\n═══ ЕДИНСТВЕННЫЙ ВЫЖИВШИЙ: отрез только БОК+LONG (8/8 стабильность) ═══")
print(f"{'год':4s} {'база OOS':>9s} {'БОК+LONG OOS':>13s} {'фильтр OOS':>11s}")
b=f=0.0
for Y in range(2019,2027):
    oos=tr[tr['y']==Y]
    if len(oos)==0: continue
    bl=oos[(oos['reg']==0)&oos['is_long']]['R'].sum()
    base_oos=oos['R'].sum(); filt=oos[~((oos['reg']==0)&oos['is_long'])]['R'].sum()
    b+=base_oos; f+=filt
    print(f"{Y:4d} {base_oos:+8.1f}R {bl:+12.1f}R {filt:+9.1f}R")
print(f"ИТОГ OOS: база {b:+.1f}R | только-БОК+LONG-отрез {f:+.1f}R | разница {f-b:+.1f}R")
# полная история для справки
full=tr[~((tr['reg']==0)&tr['is_long'])]
htr=pd.to_datetime(full['dt'],utc=True)<pd.Timestamp('2022',tz='UTC')
print(f"Полная история: {len(full)} сд, {full['R'].sum():+.1f}R (train {full[htr]['R'].sum():+.0f}/test {full[~htr]['R'].sum():+.0f})")
