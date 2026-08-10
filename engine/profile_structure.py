# -*- coding: utf-8 -*-
"""ПРОФИЛЬ ВОКРУГ жёлтой точки (структура/уровни, не форма свечи).
Для каждой точки — расстояния до живых уровней, режим, позиция в тренде,
метки. Тот же честный скан: перекос P(вверх) должен ПОВТОРИТЬСЯ на двух
половинах открытий (2017-18 и 2019-20). Тест 2021-26 заперт."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,'engine')
from reference_v5 import load, pivots, labels_stream

df=load('data/gold/xau_1h_master.csv')
t=pd.to_datetime(df['time'],utc=True)
H=df['high'].values;L=df['low'].values;O=df['open'].values;C=df['close'].values
n=len(df); rng=H-L
atr=pd.Series(np.maximum(H-L,np.maximum(np.abs(H-np.roll(C,1)),np.abs(L-np.roll(C,1))))).rolling(14).mean().values

# быстрые пивоты 2/2 → живые уровни HH/LL/LH/HL по мере поступления
labs=labels_stream(pivots(df))  # (conf, label, side, price)
by_conf={}
for conf,lab,side,price in labs: by_conf.setdefault(conf,[]).append((lab,price))

# симуляция: на каждом баре — списки живых уровней + режим D3
aliveHH=[]; aliveLL=[]; aliveLH=[]; aliveHL=[]
# D3
d3=0; d3lvl=np.nan; lastH=lastL=''; lHp=lLp=np.nan
REG=np.zeros(n,int)
distHH=np.full(n,np.nan); distLL=np.full(n,np.nan)
distLH=np.full(n,np.nan); distHL=np.full(n,np.nan)
lastlab=np.empty(n,object); barssince=np.full(n,np.nan)
last_pivot_bar=0
for j in range(n):
    if j in by_conf:
        for lab,price in by_conf[j]:
            last_pivot_bar=j
            if lab=='HH': aliveHH.append(price); lastH='HH'; lHp=price
            elif lab=='LH': aliveLH.append(price); lastH='LH'; lHp=price
            elif lab=='HL': aliveHL.append(price); lastL='HL'; lLp=price
            elif lab=='LL': aliveLL.append(price); lastL='LL'; lLp=price
            if d3==1 and lab=='HL': d3lvl=price
            if d3==-1 and lab=='LH': d3lvl=price
            if d3==0:
                if (lastH,lastL)==('HH','HL'): d3=1; d3lvl=lLp
                elif (lastH,lastL)==('LH','LL'): d3=-1; d3lvl=lHp
    # смерть уровней пробоем
    aliveHH=[p for p in aliveHH if H[j]<p]; aliveLH=[p for p in aliveLH if H[j]<p]
    aliveLL=[p for p in aliveLL if L[j]>p]; aliveHL=[p for p in aliveHL if L[j]>p]
    if d3==1 and not np.isnan(d3lvl) and min(O[j],C[j])<d3lvl: d3=0; d3lvl=np.nan
    elif d3==-1 and not np.isnan(d3lvl) and max(O[j],C[j])>d3lvl: d3=0; d3lvl=np.nan
    REG[j]=d3
    a=atr[j] if atr[j]>0 else np.nan
    hh_ab=[p for p in aliveHH if p>C[j]]; ll_be=[p for p in aliveLL if p<C[j]]
    lh_ab=[p for p in aliveLH if p>C[j]]; hl_be=[p for p in aliveHL if p<C[j]]
    if hh_ab: distHH[j]=(min(hh_ab)-C[j])/a
    if ll_be: distLL[j]=(C[j]-max(ll_be))/a
    if lh_ab: distLH[j]=(min(lh_ab)-C[j])/a
    if hl_be: distHL[j]=(C[j]-max(hl_be))/a
    lastlab[j]=(lastH,lastL); barssince[j]=j-last_pivot_bar

def p_up(i,Hb=12):
    en=C[i]; d=rng[i]
    if d<=0: return np.nan
    for j in range(i+1,min(i+Hb+1,n)):
        if L[j]<=en-d: return 0
        if H[j]>=en+d: return 1
    return np.nan

disc=((t>='2017-01-01')&(t<'2021-01-01')).values
trig=np.where((df['trigger'].values==1)&disc)[0]; trig=trig[(trig>100)&(trig<n-14)]
pu=np.array([p_up(i) for i in trig])
h1=(t.iloc[trig]<'2019-01-01').values; h2=~h1
b1=np.nanmean(pu[h1])*100; b2=np.nanmean(pu[h2])*100
print(f"Точек 2017-20: {len(trig)} | база P(вверх) {np.nanmean(pu)*100:.1f}% | H1 {b1:.1f}% H2 {b2:.1f}%")

F={}
F['до HH сверху(ATR)']=distHH[trig]
F['до LL снизу(ATR)']=distLL[trig]
F['до LH сверху(ATR)']=distLH[trig]
F['до HL снизу(ATR)']=distHL[trig]
F['ближ.уровень(ATR)']=np.nanmin(np.vstack([distHH[trig],distLL[trig],distLH[trig],distHL[trig]]),axis=0)
F['режим D3']=REG[trig].astype(float)
F['баров с пивота']=barssince[trig]

def scan():
    hits=[]
    for name,arr in F.items():
        a=arr
        if name=='режим D3': edges=[-1.5,-0.5,0.5,1.5]
        else:
            v=a[~np.isnan(a)]
            if len(v)<200: continue
            edges=np.unique(np.nanquantile(a,[0,.25,.5,.75,1.0]))
        cats=pd.cut(a,edges,include_lowest=True)
        for cat in cats.categories:
            m=np.asarray(cats==cat); m1=m&h1; m2=m&h2
            if m1.sum()<40 or m2.sum()<40: continue
            d1=np.nanmean(pu[m1])*100-b1; d2=np.nanmean(pu[m2])*100-b2
            if d1*d2>0:
                hits.append((min(abs(d1),abs(d2)),name,str(cat),d1,d2,int(m.sum()),
                             np.nanmean(pu[m])*100))
    return sorted(hits,reverse=True)

hits=scan()
print("\nТОП структурных бакетов (согласованы по знаку на обеих половинах):")
print(f"{'признак':20s}{'бакет':>16s}{'H1Δ':>7s}{'H2Δ':>7s}{'мин':>6s}{'P↑':>6s}{'n':>6s}")
for s,nm,c,d1,d2,nn,pa in hits[:12]:
    flag=' ✓ТОРГ' if s>=4 else ''
    print(f"{nm:20s}{c:>16s}{d1:+6.1f}{d2:+6.1f}{s:6.1f}{pa:6.0f}{nn:6d}{flag}")
strong=[h for h in hits if h[0]>=4]
print(f"\nБакетов с повтором >=4пп (торгуемо): {len(strong)}")
