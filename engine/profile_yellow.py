# -*- coding: utf-8 -*-
"""
ПРОФИЛИРОВАНИЕ жёлтых точек (идея трейдера): на каждой точке фиксируем ВСЁ,
прогоняем по всем, ищем ЧТО ПОВТОРЯЕТСЯ. Защита от обмана: бакет интересен
только если даёт тот же перекос на ДВУХ независимых половинах открытий
(2017-18 И 2019-20). Тест 2021-26 не трогаем.
Исход = P(вверх): для лонга дошло ли +d раньше -d за 12 баров (d=диапазон свечи).
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,'engine')
from reference_v5 import load

df=load('data/gold/xau_1h_master.csv')
t=pd.to_datetime(df['time'],utc=True)
H=df['high'].values;L=df['low'].values;O=df['open'].values;C=df['close'].values
n=len(df); rng=H-L
atr=pd.Series(np.maximum(H-L,np.maximum(np.abs(H-np.roll(C,1)),np.abs(L-np.roll(C,1))))).rolling(14).mean().values

def p_up(i,Hb=12):
    en=C[i]; d=rng[i]
    if d<=0: return np.nan
    for j in range(i+1,min(i+Hb+1,n)):
        if L[j]<=en-d: return 0
        if H[j]>=en+d: return 1
    return np.nan

disc=((t>='2017-01-01')&(t<'2021-01-01')).values
trig=np.where((df['trigger'].values==1)&disc)[0]; trig=trig[(trig>30)&(trig<n-14)]

# ── ПРИЗНАКИ в моменте закрытия жёлтой свечи (без подглядывания) ──
F={}
F['поз.закрытия']   = np.where(rng>0,(C-L)/rng,.5)
F['тело/диап']      = np.where(rng>0,np.abs(C-O)/rng,0)
F['верх.фитиль']    = np.where(rng>0,(H-np.maximum(O,C))/rng,0)
F['ниж.фитиль']     = np.where(rng>0,(np.minimum(O,C)-L)/rng,0)
F['бычья']          = (C>O).astype(float)
F['размер/ATR']     = np.where(atr>0,rng/atr,np.nan)
F['размер%цены']    = rng/C*100
def ret(k):
    r=np.full(n,np.nan); r[k:]=(C[k:]-C[:-k])/C[:-k]*100; return r
F['движ.1бар%']=ret(1); F['движ.6бар%']=ret(6); F['движ.24бар%']=ret(24)
# позиция в диапазоне последних 24 баров
hi24=pd.Series(H).rolling(24).max().values; lo24=pd.Series(L).rolling(24).min().values
F['поз.в диап24']=np.where(hi24>lo24,(C-lo24)/(hi24-lo24),.5)
hi72=pd.Series(H).rolling(72).max().values; lo72=pd.Series(L).rolling(72).min().values
F['поз.в диап72']=np.where(hi72>lo72,(C-lo72)/(hi72-lo72),.5)
# серия одноцветных баров
updn=np.sign(C-O); streak=np.zeros(n)
for j in range(1,n): streak[j]=streak[j-1]+updn[j] if updn[j]==updn[j-1] else updn[j]
F['серия баров']=streak
F['ATR растёт']=(atr>np.roll(atr,6)).astype(float)
F['час']=t.dt.hour.values.astype(float)
F['день недели']=t.dt.dayofweek.values.astype(float)

pu=np.array([p_up(i) for i in trig]); base=np.nanmean(pu)*100
h1=(t.iloc[trig]<'2019-01-01').values; h2=~h1
print(f"Жёлтых точек 2017-20: {len(trig)} | базовый P(вверх): {base:.1f}%")
print(f"Половины: 2017-18 n={h1.sum()} (база {np.nanmean(pu[h1])*100:.1f}%) | 2019-20 n={h2.sum()} (база {np.nanmean(pu[h2])*100:.1f}%)")
print("\nСКАН: бакеты, где перекос P(вверх) ПОВТОРЯЕТСЯ в обеих половинах (|откл|>=4пп, n>=40):")
print(f"{'признак':16s} {'бакет':>14s} {'H1 17-18':>9s} {'H2 19-20':>9s} {'вместе':>7s}  повтор")

hits=[]
for name,arr in F.items():
    a=arr[trig]
    if name in ('бычья','ATR растёт'):
        edges=[-.5,.5,1.5]
    elif name=='час':
        edges=[0,8,13,17,24]
    elif name=='день недели':
        edges=[-.5,.5,1.5,2.5,3.5,4.5,6.5]
    elif name=='серия баров':
        edges=[-99,-2.5,-0.5,0.5,2.5,99]
    else:
        try: edges=np.nanquantile(a,[0,.25,.5,.75,1.0])
        except: continue
    cats=pd.cut(a,np.unique(edges),include_lowest=True)
    for cat in cats.categories:
        m=np.asarray(cats==cat)
        m1=m&h1; m2=m&h2
        if m1.sum()<40 or m2.sum()<40: continue
        p1=np.nanmean(pu[m1])*100; p2=np.nanmean(pu[m2])*100; pa=np.nanmean(pu[m])*100
        d1=p1-np.nanmean(pu[h1])*100; d2=p2-np.nanmean(pu[h2])*100
        rep = (d1*d2>0) and abs(d1)>=4 and abs(d2)>=4
        if rep:
            hits.append((abs(d1)+abs(d2),name,str(cat),p1,p2,pa,np.sign(d1)))
            print(f"{name:16s} {str(cat):>14s} {p1:8.1f}% {p2:8.1f}% {pa:6.1f}%  {'ВВЕРХ' if d1>0 else 'ВНИЗ'} ✓")
print(f"\nВсего повторяющихся бакетов: {len(hits)} (из ~{sum(1 for _ in F)*4} проверенных)")
print("Топ-5 по силе повтора:")
for s,nm,c,p1,p2,pa,sg in sorted(hits,reverse=True)[:5]:
    print(f"  {nm} {c}: {p1:.0f}%/{p2:.0f}% → {'ЛОНГ' if sg>0 else 'ШОРТ'}-перекос")
