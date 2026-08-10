# -*- coding: utf-8 -*-
"""ЗАМОРОЖЕННОЕ ПРАВИЛО (выведено на открытиях 2017-2020, НЕ меняется):
  Триггер: жёлтая свеча.
  Сигнал (ФЕЙД позиции закрытия в диапазоне свечи):
    закрытие в нижних 30% диапазона → ЛОНГ (ждём отскок вверх)
    закрытие в верхних 30% → ШОРТ (ждём откат вниз)
    середина 30-70% → пропуск.
  Стоп = 1× диапазон свечи, тейк = 1.5× (R:R 1.5). Горизонт 24 бара.
  Все часы. Вход по закрытию жёлтой свечи.
ОДИН прогон на запертом тесте 2021-2026."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,'engine')
from reference_v5 import load

THR, RR, HB = 0.30, 1.5, 24

df=load('data/gold/xau_1h_master.csv')
t=pd.to_datetime(df['time'],utc=True)
H=df['high'].values;L=df['low'].values;O=df['open'].values;C=df['close'].values
n=len(df); rng=H-L; clpos=np.where(rng>0,(C-L)/rng,0.5)

def trade(i,dirn):
    en=C[i]; risk=rng[i]
    sl=en-dirn*risk; tp=en+dirn*RR*risk
    for j in range(i+1,min(i+HB+1,n)):
        hs=L[j]<=sl if dirn>0 else H[j]>=sl
        ht=H[j]>=tp if dirn>0 else L[j]<=tp
        if hs: return -1.0
        if ht: return RR
    return 0.0

def backtest(mask):
    trig=np.where((df['trigger'].values==1)&mask)[0]; trig=trig[trig<n-HB-1]
    rows=[]
    for i in trig:
        cp=clpos[i]; dirn = -1 if cp>=1-THR else (1 if cp<=THR else 0)
        if dirn==0: continue
        r=trade(i,dirn)
        if r==0: continue
        rows.append((t.iloc[i],dirn,r))
    return pd.DataFrame(rows,columns=['dt','dir','R'])

def stats(tr,name):
    wk=tr.set_index('dt')['R'].groupby(pd.Grouper(freq='W')).sum()
    wk=wk[wk!=0]
    yr=tr.set_index('dt')['R'].groupby(pd.Grouper(freq='YS')).sum()
    print(f"\n{name}: {len(tr)} сделок | {tr['R'].sum():+.1f}R | WR {(tr['R']>0).mean()*100:.1f}%")
    print(f"  недель {len(wk)} | плюсовых {(wk>0).mean()*100:.0f}% | ср.нед {wk.mean():+.2f}R | худшая {wk.min():+.1f}R")
    print("  по годам: " + " ".join(f"{y.year}:{v:+.0f}" for y,v in yr.items()))
    return tr,wk

disc=((t>='2017-01-01')&(t<'2021-01-01')).values
test=((t>='2021-01-01')).values
tr_d,_=stats(backtest(disc),"ОТКРЫТИЯ 2017-2020 (где выводил правило)")
tr_t,wk_t=stats(backtest(test),"ТЕСТ 2021-2026 (ЗАПЕРТ, первый прогон)")

# контроль случайностью на тесте: случайное направление
rng_=np.random.default_rng(0)
trig=np.where((df['trigger'].values==1)&test)[0]; trig=trig[trig<n-HB-1]
sig=[i for i in trig if clpos[i]>=1-THR or clpos[i]<=THR]
sims=[]
for s in range(2000):
    tot=0.0
    for i in sig:
        d=rng_.choice([-1,1]); r=trade(i,d)
        tot+=r
    sims.append(tot)
sims=np.array(sims); real=tr_t['R'].sum()
print(f"\nКонтроль случайностью (случайное направление на тех же свечах, 2000 сим):")
print(f"  медиана {np.median(sims):+.0f}R | 5-95% [{np.percentile(sims,5):+.0f},{np.percentile(sims,95):+.0f}] | правило {real:+.0f}R | p={np.mean(sims>=real):.3f}")
