# -*- coding: utf-8 -*-
"""Дожим: v5 + режим D3. Торговать ПО тренду или ПРОТИВ?
D3 = липкое 5/5, смерть телом. Матрица режим×направление, затем фильтры.
Дисциплина: смотрим train, вывод по test. Запуск: python3 engine/exp_regime_filter.py"""
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

def pivots_ab(a, b):
    out = []
    for i in range(a, n-b):
        if H[i] > H[i-a:i].max() and H[i] > H[i+1:i+b+1].max(): out.append((i+b,i,'high',H[i]))
        if L[i] < L[i-a:i].min() and L[i] < L[i+1:i+b+1].min(): out.append((i+b,i,'low',L[i]))
    out.sort(key=lambda x:(x[0],x[1])); return out
labs = labels_stream(pivots_ab(5,5))

# D3 режим на каждом баре (липкое, смерть телом)
reg = np.zeros(n, int)
lastH=lastL=''; lvl=np.nan; state=0; li=0
lastHp=lastLp=np.nan
for j in range(n):
    while li < len(labs) and labs[li][0] <= j:
        _, lab, side, price = labs[li]
        if side=='high': lastH=lab; lastHp=price
        else: lastL=lab; lastLp=price
        if state==1 and lab=='HL': lvl=price
        if state==-1 and lab=='LH': lvl=price
        if state==0:
            if (lastH,lastL)==('HH','HL'): state=1; lvl=lastLp
            elif (lastH,lastL)==('LH','LL'): state=-1; lvl=lastHp
        li+=1
    if state==1 and not np.isnan(lvl) and min(O[j],C[j])<lvl: state=0; lvl=np.nan
    elif state==-1 and not np.isnan(lvl) and max(O[j],C[j])>lvl: state=0; lvl=np.nan
    reg[j]=state

E = e3.build_engine('data/gold/xau_1h_master.csv')
tr = trades_with_exits(E, e4, e5)
tr['reg'] = reg[tr['i'].values]
tr['is_long'] = tr['dir'] == 'LONG'
half = pd.to_datetime(tr['dt'], utc=True) < pd.Timestamp('2022', tz='UTC')

def rr(sub):
    a = sub[half.reindex(sub.index, fill_value=False)]['R'].sum()
    b = sub[~half.reindex(sub.index, fill_value=True)]['R'].sum()
    return f"{len(sub):3d} сд {sub['R'].sum():+6.1f}R (tr {a:+.0f}/te {b:+.0f})"

print("МАТРИЦА режим×направление (всё | train/test):")
for rv, rn in ((1,'ВОСХ'),(-1,'НИСХ'),(0,'БОК ')):
    for lg, ln in ((True,'LONG '),(False,'SHORT')):
        s = tr[(tr['reg']==rv)&(tr['is_long']==lg)]
        note = ''
        if rv==1 and lg: note='← по тренду'
        if rv==1 and not lg: note='← ПРОТИВ (шорт в аптренде)'
        if rv==-1 and not lg: note='← по тренду'
        if rv==-1 and lg: note='← ПРОТИВ (лонг дна = кормилец v5?)'
        print(f"  {rn} {ln}: {rr(s)}  {note}")

base_r = tr['R'].sum()
print(f"\nБАЗА v5: {rr(tr)}")

# Фильтры
def f_report(mask, name):
    s = tr[mask]
    print(f"{name}: {rr(s)}")

# 1) ПО тренду: восх→только лонг, нисх→только шорт, бок→оба
m_with = ((tr['reg']==1)&tr['is_long']) | ((tr['reg']==-1)&~tr['is_long']) | (tr['reg']==0)
f_report(m_with, "1) ПО тренду (бок=оба)          ")
# 2) ПО тренду строго: бок пропускаем
m_with_s = ((tr['reg']==1)&tr['is_long']) | ((tr['reg']==-1)&~tr['is_long'])
f_report(m_with_s, "2) ПО тренду строго (бок=скип)  ")
# 3) убрать только контртренд (оставить бок оба + попутные)
m_nocounter = ~(((tr['reg']==1)&~tr['is_long']) | ((tr['reg']==-1)&tr['is_long']))
f_report(m_nocounter, "3) без контртренда (бок=оба)    ")
# 4) ПРОТИВ тренда (разворотная логика v5 в чистом виде)
m_against = ((tr['reg']==1)&~tr['is_long']) | ((tr['reg']==-1)&tr['is_long']) | (tr['reg']==0)
f_report(m_against, "4) ПРОТИВ тренда (бок=оба)      ")
# 5) только боковик
f_report(tr['reg']==0, "5) только боковик              ")

print("\n═══ DATA-DRIVEN ОТСЕЧЕНИЕ (бакеты, минусовые на ОБЕИХ половинах) ═══")
# кандидаты на отрез: НИСХ+SHORT (tr -1/te -13), БОК+LONG (tr -6/te -27)
drop1 = (tr['reg']==-1)&~tr['is_long']   # нисх шорт
drop2 = (tr['reg']==0)&tr['is_long']     # бок лонг
f_report(~drop2, "убрать БОК+LONG                ")
f_report(~drop1, "убрать НИСХ+SHORT              ")
f_report(~(drop1|drop2), "убрать оба                     ")
# контроль случайностью для «убрать оба»
kept = tr[~(drop1|drop2)]
rng = np.random.default_rng(1)
R = tr['R'].values; nd = len(tr)-len(kept)
sims = np.array([R[np.random.default_rng(s).permutation(len(R))[:len(R)-nd]].sum() for s in range(3000)])
print(f"  контроль случайностью (выброс {nd} случайных, 3000): медиана {np.median(sims):+.1f}R, p95 {np.percentile(sims,95):+.1f}, реально {kept['R'].sum():+.1f}R, p={np.mean(sims>=kept['R'].sum()):.3f}")
# помесячно для «убрать оба»
dr = tr[drop1|drop2]
gain = -dr.groupby(pd.to_datetime(dr['dt'],utc=True).dt.strftime('%Y-%m'))['R'].sum()
top = gain.sort_values(ascending=False)
print(f"  прирост по месяцам: топ-3 = {top.head(3).sum()/gain.sum()*100:.0f}% от {gain.sum():+.1f}R | плюс-мес {int((gain>0).sum())}/минус-мес {int((gain<0).sum())}")
