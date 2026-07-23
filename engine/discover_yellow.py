# -*- coding: utf-8 -*-
"""
ОТКРЫТИЕ (не проверка гипотезы). Цель: НЕ приносить готовую идею, а ИЗУЧИТЬ,
что цена делает после жёлтой свечи, и дать данным подсказать правила.
Участок ОТКРЫТИЙ: 2017-2020 (изучаю). ТЕСТ 2021-2026 заперт до конца.
Шаг 1: есть ли вообще дрейф после жёлтой свечи? И как он зависит от
простейших свойств самой свечи (тело, фитили, бычья/медвежья, размер)?
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'engine')
from reference_v5 import load

df = load('data/gold/xau_1h_master.csv')
t = pd.to_datetime(df['time'], utc=True)
disc = (t >= '2017-01-01') & (t < '2021-01-01')   # участок открытий
H = df['high'].values; L = df['low'].values; O = df['open'].values; C = df['close'].values
n = len(df)
trig = np.where((df['trigger'].values == 1) & disc.values)[0]
trig = trig[trig < n - 30]   # чтобы был горизонт вперёд
print(f"Жёлтых свечей на участке открытий 2017-2020: {len(trig)}")

# признаки жёлтой свечи
rng = H - L
body = np.abs(C - O)
bull = (C > O).astype(int)
upw = np.where(C >= O, H - C, H - O)
loww = np.where(C >= O, O - L, C - L)
clpos = np.where(rng > 0, (C - L) / rng, 0.5)   # где закрытие в диапазоне (0 низ, 1 верх)
sz = rng / C * 100

# форвардные доходности (close→close, в %), несколько горизонтов
def fwd(h):
    out = np.full(n, np.nan)
    for i in trig:
        out[i] = (C[i + h] - C[i]) / C[i] * 100
    return out

print("\n1) БАЗОВЫЙ ДРЕЙФ после жёлтой свечи (среднее/медиана форв. доходности, %):")
print(f"{'гор':>4s} {'среднее':>9s} {'медиана':>9s} {'дол.>0':>7s}  (если ~0 и ~50% — направления нет)")
for h in (1, 3, 6, 12, 24):
    f = fwd(h)[trig]
    print(f"{h:4d} {np.mean(f):+8.4f}% {np.median(f):+8.4f}% {(f>0).mean()*100:6.1f}%")

# условный дрейф по свойствам свечи: делим на группы, смотрим форвард на h=6
h = 6
f6 = fwd(h)[trig]
def cond(name, arr, bins):
    a = arr[trig]
    print(f"\n{name} → форвард {h} баров (среднее %, доля>0, n):")
    cats = pd.cut(a, bins) if not isinstance(bins, list) or not isinstance(bins[0], str) else None
    q = pd.qcut(a, bins, duplicates='drop') if isinstance(bins, int) else pd.cut(a, bins)
    g = pd.DataFrame({'x': q, 'f': f6})
    for name2, sub in g.groupby('x', observed=True):
        print(f"  {str(name2):>18s}: {sub['f'].mean():+.4f}%  >0 {(sub['f']>0).mean()*100:4.0f}%  n={len(sub)}")

cond("Бычья(1)/медвежья(0)", bull, [-0.5,0.5,1.5])
cond("Позиция закрытия в диапазоне (0 низ..1 верх)", clpos, [0,0.2,0.4,0.6,0.8,1.0])
cond("Размер свечи, % (квинтили)", sz, 5)
cond("Верхний фитиль / диапазон", np.where(rng>0,upw/rng,0), [0,0.2,0.4,0.6,1.0])
cond("Нижний фитиль / диапазон", np.where(rng>0,loww/rng,0), [0,0.2,0.4,0.6,1.0])
