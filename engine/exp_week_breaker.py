# -*- coding: utf-8 -*-
"""Дожим предохранителя -XR на v5: помесячная развёртка, чувствительность
к порогу, погодовая. Запуск: python3 engine/exp_week_breaker.py"""
import importlib.util
import sys
import numpy as np
import pandas as pd


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


base = 'engine/'
sys.path.insert(0, base)
e3 = load_mod('e3', base + 'engine_v3.py')
e4 = load_mod('e4', base + 'engine_v4.py')
e5 = load_mod('e5', base + 'engine_v5.py')
from exp_one_pos import trades_with_exits

E = e3.build_engine('data/gold/xau_1h_master.csv')
df = E['df']
tr5 = trades_with_exits(E, e4, e5)
tr5['exit_dt'] = df['time'].values[tr5['exit'].values]
tr5 = tr5.sort_values('i').reset_index(drop=True)
wk_of = lambda ts: pd.Timestamp(ts).strftime('%G-W%V')


def apply_breaker(thr):
    booked = {}
    kept = []
    for _, r in tr5.iterrows():
        if booked.get(wk_of(r['dt']), 0.0) <= -thr:
            continue
        kept.append(r)
        w = wk_of(r['exit_dt'])
        booked[w] = booked.get(w, 0.0) + r['R']
    return pd.DataFrame(kept)


print("1) ЧУВСТВИТЕЛЬНОСТЬ К ПОРОГУ (v5, вся история; без: 939 сделок +53.2R):")
for thr in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
    k = apply_breaker(thr)
    tt = pd.to_datetime(k['dt'], utc=True) < pd.Timestamp('2022', tz='UTC')
    print(f"   -{thr}: {len(k):3d} сделок {k['R'].sum():+7.1f}R | train {k[tt]['R'].sum():+6.1f} | test {k[~tt]['R'].sum():+6.1f}")

k3 = apply_breaker(3.0)
dropped = tr5[~tr5['i'].isin(k3['i'])]
gain = -dropped.groupby(pd.to_datetime(dropped['dt'], utc=True).dt.strftime('%Y-%m'))['R'].sum()
print(f"\n2) ПОМЕСЯЧНАЯ РАЗВЁРТКА ПРИРОСТА (-3R): всего {gain.sum():+.1f}R за {len(gain)} месяцев с блокировками")
top = gain.sort_values(ascending=False)
print(f"   Топ-5 месяцев: {', '.join(f'{m} {v:+.1f}R' for m, v in top.head(5).items())}")
print(f"   Худшие 3 (прирост < 0 = предохранитель навредил): {', '.join(f'{m} {v:+.1f}R' for m, v in top.tail(3).items())}")
print(f"   Доля топ-3 в приросте: {top.head(3).sum() / gain.sum() * 100:.0f}%")
print(f"   Месяцев с плюс-эффектом: {(gain > 0).sum()} | с минус-эффектом: {(gain < 0).sum()} | нулевых: {(gain == 0).sum()}")

print("\n3) ПОГОДОВАЯ (v5 без | v5 с -3R):")
y0 = tr5.groupby(pd.to_datetime(tr5['dt'], utc=True).dt.year)['R'].sum()
y1 = k3.groupby(pd.to_datetime(k3['dt'], utc=True).dt.year)['R'].sum()
better = 0
for y in y0.index:
    a, b = y0[y], y1.get(y, 0.0)
    better += b >= a
    print(f"   {y}: {a:+7.1f} | {b:+7.1f}  {'✓' if b >= a else '✗'}")
print(f"   Лет не хуже: {better} из {len(y0)}")
