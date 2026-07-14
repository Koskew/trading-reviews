# -*- coding: utf-8 -*-
"""
Обязательные проверки v5 (из docs/system_v5.txt, раздел «НЕ СДЕЛАНО»):
1. Контроль случайностью пакета запретов (v4 -> v5 выбрасывает 307 сделок).
2. Помесячная развёртка v5 (не тянут ли весь плюс 2-3 месяца).
3. Walk-forward: годовая развёртка + честный (растущий) порог размера свечи
   вместо порога по всей истории (единственный подгоняемый параметр v5).
4. Бонус: каждый запрет по отдельности на обеих половинах (проверка
   утверждения доков «каждый минусовый на обеих половинах»).
Запуск: python3 engine/exp_v5_checks.py
"""
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
e3 = load_mod('e3', base + 'engine_v3.py')
e4 = load_mod('e4', base + 'engine_v4.py')
e5 = load_mod('e5', base + 'engine_v5.py')
E = e3.build_engine('data/gold/xau_1h_master.csv')
df = E['df']

# ── все сделки v4 с хвостами и отметкой запрета ──
rows = []
for i in np.where(E['TRIG'])[0]:
    d = e4.decide_v4(E, i)
    if d.get('decision') != 'ВХОД':
        continue
    o = e4.resolve(E, i, d['dir'] == 'LONG', d['stop'], d['tp'])
    if o not in ('TP', 'SL'):
        continue
    ban = e5.banned_tail(d['labels'])
    rows.append(dict(dt=df['time'].iloc[i], i=i, last=d['labels'][-1],
                     ban=ban, out=o, R=d['RR'] if o == 'TP' else -1.0))
v4t = pd.DataFrame(rows)
v5t = v4t[v4t['ban'].isna()]
banned = v4t[v4t['ban'].notna()]
print(f"v4: {len(v4t)} сделок {v4t['R'].sum():+.1f}R | "
      f"v5: {len(v5t)} сделок {v5t['R'].sum():+.1f}R | "
      f"выброшено запретами: {len(banned)} ({banned['R'].sum():+.1f}R)")

# ── 1. КОНТРОЛЬ СЛУЧАЙНОСТЬЮ ПАКЕТА ЗАПРЕТОВ ──
rng = np.random.default_rng(42)
R4 = v4t['R'].values
n_drop = len(banned)
sims = np.empty(10000)
for s in range(10000):
    drop = rng.choice(len(R4), size=n_drop, replace=False)
    mask = np.ones(len(R4), bool)
    mask[drop] = False
    sims[s] = R4[mask].sum()
real = v5t['R'].sum()
p = (sims >= real).mean()
print(f"\n1) КОНТРОЛЬ СЛУЧАЙНОСТЬЮ (10000 симуляций выброса {n_drop} случайных из v4):")
print(f"   Медиана: {np.median(sims):+.1f}R | 5-95%: "
      f"[{np.percentile(sims, 5):+.1f}, {np.percentile(sims, 95):+.1f}] | максимум: {sims.max():+.1f}R")
print(f"   Пакет запретов: {real:+.1f}R | p-value: {p:.4f}")

# ── 2. ПОМЕСЯЧНАЯ РАЗВЁРТКА v5 ──
v5m = v5t.copy()
v5m['ym'] = pd.to_datetime(v5m['dt'], utc=True).dt.strftime('%Y-%m')
mo = v5m.groupby('ym')['R'].sum().sort_values(ascending=False)
tot = v5m['R'].sum()
print(f"\n2) ПОМЕСЯЧНАЯ РАЗВЁРТКА v5 ({len(mo)} месяцев со сделками):")
print(f"   Плюсовых: {(mo > 0).sum()} | минусовых: {(mo < 0).sum()}")
print(f"   Топ-3 месяца: {', '.join(f'{k} {v:+.1f}R' for k, v in mo.head(3).items())}")
print(f"   Итог БЕЗ топ-3 месяцев: {tot - mo.head(3).sum():+.1f}R (весь итог {tot:+.1f}R)")
print(f"   Худшие 3: {', '.join(f'{k} {v:+.1f}R' for k, v in mo.tail(3).items())}")

# ── 3. WALK-FORWARD ──
v5y = v5t.copy()
v5y['y'] = pd.to_datetime(v5y['dt'], utc=True).dt.year
yr = v5y.groupby('y')['R'].agg(['sum', 'count'])
print(f"\n3а) ГОДОВАЯ РАЗВЁРТКА v5 (порог по всей истории, как движок):")
for y, r in yr.iterrows():
    print(f"   {y}: {r['sum']:+7.1f}R  ({int(r['count'])} сделок)")
print(f"   Плюсовых лет: {(yr['sum'] > 0).sum()} из {len(yr)}")

# честный растущий порог вместо порога по всей истории
sys.path.insert(0, base)
from reference_v5 import load, pivots, labels_stream, BANNED_TAILS
dfr = load('data/gold/xau_1h_master.csv')
h, l, c = dfr['high'].values, dfr['low'].values, dfr['close'].values
trig_idx = np.where(dfr['trigger'].values == 1)[0]
sz_trig = (h[trig_idx] - l[trig_idx]) / c[trig_idx] * 100
labs = labels_stream(pivots(dfr))
lab_confs = np.array([x[0] for x in labs])
rows = []
for t_ord, i in enumerate(trig_idx):
    seen = sz_trig[:t_ord + 1]
    if len(seen) < 20:
        continue
    thr = np.percentile(seen, 80)
    k = np.searchsorted(lab_confs, i, side='right')
    if k == 0:
        continue
    w = labs[max(0, k - 5):k]
    seq = [x[1] for x in w]
    lastL = seq[-1]
    is_long = lastL in ('HL', 'LL')
    if lastL in ('LL', 'HH') and (i - w[-1][0]) <= 1:
        continue
    if (h[i] - l[i]) / c[i] * 100 > thr:
        continue
    up = sum(1 for s_ in seq if s_ in ('HH', 'HL'))
    dn = sum(1 for s_ in seq if s_ in ('LH', 'LL'))
    had_slom = any((a == 'HH' and b == 'LL') or (a == 'LL' and b == 'HL')
                   for a, b in zip(seq, seq[1:]))
    if not had_slom:
        if not is_long and up >= (3 if lastL == 'HH' else 4):
            continue
        if is_long and dn >= 4:
            continue
    if len(seq) >= 2 and ((seq[-2] == 'HH' and seq[-1] == 'LL') or
                          (seq[-2] == 'LL' and seq[-1] == 'HL')):
        continue
    en = c[i]
    sl = l[i] * 0.999 if is_long else h[i] * 1.001
    if (is_long and sl >= en) or (not is_long and sl <= en):
        continue
    risk = abs(en - sl)
    cand = sorted([x[3] for x in w if x[2] == ('high' if is_long else 'low') and
                   ((is_long and x[3] > en) or (not is_long and x[3] < en))],
                  reverse=not is_long)
    tp = next((p_ for p_ in cand if abs(p_ - en) / risk >= 3), None)
    if tp is None:
        continue
    if any('>'.join(seq).endswith(b) for b in BANNED_TAILS):
        continue
    out = None
    for j in range(i + 1, min(i + 1500, len(dfr))):
        hit_sl = l[j] <= sl if is_long else h[j] >= sl
        hit_tp = h[j] >= tp if is_long else l[j] <= tp
        if hit_sl:
            out = 'SL'
            break
        if hit_tp:
            out = 'TP'
            break
    if out is None:
        continue
    rows.append(dict(dt=dfr['time'].iloc[i], y=int(str(dfr['time'].iloc[i])[:4]),
                     R=abs(tp - en) / risk if out == 'TP' else -1.0))
wf = pd.DataFrame(rows)
print(f"\n3б) WALK-FORWARD ПОРОГА (растущее окно, без подглядывания):")
print(f"   Итог: {len(wf)} сделок, {wf['R'].sum():+.1f}R "
      f"(движок с полной историей: 939, +53.2R)")
yr2 = wf.groupby('y')['R'].agg(['sum', 'count'])
for y, r in yr2.iterrows():
    print(f"   {y}: {r['sum']:+7.1f}R  ({int(r['count'])} сделок)")
print(f"   Плюсовых лет: {(yr2['sum'] > 0).sum()} из {len(yr2)}")

# ── 4. КАЖДЫЙ ЗАПРЕТ ПО ОТДЕЛЬНОСТИ НА ОБЕИХ ПОЛОВИНАХ ──
print(f"\n4) ВЫБРОШЕННЫЕ КАЖДЫМ ЗАПРЕТОМ (их суммарный R; минус = запрет полезен):")
banned2 = banned.copy()
banned2['half'] = np.where(pd.to_datetime(banned2['dt'], utc=True) <
                           pd.Timestamp('2022-01-01', tz='UTC'), 'train', 'test')
for b in e5.BANNED_TAILS:
    sub = banned2[banned2['ban'] == b]
    tr_ = sub[sub['half'] == 'train']['R'].sum()
    te_ = sub[sub['half'] == 'test']['R'].sum()
    print(f"   {b:10s}: всего {len(sub):3d} сделок | train {tr_:+7.1f}R | test {te_:+7.1f}R")
