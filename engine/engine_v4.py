# -*- coding: utf-8 -*-
"""
ДВИЖОК v4 (=v3d) — решающая логика поверх engine_v3.py
Использование:
    import importlib.util
    spec = importlib.util.spec_from_file_location("e3", "engine_v3.py")
    e3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e3)
    spec4 = importlib.util.spec_from_file_location("e4", "engine_v4.py")
    e4 = importlib.util.module_from_spec(spec4); spec4.loader.exec_module(e4)
    E = e3.build_engine('xau_1h_master.csv')
    trades = e4.run_v4(E)   # DataFrame сделок

ОТЛИЧИЕ v4 ОТ v3: порог согласованности против тренда = 3/5 ТОЛЬКО для шортов
от HH (в v3 было 4/5 для всего). Остальная логика идентична v3.
Итог v4 на золоте 9 лет: 1246 сделок, -38.5R (train -20.3 / test -18.2).
Кандидат v5 (НЕ зашит): не шортить HH, если прямо перед ней метка LL (отскок из низов).
"""
import numpy as np
import pandas as pd
import bisect

HORIZON = 1500  # баров на резолв сделки


def decide_v4(E, i):
    """Решение движка v4 на триггерном баре i.
    Возвращает dict: decision ('ВХОД' или причина пропуска), dir, entry, stop, tp, RR, labels.
    """
    df = E['df']; confs = E['confs']; fast = E['fast']
    csize = E['csize']; C = E['C']; H = E['H']; L = E['L']
    size_thr = E['size_thr_pct']

    k = bisect.bisect_right(confs, i)
    seq = fast[max(0, k - 5):k]
    if not seq:
        return dict(decision='нет меток')
    labels = [r['label'] for r in seq]
    last = labels[-1]
    if last in ('HL', 'LL'):
        is_long = True
    elif last in ('LH', 'HH'):
        is_long = False
    else:
        return dict(decision='метка?')
    out = dict(dir='LONG' if is_long else 'SHORT', labels=labels)
    entry = C[i]
    lag = i - seq[-1]['conf']

    # 1) заморозка свежего экстремума (LL/HH, лаг 0-1)
    if last in ('LL', 'HH') and lag <= 1:
        out['decision'] = 'заморозка'; return out
    # 2) большая свеча (перцентильный фильтр размера)
    if csize[i] / C[i] * 100 > size_thr:
        out['decision'] = 'бол.свеча'; return out
    # 3) согласованность против тренда (v4: 3/5 только для HH-шортов)
    up = sum(1 for x in labels if x in ('HH', 'HL'))
    dn = sum(1 for x in labels if x in ('LH', 'LL'))
    had_slom = any((labels[j-1] == 'HH' and labels[j] == 'LL') or
                   (labels[j-1] == 'LL' and labels[j] == 'HL')
                   for j in range(1, len(labels)))
    if not had_slom:
        tmin = 3 if last == 'HH' else 4       # <<< правка v4
        if up >= tmin and not is_long:
            out['decision'] = 'против тренда'; return out
        if dn >= 4 and is_long:
            out['decision'] = 'против тренда'; return out
    # 4) свежий слом (HH->LL / LL->HL на последних двух метках)
    if len(labels) >= 2 and ((labels[-2] == 'HH' and labels[-1] == 'LL') or
                             (labels[-2] == 'LL' and labels[-1] == 'HL')):
        out['decision'] = 'слом'; return out
    # 5) стоп за триггерной свечой +-0.1%
    sl = L[i] * 0.999 if is_long else H[i] * 1.001
    if (is_long and sl >= entry) or ((not is_long) and sl <= entry):
        out['decision'] = 'стоп?'; return out
    risk = abs(entry - sl)
    # 6) магнит: ПЕРВАЯ метка нужной стороны с RR>=3 (перепрыгивание ближних с RR<3 разрешено)
    side = 'high' if is_long else 'low'
    cand = [r for r in seq if r['side'] == side]
    tp = np.nan
    if is_long:
        for r in sorted([r for r in cand if r['price'] > entry], key=lambda r: r['price']):
            if (r['price'] - entry) / risk >= 3:
                tp = r['price']; break
    else:
        for r in sorted([r for r in cand if r['price'] < entry], key=lambda r: -r['price']):
            if (entry - r['price']) / risk >= 3:
                tp = r['price']; break
    if np.isnan(tp):
        out['decision'] = 'нет RR'; return out
    out.update(decision='ВХОД', entry=entry, stop=sl, tp=tp, RR=abs(tp - entry) / risk)
    return out


def resolve(E, i, is_long, sl, tp):
    """Исход сделки: TP / SL / NONE. Одновременное касание SL и TP в баре = SL (консервативно)."""
    H = E['H']; L = E['L']; n = E['N']
    for j in range(i + 1, min(i + HORIZON, n)):
        hit_sl = (L[j] <= sl) if is_long else (H[j] >= sl)
        hit_tp = (H[j] >= tp) if is_long else (L[j] <= tp)
        if hit_sl and hit_tp:
            return 'SL'
        if hit_tp:
            return 'TP'
        if hit_sl:
            return 'SL'
    return 'NONE'


def run_v4(E):
    """Полный прогон v4: DataFrame сделок (dt, dir, last, entry, stop, tp, RR, out, R)."""
    df = E['df']; TRIG = E['TRIG']
    rows = []
    for i in np.where(TRIG)[0]:
        d = decide_v4(E, i)
        if d.get('decision') != 'ВХОД':
            continue
        is_long = d['dir'] == 'LONG'
        o = resolve(E, i, is_long, d['stop'], d['tp'])
        if o not in ('TP', 'SL'):
            continue
        R = d['RR'] if o == 'TP' else -1.0
        rows.append(dict(dt=df['time'].iloc[i], i=i, dir=d['dir'], last=d['labels'][-1],
                         entry=d['entry'], stop=d['stop'], tp=d['tp'],
                         RR=round(d['RR'], 2), out=o, R=R))
    return pd.DataFrame(rows)
