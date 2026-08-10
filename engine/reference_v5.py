# -*- coding: utf-8 -*-
"""
Эталонная реализация правил v5 по документации (docs/V5_инструкция.md)
и по логике Pine-индикатора v7 — для сверки с движком engine_v5.py.
Эталон (золото 9 лет): 939 сделок, +53.2R, WR 20%.
v4 (без запретов): 1246 сделок, -38.5R.

Триггеры берутся из колонки CSV (уже размечены), пивоты/метки/цепочка
решения считаются здесь. Запуск: python3 engine/reference_v5.py
"""
import numpy as np
import pandas as pd

BANNED_TAILS = ('LL>HH', 'LL>LH>HL', 'HH>LL>LL')  # + 'HH>LH' — все 4 ниже
BANNED_TAILS = ('LL>HH', 'LL>LH>HL', 'HH>LH', 'HH>LL>LL')


def load(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'trigger']
    return df


def pivots(df, left=2, right=2):
    """Пивоты 2/2 (строго выше/ниже соседей), подтверждение через `right` баров.
    Возвращает список (conf_bar, pivot_bar, side, price) в порядке подтверждения."""
    h = df['high'].values
    l = df['low'].values
    n = len(df)
    out = []
    for i in range(left, n - right):
        win_h = h[i - left:i + right + 1]
        win_l = l[i - left:i + right + 1]
        if h[i] == win_h.max() and (win_h == h[i]).sum() == 1:
            out.append((i + right, i, 'high', h[i]))
        if l[i] == win_l.min() and (win_l == l[i]).sum() == 1:
            out.append((i + right, i, 'low', l[i]))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def labels_stream(piv):
    """Метки: новый пивот против предыдущего того же типа. Равные цены — пропуск.
    Возвращает список (conf_bar, label, side, price)."""
    last = {'high': None, 'low': None}
    out = []
    for conf, bar, side, price in piv:
        prev = last[side]
        last[side] = price
        if prev is None or price == prev:
            continue
        if side == 'high':
            lab = 'HH' if price > prev else 'LH'
        else:
            lab = 'HL' if price > prev else 'LL'
        out.append((conf, lab, side, price))
    return out


def run(df, use_bans=True, rr_min=3.0, buf_pct=0.1, pctl=80.0, sl_first=True):
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    trig_idx = np.where(df['trigger'].values == 1)[0]

    # порог размера: перцентиль по ВСЕМ триггерным свечам (как движок, lookahead)
    sz_all = (h[trig_idx] - l[trig_idx]) / c[trig_idx] * 100
    thr = np.percentile(sz_all, pctl)

    labs = labels_stream(pivots(df))
    lab_confs = np.array([x[0] for x in labs])

    rows = []
    for i in trig_idx:
        # рабочее окно: 5 последних меток, подтверждённых к бару i включительно
        k = np.searchsorted(lab_confs, i, side='right')
        if k == 0:
            continue
        w = labs[max(0, k - 5):k]
        seq = [x[1] for x in w]
        lastL = seq[-1]
        last_conf = w[-1][0]
        is_long = lastL in ('HL', 'LL')

        # ШАГ 2: заморозка свежего экстремума
        if lastL in ('LL', 'HH') and (i - last_conf) <= 1:
            continue
        # ШАГ 3: размер триггерной свечи
        sz = (h[i] - l[i]) / c[i] * 100
        if sz > thr:
            continue
        # ШАГ 4: согласованность (только без слома внутри пятёрки)
        up = sum(1 for s in seq if s in ('HH', 'HL'))
        dn = sum(1 for s in seq if s in ('LH', 'LL'))
        had_slom = any((a == 'HH' and b == 'LL') or (a == 'LL' and b == 'HL')
                       for a, b in zip(seq, seq[1:]))
        if not had_slom:
            if not is_long and up >= (3 if lastL == 'HH' else 4):
                continue
            if is_long and dn >= 4:
                continue
        # ШАГ 5: свежий слом
        if len(seq) >= 2:
            a, b = seq[-2], seq[-1]
            if (a == 'HH' and b == 'LL') or (a == 'LL' and b == 'HL'):
                continue
        # ШАГ 7: стоп
        en = c[i]
        sl = l[i] * (1 - buf_pct / 100) if is_long else h[i] * (1 + buf_pct / 100)
        if (is_long and sl >= en) or (not is_long and sl <= en):
            continue
        risk = abs(en - sl)
        # ШАГ 8: магнит — метки стороны тейка дальше входа, от ближней, первая RR>=3
        side_need = 'high' if is_long else 'low'
        cand = [x[3] for x in w if x[2] == side_need and
                ((is_long and x[3] > en) or (not is_long and x[3] < en))]
        cand.sort(reverse=not is_long)
        tp = None
        for p in cand:
            if abs(p - en) / risk >= rr_min:
                tp = p
                break
        if tp is None:
            continue
        rr = abs(tp - en) / risk
        # ШАГ 6 (в движке — последним, поверх v4): запреты структур
        if use_bans:
            s5 = '>'.join(seq)
            if any(s5.endswith(b) for b in BANNED_TAILS):
                continue
        # резолв: с бара i+1; SL и TP в одном баре = SL (консервативно)
        out = None
        for j in range(i + 1, len(df)):
            hit_sl = l[j] <= sl if is_long else h[j] >= sl
            hit_tp = h[j] >= tp if is_long else l[j] <= tp
            if hit_sl and hit_tp:
                out = 'SL' if sl_first else 'TP'
                break
            if hit_sl:
                out = 'SL'
                break
            if hit_tp:
                out = 'TP'
                break
        if out is None:
            continue
        R = round(rr, 2) if out == 'TP' else -1.0
        rows.append(dict(dt=df['time'].iloc[i], i=i, dir='LONG' if is_long else 'SHORT',
                         last=lastL, entry=en, stop=sl, tp=tp, RR=round(rr, 2),
                         out=out, R=R))
    return pd.DataFrame(rows), thr


def report(tr, title):
    total = tr['R'].sum()
    wr = (tr['out'] == 'TP').mean() * 100
    print(f"\n=== {title} ===")
    print(f"Сделок: {len(tr)} | Итог: {total:+.1f}R | WR: {wr:.0f}%")
    print("Бакеты по последней метке:")
    for lab in ('LL', 'LH', 'HL', 'HH'):
        sub = tr[tr['last'] == lab]
        print(f"  {lab}: {len(sub):4d} сделок, {sub['R'].sum():+8.1f}R")
    half = tr['dt'] < '2022'
    print(f"Train (<2022): {tr[half]['R'].sum():+.1f}R ({half.sum()} сделок) | "
          f"Test (>=2022): {tr[~half]['R'].sum():+.1f}R ({(~half).sum()} сделок)")


if __name__ == '__main__':
    df = load('data/gold/xau_1h_master.csv')
    print(f"Баров: {len(df)}, триггеров: {int(df['trigger'].sum())}")
    tr5, thr = run(df, use_bans=True)
    print(f"Порог P80 размера триггерных: {thr:.4f}%")
    report(tr5, "v5 (с запретами) — эталон движка: 939 сделок, +53.2R, WR 20%")
    tr4, _ = run(df, use_bans=False)
    report(tr4, "v4 (без запретов) — эталон движка: 1246 сделок, -38.5R")
