# -*- coding: utf-8 -*-
"""
Проверка кандидата: блокировка «одна позиция» поверх v5.
Наблюдение: 723 сделки, +64.1R против эталона 939 / +53.1R.
Проверки по протоколу: train/test + контроль случайностью.
Запуск: python3 engine/exp_one_pos.py
"""
import importlib.util
import numpy as np
import pandas as pd


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def trades_with_exits(E, e4, e5):
    """Все сделки v5 + бар выхода (для расчёта блокировки)."""
    df = E['df']; TRIG = E['TRIG']
    H = E['H']; L = E['L']; n = E['N']
    rows = []
    for i in np.where(TRIG)[0]:
        d = e5.decide_v5(E, i, e4)
        if d.get('decision') != 'ВХОД':
            continue
        is_long = d['dir'] == 'LONG'
        sl, tp = d['stop'], d['tp']
        out, exit_bar = None, None
        for j in range(i + 1, min(i + 1500, n)):
            hit_sl = (L[j] <= sl) if is_long else (H[j] >= sl)
            hit_tp = (H[j] >= tp) if is_long else (L[j] <= tp)
            if hit_sl:
                out, exit_bar = 'SL', j
                break
            if hit_tp:
                out, exit_bar = 'TP', j
                break
        if out is None:
            continue
        R = d['RR'] if out == 'TP' else -1.0
        rows.append(dict(dt=df['time'].iloc[i], i=i, exit=exit_bar,
                         dir=d['dir'], last=d['labels'][-1], out=out, R=R))
    return pd.DataFrame(rows)


def one_pos_filter(tr, allow_reentry_on_exit_bar=False):
    """Оставляет сделки, не перекрывающиеся с предыдущей принятой."""
    keep = []
    pos_exit = -1
    for idx, row in tr.iterrows():
        blocked = row['i'] < pos_exit or (row['i'] == pos_exit and not allow_reentry_on_exit_bar)
        if blocked:
            continue
        keep.append(idx)
        pos_exit = row['exit']
    return tr.loc[keep]


def halves(tr):
    m = tr['dt'] < '2022'
    return tr[m], tr[~m]


if __name__ == '__main__':
    base = 'engine/'
    e3 = load_mod('e3', base + 'engine_v3.py')
    e4 = load_mod('e4', base + 'engine_v4.py')
    e5 = load_mod('e5', base + 'engine_v5.py')
    E = e3.build_engine('data/gold/xau_1h_master.csv')

    tr = trades_with_exits(E, e4, e5)
    print(f"Эталон v5: {len(tr)} сделок, {tr['R'].sum():+.1f}R")

    op = one_pos_filter(tr)
    tra, tst = halves(op)
    print(f"\nОдна позиция: {len(op)} сделок, {op['R'].sum():+.1f}R, "
          f"WR {(op['out'] == 'TP').mean() * 100:.0f}%")
    print(f"  Train (<2022): {tra['R'].sum():+.1f}R ({len(tra)} сделок)")
    print(f"  Test (>=2022): {tst['R'].sum():+.1f}R ({len(tst)} сделок)")
    op2 = one_pos_filter(tr, allow_reentry_on_exit_bar=True)
    print(f"  Чувствительность (вход в бар выхода разрешён): "
          f"{len(op2)} сделок, {op2['R'].sum():+.1f}R")

    dropped = tr[~tr['i'].isin(op['i'])]
    print(f"\nВыброшено блокировкой: {len(dropped)} сделок, "
          f"{dropped['R'].sum():+.1f}R (в среднем {dropped['R'].mean():+.3f}R/сделку; "
          f"у оставшихся {op['R'].mean():+.3f}R/сделку)")
    print("Выброшенные по баке��ам:")
    for lab in ('LL', 'LH', 'HL', 'HH'):
        sub = dropped[dropped['last'] == lab]
        print(f"  {lab}: {len(sub):3d} сделок, {sub['R'].sum():+7.1f}R")

    # Контроль случайностью: 10000 раз выбрасываем столько же СЛУЧАЙНЫХ сделок
    rng = np.random.default_rng(42)
    R = tr['R'].values
    n_drop = len(dropped)
    sims = np.empty(10000)
    for s in range(10000):
        drop_idx = rng.choice(len(R), size=n_drop, replace=False)
        mask = np.ones(len(R), bool)
        mask[drop_idx] = False
        sims[s] = R[mask].sum()
    real = op['R'].sum()
    p = (sims >= real).mean()
    print(f"\nКонтроль случайностью (10000 симуляций, выброс {n_drop} случайных):")
    print(f"  Медиана случайного выброса: {np.median(sims):+.1f}R | "
          f"5-95%: [{np.percentile(sims, 5):+.1f}, {np.percentile(sims, 95):+.1f}]")
    print(f"  Блокировка: {real:+.1f}R | p-value (случайность >= блокировки): {p:.3f}")

    # Помесячная развёртка прироста: не тянут ли весь эффект 2-3 месяца
    both = tr.copy()
    both['kept'] = both['i'].isin(op['i'])
    both['ym'] = pd.to_datetime(both['dt'], utc=True).dt.strftime('%Y-%m')
    monthly_gain = both[~both['kept']].groupby('ym')['R'].sum() * -1  # выброс минусовых = прирост
    top = monthly_gain.sort_values(ascending=False)
    print(f"\nПрирост от блокировки по месяцам (топ-5 из {len(top)}):")
    for ym, g in top.head(5).items():
        print(f"  {ym}: {g:+.1f}R")
    print(f"  Суммарный прирост: {monthly_gain.sum():+.1f}R | "
          f"доля топ-3 месяцев: {top.head(3).sum() / monthly_gain.sum() * 100 if monthly_gain.sum() != 0 else float('nan'):.0f}%")
