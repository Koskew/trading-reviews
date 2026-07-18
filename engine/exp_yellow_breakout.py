# -*- coding: utf-8 -*-
"""
ИГРА: новая система от жёлтой точки — ПРОБОЙ триггерной свечи.
Идея: жёлтая точка (нерешительность) = приготовиться; пробой её high/low
в течение W баров = импульс, вход по направлению пробоя стоп-ордером.
Стоп за противоположной стороной свечи, тейк = k*R фиксированно.
Одна позиция за раз; новая жёлтая точка обновляет ожидание.
Дисциплина: подбор параметров ТОЛЬКО на train (2017-21), test — один раз.
Запуск: python3 engine/exp_yellow_breakout.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'engine')
from reference_v5 import load


def backtest(df, W=4, k=2.0, buf=0.001, size_thr=None, trend_sma=None,
             hours=None, both_cancel=True):
    """Стейт-машина: флэт → ожидание пробоя → позиция. Одна позиция."""
    H = df['high'].values
    L = df['low'].values
    C = df['close'].values
    O = df['open'].values
    T = df['trigger'].values
    n = len(df)
    hrs = pd.to_datetime(df['time'], utc=True).dt.hour.values
    sma = pd.Series(C).rolling(trend_sma).mean().values if trend_sma else None

    trades = []
    # состояние
    pend_until = -1
    pend_hi = pend_lo = np.nan
    pend_long_ok = pend_short_ok = True
    pos = 0  # 0 флэт, +1 лонг, -1 шорт
    en = sl = tp = np.nan

    for j in range(1, n):
        # 1) сопровождение позиции (SL первым — консервативно)
        if pos != 0:
            hit_sl = L[j] <= sl if pos > 0 else H[j] >= sl
            hit_tp = H[j] >= tp if pos > 0 else L[j] <= tp
            if hit_sl:
                trades.append((df['time'].iloc[j], pos, -1.0))
                pos = 0
            elif hit_tp:
                trades.append((df['time'].iloc[j], pos, k))
                pos = 0
        # 2) ожидание пробоя
        if pos == 0 and j <= pend_until:
            up = H[j] >= pend_hi and pend_long_ok
            dn = L[j] <= pend_lo and pend_short_ok
            if up and dn and both_cancel:
                pend_until = -1  # обе стороны в одном баре — отмена
            elif up or dn:
                if up:
                    pos = 1
                    en = pend_hi
                    sl = pend_lo * (1 - buf)
                else:
                    pos = -1
                    en = pend_lo
                    sl = pend_hi * (1 + buf)
                risk = abs(en - sl)
                tp = en + k * risk if pos > 0 else en - k * risk
                pend_until = -1
                # вход и стоп в одном баре (консервативно: SL)
                hit_sl = L[j] <= sl if pos > 0 else H[j] >= sl
                if hit_sl:
                    trades.append((df['time'].iloc[j], pos, -1.0))
                    pos = 0
        # 3) новая жёлтая точка (закрытие бара j) — обновляет ожидание
        if T[j] == 1 and pos == 0:
            if size_thr is not None and (H[j] - L[j]) / C[j] * 100 > size_thr:
                continue
            if hours is not None and hrs[j] not in hours:
                continue
            pend_hi = H[j]
            pend_lo = L[j]
            pend_until = j + W
            if sma is not None and not np.isnan(sma[j]):
                pend_long_ok = C[j] > sma[j]
                pend_short_ok = C[j] < sma[j]
            else:
                pend_long_ok = pend_short_ok = True
    return pd.DataFrame(trades, columns=['dt', 'dir', 'R'])


def weekly(tr):
    if len(tr) == 0:
        return None
    t = tr.copy()
    t['w'] = pd.to_datetime(t['dt'], utc=True).dt.strftime('%G-W%V')
    wk = t.groupby('w')['R'].sum()
    return wk


def report(tr, name, brief=False):
    if len(tr) == 0:
        print(f"{name}: нет сделок")
        return
    wk = weekly(tr)
    tot = tr['R'].sum()
    wr = (tr['R'] > 0).mean() * 100
    pos_w = (wk > 0).mean() * 100
    line = (f"{name}: {len(tr):4d} сделок | {tot:+7.1f}R | WR {wr:4.1f}% | "
            f"недель {len(wk)}, плюсовых {pos_w:4.1f}% | "
            f"ср.нед {wk.mean():+5.2f}R | худшая {wk.min():+5.1f}R")
    print(line)


if __name__ == '__main__':
    df = load('data/gold/xau_1h_master.csv')
    years = pd.to_datetime(df['time'], utc=True).dt.year
    train = df[years < 2022].reset_index(drop=True)

    print("═══ ЭТАП 1: базовая механика, сетка W × k (ТОЛЬКО TRAIN 2017-21) ═══")
    for W in (2, 3, 4, 6):
        for k in (1.5, 2.0, 2.5, 3.0):
            tr = backtest(train, W=W, k=k)
            report(tr, f"W={W} k={k}")


def backtest2(df, W=3, k=2.5, buf=0.001, size_thr=None, trend_sma=None,
              sessions=None, gap_honest=True):
    """Этап 2: + гэп-честность (вход по open при гэпе), фильтры."""
    H = df['high'].values
    L = df['low'].values
    C = df['close'].values
    O = df['open'].values
    T = df['trigger'].values
    n = len(df)
    hrs = pd.to_datetime(df['time'], utc=True).dt.hour.values
    sma = pd.Series(C).rolling(trend_sma).mean().values if trend_sma else None

    trades = []
    pend_until = -1
    pend_hi = pend_lo = np.nan
    ok_l = ok_s = True
    pos = 0
    en = sl = tp = np.nan

    for j in range(1, n):
        if pos != 0:
            hit_sl = L[j] <= sl if pos > 0 else H[j] >= sl
            hit_tp = H[j] >= tp if pos > 0 else L[j] <= tp
            if hit_sl:
                trades.append((df['time'].iloc[j], pos, -1.0))
                pos = 0
            elif hit_tp:
                trades.append((df['time'].iloc[j], pos, k))
                pos = 0
        if pos == 0 and j <= pend_until:
            up = H[j] >= pend_hi and ok_l
            dn = L[j] <= pend_lo and ok_s
            if up and dn:
                pend_until = -1
            elif up or dn:
                if up:
                    pos = 1
                    en = max(O[j], pend_hi) if gap_honest else pend_hi
                    sl = pend_lo * (1 - buf)
                else:
                    pos = -1
                    en = min(O[j], pend_lo) if gap_honest else pend_lo
                    sl = pend_hi * (1 + buf)
                risk = abs(en - sl)
                tp = en + k * risk if pos > 0 else en - k * risk
                pend_until = -1
                hit_sl = L[j] <= sl if pos > 0 else H[j] >= sl
                if hit_sl:
                    trades.append((df['time'].iloc[j], pos, -1.0))
                    pos = 0
        if T[j] == 1 and pos == 0:
            if size_thr is not None and (H[j] - L[j]) / C[j] * 100 > size_thr:
                continue
            if sessions is not None and hrs[j] not in sessions:
                continue
            pend_hi = H[j]
            pend_lo = L[j]
            pend_until = j + W
            if sma is not None and not np.isnan(sma[j]):
                ok_l = C[j] > sma[j]
                ok_s = C[j] < sma[j]
            else:
                ok_l = ok_s = True
    return pd.DataFrame(trades, columns=['dt', 'dir', 'R'])


def stage2():
    df = load('data/gold/xau_1h_master.csv')
    years = pd.to_datetime(df['time'], utc=True).dt.year
    train = df[years < 2022].reset_index(drop=True)
    tt = df[years < 2022]
    sz = ((tt['high'] - tt['low']) / tt['close'] * 100)[tt['trigger'] == 1]
    p80 = np.percentile(sz, 80)
    print(f"P80 размера триггеров на train: {p80:.4f}%")

    print("\n═══ ЭТАП 2а: гэп-честность и фильтры (TRAIN, W=3) ═══")
    for k in (1.5, 2.5):
        report(backtest2(train, k=k, gap_honest=False), f"k={k} БЕЗ гэп-честности")
        report(backtest2(train, k=k), f"k={k} база (гэп-честно)   ")
        report(backtest2(train, k=k, size_thr=p80), f"k={k} + размер<P80        ")
        report(backtest2(train, k=k, trend_sma=200), f"k={k} + тренд SMA200      ")
        report(backtest2(train, k=k, size_thr=p80, trend_sma=200), f"k={k} + размер + тренд   ")

    print("\n═══ ЭТАП 2б: почасовая развёртка базы k=2.5 (TRAIN) — для сессий ═══")
    tr = backtest2(train, k=2.5)
    tr['h'] = pd.to_datetime(tr['dt'], utc=True).dt.hour
    hr = tr.groupby('h')['R'].agg(['sum', 'count'])
    for h, r in hr.iterrows():
        print(f"  {h:02d} UTC: {r['sum']:+7.1f}R ({int(r['count'])})")


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'stage2':
    stage2()


def stage3():
    df = load('data/gold/xau_1h_master.csv')
    years = pd.to_datetime(df['time'], utc=True).dt.year
    train = df[years < 2022].reset_index(drop=True)

    print("═══ ЭТАП 3а: лонг vs шорт (база k=2.5, TRAIN) ═══")
    tr = backtest2(train, k=2.5)
    for d, nm in ((1, 'LONG '), (-1, 'SHORT')):
        sub = tr[tr['dir'] == d]
        report(sub, nm)

    print("\n═══ ЭТАП 3б: сессии крупными блоками (TRAIN) ═══")
    sess = {
        'Европа+США 08-20': set(range(8, 21)),
        'США 12-20       ': set(range(12, 21)),
        'ядро США 13-16  ': set(range(13, 17)),
    }
    for k in (1.5, 2.0, 2.5):
        for nm, hs in sess.items():
            report(backtest2(train, k=k, sessions=hs), f"k={k} {nm}")

    print("\n═══ ЭТАП 3в: размер наоборот (только крупные жёлтые, TRAIN, США 12-20) ═══")
    tt = train[train['trigger'] == 1]
    szs = (tt['high'] - tt['low']) / tt['close'] * 100
    for pmin, nm in ((None, 'без фильтра   '), (np.percentile(szs, 20), '>P20'), (np.percentile(szs, 50), '>P50')):
        if pmin is None:
            report(backtest2(train, k=2.5, sessions=set(range(12, 21))), f"размер {nm}")
        else:
            # min-size: инвертированный фильтр
            H = train['high'].values; L = train['low'].values; C = train['close'].values
            df2 = train.copy()
            small = (H - L) / C * 100 < pmin
            df2.loc[small & (df2['trigger'] == 1), 'trigger'] = 0
            report(backtest2(df2, k=2.5, sessions=set(range(12, 21))), f"размер {nm:14s}")


if len(sys.argv) > 1 and sys.argv[1] == 'stage3':
    stage3()


def backtest3(df, W=3, k=2.5, buf=0.001, sessions=None, week_stop=None):
    """Этап 4: + недельный предохранитель (после -X R за неделю — стоп до пн)."""
    H = df['high'].values
    L = df['low'].values
    C = df['close'].values
    O = df['open'].values
    T = df['trigger'].values
    n = len(df)
    tt = pd.to_datetime(df['time'], utc=True)
    hrs = tt.dt.hour.values
    wk_id = tt.dt.strftime('%G-W%V').values

    trades = []
    pend_until = -1
    pend_hi = pend_lo = np.nan
    pos = 0
    en = sl = tp = np.nan
    cur_wk = ''
    wk_r = 0.0

    for j in range(1, n):
        if wk_id[j] != cur_wk:
            cur_wk = wk_id[j]
            wk_r = 0.0
        if pos != 0:
            hit_sl = L[j] <= sl if pos > 0 else H[j] >= sl
            hit_tp = H[j] >= tp if pos > 0 else L[j] <= tp
            if hit_sl:
                trades.append((df['time'].iloc[j], pos, -1.0))
                wk_r -= 1.0
                pos = 0
            elif hit_tp:
                trades.append((df['time'].iloc[j], pos, k))
                wk_r += k
                pos = 0
        blocked = week_stop is not None and wk_r <= -week_stop
        if pos == 0 and j <= pend_until and not blocked:
            up = H[j] >= pend_hi
            dn = L[j] <= pend_lo
            if up and dn:
                pend_until = -1
            elif up or dn:
                if up:
                    pos = 1
                    en = max(O[j], pend_hi)
                    sl = pend_lo * (1 - buf)
                else:
                    pos = -1
                    en = min(O[j], pend_lo)
                    sl = pend_hi * (1 + buf)
                risk = abs(en - sl)
                tp = en + k * risk if pos > 0 else en - k * risk
                pend_until = -1
                hit_sl = L[j] <= sl if pos > 0 else H[j] >= sl
                if hit_sl:
                    trades.append((df['time'].iloc[j], pos, -1.0))
                    wk_r -= 1.0
                    pos = 0
        if T[j] == 1 and pos == 0 and not blocked:
            if sessions is not None and hrs[j] not in sessions:
                continue
            pend_hi = H[j]
            pend_lo = L[j]
            pend_until = j + W
    return pd.DataFrame(trades, columns=['dt', 'dir', 'R'])


def stage4():
    df = load('data/gold/xau_1h_master.csv')
    years = pd.to_datetime(df['time'], utc=True).dt.year
    train = df[years < 2022].reset_index(drop=True)

    print("═══ ЭТАП 4: недельный предохранитель (TRAIN, W=3, k=2.5) ═══")
    for ws in (None, 3, 4, 5, 6):
        report(backtest3(train, week_stop=ws), f"стоп-неделя {str(ws):4s}, без сессии")
    for ws in (None, 3, 4, 5):
        report(backtest3(train, week_stop=ws, sessions=set(range(8, 21))), f"стоп-неделя {str(ws):4s}, 08-20 UTC ")


if len(sys.argv) > 1 and sys.argv[1] == 'stage4':
    stage4()


def stage5():
    df = load('data/gold/xau_1h_master.csv')
    years = pd.to_datetime(df['time'], utc=True).dt.year
    train = df[years < 2022].reset_index(drop=True)
    test = df[years >= 2022].reset_index(drop=True)

    print("═══ ЭТАП 5а: выбор k при сессии 08-20 + стоп-неделя 3 (TRAIN) ═══")
    for k in (1.5, 2.0, 2.5):
        report(backtest3(train, k=k, week_stop=3, sessions=set(range(8, 21))), f"k={k}")

    print("\n═══ ЭТАП 5б: ФИНАЛ — ЕДИНСТВЕННЫЙ прогон TEST (2022-26) ═══")
    cfg = dict(W=3, k=2.5, week_stop=3, sessions=set(range(8, 21)))
    tr_train = backtest3(train, **cfg)
    tr_test = backtest3(test, **cfg)
    report(tr_train, "TRAIN 2017-21")
    report(tr_test,  "TEST  2022-26")

    wk = weekly(tr_test)
    print(f"\nНедельное распределение TEST: медиана {wk.median():+.2f}R | "
          f"квартели [{wk.quantile(.25):+.1f}, {wk.quantile(.75):+.1f}] | "
          f"мин {wk.min():+.1f} | макс {wk.max():+.1f}")
    ym = pd.to_datetime(tr_test['dt'], utc=True).dt.strftime('%Y')
    print("Погодовая развёртка TEST:")
    for y, s in tr_test.groupby(ym)['R'].agg(['sum', 'count']).iterrows():
        print(f"  {y}: {s['sum']:+7.1f}R ({int(s['count'])})")
    # серия недель: макс подряд минусовых
    signs = (wk > 0).values
    cur = mx = 0
    for s_ in signs:
        cur = cur + 1 if not s_ else 0
        mx = max(mx, cur)
    print(f"Макс подряд неплюсовых недель (TEST): {mx}")


if len(sys.argv) > 1 and sys.argv[1] == 'stage5':
    stage5()


def stage6():
    df = load('data/gold/xau_1h_master.csv')
    cfg = dict(W=3, k=2.5, week_stop=3, sessions=set(range(8, 21)))
    tr = backtest3(df, **cfg)
    report(tr, "ПОЛНАЯ ИСТОРИЯ 2017-26")
    # чувствительность к издержкам (спред ~7% риска)
    for c in (0.05, 0.10):
        print(f"  с издержками {c}R/сделку: {tr['R'].sum() - len(tr) * c:+.1f}R")
    kyiv = pd.to_datetime(tr['dt'], utc=True).dt.tz_convert('Europe/Kyiv')
    print("\nПоследние 10 сделок (Киев, бар ВЫХОДА | направление | R):")
    for i in range(len(tr) - 10, len(tr)):
        print(f"  {kyiv.iloc[i].strftime('%d.%m %H:%M')} | {'LONG ' if tr['dir'].iloc[i] > 0 else 'SHORT'} | {tr['R'].iloc[i]:+.1f}")


if len(sys.argv) > 1 and sys.argv[1] == 'stage6':
    stage6()
