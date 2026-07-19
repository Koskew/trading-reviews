# -*- coding: utf-8 -*-
"""
Проверка теории тренда (проверка №1 из доков + формулировка трейдера):
«живой нисходящий тренд не пробивает последний LH; восходящий не перебивает HL».

Фальсифицируемая версия: в нисходящей структуре (последний хай = LH,
последний лоу = LL) — что наступает раньше: новый LL (тренд продолжился)
или пробой уровня LH (фитилём / телом)? И что происходит ПОСЛЕ пробоя
фитилём без тела — тренд умирает или живёт?

Плюс: R движка v5 по режимам (ВОСХ/НИСХ/БОКОВИК на баре входа).
Запуск: python3 engine/exp_trend_state.py
"""
import importlib.util
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'engine')
from reference_v5 import load, pivots, labels_stream


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


df = load('data/gold/xau_1h_master.csv')
H = df['high'].values
L = df['low'].values
O = df['open'].values
C = df['close'].values
n = len(df)
labs = labels_stream(pivots(df))  # (conf, label, side, price)

# ── режим на каждом баре: по последним меткам каждой стороны ──
# ВОСХ = последний хай HH и последний лоу HL; НИСХ = LH и LL; иначе БОКОВИК/переход
regime = np.zeros(n, dtype=int)  # +1 ВОСХ, -1 НИСХ, 0 БОКОВИК
lastH = ''
lastL_ = ''
li = 0
for j in range(n):
    while li < len(labs) and labs[li][0] <= j:
        _, lab, side, price = labs[li]
        if side == 'high':
            lastH = lab
        else:
            lastL_ = lab
        li += 1
    regime[j] = 1 if (lastH == 'HH' and lastL_ == 'HL') else (-1 if (lastH == 'LH' and lastL_ == 'LL') else 0)

share = [(regime == v).mean() * 100 for v in (1, -1, 0)]
print(f"Время в режимах: ВОСХ {share[0]:.0f}% | НИСХ {share[1]:.0f}% | БОКОВИК/переход {share[2]:.0f}%")

# ── событийная проверка: каждый подтверждённый LH при последнем лоу LL ──
def run_side(down=True):
    """down=True: НИСХ (уровень = LH, продолжение = новый LL).
    Возвращает исходы: cont (новый экстремум раньше пробоя), wick_first, body_first,
    и судьбу тренда после чисто фитильного пробоя."""
    cont = wick_first = body_first = 0
    after_wick_resume = after_wick_die = 0
    lastSide = {'high': ('', np.nan), 'low': ('', np.nan)}
    events = []
    li2 = 0
    for idx, (conf, lab, side, price) in enumerate(labs):
        prevH, prevHp = lastSide['high']
        prevL, prevLp = lastSide['low']
        lastSide[side] = (lab, price)
        if down and lab == 'LH' and prevL == 'LL':
            events.append((conf, price, idx))
        if not down and lab == 'HL' and prevH == 'HH':
            events.append((conf, price, idx))
    for conf, lvl, idx in events:
        # вперёд по барам: что раньше — подтверждение нового экстремума или пробой уровня
        nxt_ext = None  # бар подтверждения нового LL (или HH для UP-версии... см. ниже)
        for conf2, lab2, side2, price2 in labs[idx + 1:]:
            if down and lab2 == 'LL':
                nxt_ext = conf2
                break
            if not down and lab2 == 'HH':
                nxt_ext = conf2
                break
        wick_bar = body_bar = None
        for j in range(conf + 1, n):
            if down:
                if wick_bar is None and H[j] > lvl:
                    wick_bar = j
                if body_bar is None and max(O[j], C[j]) > lvl:
                    body_bar = j
            else:
                if wick_bar is None and L[j] < lvl:
                    wick_bar = j
                if body_bar is None and min(O[j], C[j]) < lvl:
                    body_bar = j
            if wick_bar is not None and body_bar is not None:
                break
            if nxt_ext is not None and j > nxt_ext and wick_bar is not None:
                break
        e = nxt_ext if nxt_ext is not None else 10 ** 9
        w = wick_bar if wick_bar is not None else 10 ** 9
        b = body_bar if body_bar is not None else 10 ** 9
        if e < w and e < b:
            cont += 1
        elif w < b:
            wick_first += 1
            # судьба после фитильного пробоя (тело ещё не пробило): что раньше —
            # новый экстремум тренда (тренд жив) или пробой телом (тренд умер)
            if e < b:
                after_wick_resume += 1
            else:
                after_wick_die += 1
        else:
            body_first += 1
    return cont, wick_first, body_first, after_wick_resume, after_wick_die


for down, nm, ext, lvlname in ((True, 'НИСХОДЯЩИЙ', 'LL', 'LH'), (False, 'ВОСХОДЯЩИЙ', 'HH', 'HL')):
    cont, wf, bf, awr, awd = run_side(down)
    tot = cont + wf + bf
    print(f"\n{nm} тренд ({tot} эпизодов «{lvlname} при живом тренде»):")
    print(f"  Новый {ext} раньше любого пробоя {lvlname}: {cont} ({cont / tot * 100:.0f}%) — тренд продолжился")
    print(f"  Пробой {lvlname} ФИТИЛЁМ раньше: {wf} ({wf / tot * 100:.0f}%)")
    print(f"    из них после фитиля тренд ВЫЖИЛ (новый {ext} раньше тела): {awr} ({awr / max(wf, 1) * 100:.0f}%)")
    print(f"    из них тренд умер (тело добило): {awd} ({awd / max(wf, 1) * 100:.0f}%)")
    print(f"  Пробой {lvlname} сразу ТЕЛОМ: {bf} ({bf / tot * 100:.0f}%)")

# ── R движка v5 по режимам ──
e3 = load_mod('e3', 'engine/engine_v3.py')
e4 = load_mod('e4', 'engine/engine_v4.py')
e5 = load_mod('e5', 'engine/engine_v5.py')
from exp_one_pos import trades_with_exits
E = e3.build_engine('data/gold/xau_1h_master.csv')
tr5 = trades_with_exits(E, e4, e5)
tr5['reg'] = regime[tr5['i'].values]
print(f"\nR ДВИЖКА v5 ПО РЕЖИМАМ (на баре входа):")
half = pd.to_datetime(tr5['dt'], utc=True) < pd.Timestamp('2022', tz='UTC')
for v, nm in ((1, 'ВОСХ'), (-1, 'НИСХ'), (0, 'БОКОВИК')):
    s = tr5[tr5['reg'] == v]
    st = s[half.reindex(s.index, fill_value=False)]
    se = s[~half.reindex(s.index, fill_value=True)]
    print(f"  {nm:8s}: {len(s):3d} сделок {s['R'].sum():+7.1f}R | train {st['R'].sum():+6.1f} | test {se['R'].sum():+6.1f}")
