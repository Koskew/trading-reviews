# -*- coding: utf-8 -*-
"""
ДВИЖОК v5 = v4 + 4 запрета структур (правила ЗОЛОТА).
Использование:
    e3 = <import engine_v3>; e4 = <import engine_v4>; e5 = <import engine_v5>
    E = e3.build_engine('xau_1h_master.csv')
    trades = e5.run_v5(E, e4)   # DataFrame сделок
Эталон (золото 9 лет): 939 сделок, +53.2R (train +31.2 / test +22.0), WR 20%.
ВНИМАНИЕ: на BTC/серебре пакет запретов НЕ работает. Не переносить без проверки.
НЕ сделано: контроль случайностью, помесячная развёртка, walk-forward.
"""
import numpy as np
import pandas as pd
import bisect

BANNED_TAILS = ('LL>HH', 'LL>LH>HL', 'HH>LH', 'HH>LL>LL')


def banned_tail(labels):
    """Возвращает имя запрета, если хвост последовательности меток запрещён, иначе None."""
    s = '>'.join(labels)
    for b in BANNED_TAILS:
        if s.endswith(b):
            return b
    return None


def decide_v5(E, i, e4):
    """Решение v5 на триггерном баре i: v4 + запреты структур."""
    d = e4.decide_v4(E, i)
    if d.get('decision') != 'ВХОД':
        return d
    b = banned_tail(d['labels'])
    if b is not None:
        d['decision'] = 'запрет ' + b
    return d


def run_v5(E, e4):
    """Полный прогон v5: DataFrame сделок."""
    df = E['df']; TRIG = E['TRIG']
    rows = []
    for i in np.where(TRIG)[0]:
        d = decide_v5(E, i, e4)
        if d.get('decision') != 'ВХОД':
            continue
        is_long = d['dir'] == 'LONG'
        o = e4.resolve(E, i, is_long, d['stop'], d['tp'])
        if o not in ('TP', 'SL'):
            continue
        R = d['RR'] if o == 'TP' else -1.0
        rows.append(dict(dt=df['time'].iloc[i], i=i, dir=d['dir'], last=d['labels'][-1],
                         entry=d['entry'], stop=d['stop'], tp=d['tp'],
                         RR=round(d['RR'], 2), out=o, R=R))
    return pd.DataFrame(rows)
