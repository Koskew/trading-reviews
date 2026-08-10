"""
ДВИЖОК v3 — три правки по неделе 02-09.10.2025:
  1. Перцентильный фильтр свечи (80% распределения триггеров) вместо ATR-окна
  2. Слом HH->LL / LL->HL = МЯГКАЯ заморозка (до подтверждающей метки в новую сторону)
  3. Согласованность: вход против явного тренда (>=4/5 меток) запрещён, КРОМЕ случая слома
Собран по итогам разбора недели 7-15.04.2020.

КОНФИГУРАЦИЯ:
- Направление по ПОСЛЕДНЕЙ метке (не по преобладанию):
    HL -> лонг | LL -> лонг (разворотный отскок)
    LH -> шорт | HH -> шорт (разворотный откат)
- ЗАПРЕТ входа: шорт от LL, лонг от HH (метки истощения) -- подтверждён статистикой
- Стоп: за свечой +0.1% основной; считаются ВСЕ 4 варианта для статистики
- Тейк: первый непробитый магнит в свою сторону среди 5 ПОСЛЕДНИХ меток, RR>=3, иначе пропуск
- Заморозка: свежий экстремум (LL/HH) на триггере лаг 0-1 -> пропуск
- Одна позиция за раз
- Логирование (не фильтр): связки HH->LL и LL->HL как сломы
"""
import pandas as pd, numpy as np, bisect

GRID=5.0; HORIZON=1500; COST=0.20; BUF=0.001; FRESH_MARKS=5
ATR_WIN=5; ATR_MULT=1.5  # (старое, не используется в v3)
SIZE_PERCENTILE=80  # свеча больше 80-го перцентиля триггеров -> большая
TREND_MARKS=5; TREND_MIN=4  # явный тренд: >=4 из 5 меток в одну сторону

def fp(v,a,b,mode,N,H,L):
    out=[]
    for i in range(a,N-b):
        cc=v[i]
        if mode=='high':
            if cc>v[i-a:i].max() and cc>v[i+1:i+b+1].max(): out.append(i)
        else:
            if cc<v[i-a:i].min() and cc<v[i+1:i+b+1].min(): out.append(i)
    return out
def mk(a,b,N,H,L):
    rows=[]; prev=None
    for i in fp(H,a,b,'high',N,H,L):
        if prev is not None and H[i]!=prev: rows.append(dict(bar=i,conf=i+b,label='HH' if H[i]>prev else 'LH',price=H[i],side='high'))
        prev=H[i]
    prev=None
    for i in fp(L,a,b,'low',N,H,L):
        if prev is not None and L[i]!=prev: rows.append(dict(bar=i,conf=i+b,label='HL' if L[i]>prev else 'LL',price=L[i],side='low'))
        prev=L[i]
    rows.sort(key=lambda r:(r['conf'],r['bar'])); return rows

def build_engine(path):
    df=pd.read_csv(path); df['time']=pd.to_datetime(df['time'],utc=True)
    df['kyiv']=df['time'].dt.tz_convert('Europe/Kyiv')
    O,H,L,C=df['open'].values,df['high'].values,df['low'].values,df['close'].values; N=len(df)
    TRIG=(df['триггер'].values==1) if 'триггер' in df.columns else np.zeros(N,bool)
    trr=np.maximum(H-L,np.maximum(np.abs(H-np.roll(C,1)),np.abs(L-np.roll(C,1)))); trr[0]=H[0]-L[0]
    atr=pd.Series(trr).rolling(14).mean().values; csize=H-L
    # перцентильный порог размера триггерной свечи (в % от цены)
    _ti=np.where(TRIG)[0]
    _sz=csize[_ti]/C[_ti]*100 if len(_ti)>0 else np.array([1.0])
    size_thr_pct=np.percentile(_sz,SIZE_PERCENTILE)
    fast=mk(2,2,N,H,L); confs=[r['conf'] for r in fast]
    full=fast  # все метки по порядку подтверждения
    def fh(cond,s):
        seg=cond[s:s+HORIZON]
        if len(seg)==0: return -1
        w=int(np.argmax(seg)); return s+w if seg[w] else -1
    def resolve(i,is_long,sl,tp):
        s=i+1
        itp=fh(H>=tp,s) if is_long else fh(L<=tp,s); isl=fh(L<=sl,s) if is_long else fh(H>=sl,s)
        if itp==-1 and isl==-1: return ('NONE',-1,-1)
        return (('TP' if (isl==-1 or (itp!=-1 and itp<isl)) else 'SL'), itp, isl)
    def stop_variants(i,is_long,entry,lastprice):
        if is_long:
            lvl=np.floor(entry/GRID)*GRID
            while lvl>=L[i]: lvl-=GRID
            return {'свеча':L[i]*(1-BUF),'уровень':lvl*(1-BUF),'метка':lastprice*(1-BUF),'хвост':L[i]*(1-BUF)}
        else:
            lvl=np.ceil(entry/GRID)*GRID
            while lvl<=H[i]: lvl+=GRID
            return {'свеча':H[i]*(1+BUF),'уровень':lvl*(1+BUF),'метка':lastprice*(1+BUF),'хвост':H[i]*(1+BUF)}
    return dict(df=df,O=O,H=H,L=L,C=C,N=N,TRIG=TRIG,atr=atr,csize=csize,
                fast=fast,confs=confs,fh=fh,resolve=resolve,stop_variants=stop_variants,
                size_thr_pct=size_thr_pct)

def decide_point(E, i, stopmode='свеча'):
    """Возвращает решение движка на жёлтой точке i."""
    H,L,C,fast,confs=E['H'],E['L'],E['C'],E['fast'],E['confs']
    csize,atr=E['csize'],E['atr']
    k=bisect.bisect_right(confs,i)
    seq=fast[max(0,k-FRESH_MARKS):k]  # 5 последних меток
    if not seq: return dict(decision='ПРОПУСК', reason='нет меток')
    last=seq[-1]; lastlabel=last['label']; lastprice=last['price']; lastbar=last['bar']
    labels=[r['label'] for r in seq]
    # направление по последней метке + запрет истощения
    if lastlabel=='HL': is_long=True
    elif lastlabel=='LH': is_long=False
    elif lastlabel=='LL': is_long=True   # разворотный лонг; шорт от LL ЗАПРЕЩЁН
    elif lastlabel=='HH': is_long=False  # разворотный шорт; лонг от HH ЗАПРЕЩЁН
    else: return dict(decision='ПРОПУСК', reason='метка?')
    # ПЕРЦЕНТИЛЬНЫЙ фильтр размера свечи (vs распределение всех триггеров)
    szpct=E['csize'][i]/E['C'][i]*100
    if szpct>E['size_thr_pct']:
        return dict(decision='ПРОПУСК', reason=f'большая свеча {szpct:.3f}% (>{SIZE_PERCENTILE}-й перц {E["size_thr_pct"]:.3f}%)', dir='LONG' if is_long else 'SHORT')
    # заморозка: свежий экстремум на триггере (лаг 0-1)
    if lastlabel in ('LL','HH'):
        lag = i - last['conf']
        if lag<=1:
            return dict(decision='ПРОПУСК', reason=f'заморозка: свежий {lastlabel} лаг {lag}', dir='LONG' if is_long else 'SHORT')
    # слом и согласованность
    slom=None
    if len(labels)>=2:
        if labels[-2]=='HH' and labels[-1]=='LL': slom='HH->LL (слом вверх)'
        if labels[-2]=='LL' and labels[-1]=='HL': slom='LL->HL (слом вниз)'
    # явный тренд по последним TREND_MARKS меткам
    up=sum(1 for x in labels[-TREND_MARKS:] if x in ('HH','HL'))
    dn=sum(1 for x in labels[-TREND_MARKS:] if x in ('LH','LL'))
    trend='UP' if up>=TREND_MIN else ('DN' if dn>=TREND_MIN else None)
    # был ли слом где-то в последних метках (тогда тренд не считается явным)
    had_slom=False
    for j in range(1,len(labels)):
        a,b=labels[j-1],labels[j]
        if (a=='HH' and b=='LL') or (a=='LL' and b=='HL'): had_slom=True
    # ПРАВКА 3: вход против явного тренда запрещён, КРОМЕ слома
    if trend is not None and not had_slom:
        if trend=='UP' and not is_long:
            return dict(decision='ПРОПУСК', reason=f'против тренда: шорт в аптренде ({up}/5)', dir='SHORT', slom=slom)
        if trend=='DN' and is_long:
            return dict(decision='ПРОПУСК', reason=f'против тренда: лонг в даунтренде ({dn}/5)', dir='LONG', slom=slom)
    # ПРАВКА 2: мягкая заморозка по слому — пока слом на самом конце (последняя пара),
    # ждём подтверждающую метку в новую сторону
    if len(labels)>=2:
        if labels[-2]=='HH' and labels[-1]=='LL':
            return dict(decision='ПРОПУСК', reason='слом HH->LL: жду подтверждения (LH)', dir='LONG' if is_long else 'SHORT', slom=slom)
        if labels[-2]=='LL' and labels[-1]=='HL':
            return dict(decision='ПРОПУСК', reason='слом LL->HL: жду подтверждения', dir='LONG' if is_long else 'SHORT', slom=slom)
    entry=C[i]
    sv=E['stop_variants'](i,is_long,entry,lastprice)
    sl=sv[stopmode]
    if (is_long and sl>=entry) or ((not is_long) and sl<=entry):
        return dict(decision='ПРОПУСК', reason='стоп с неправильной стороны')
    risk=abs(entry-sl)
    # магнит среди 5 последних меток в свою сторону
    side_needed='high' if is_long else 'low'
    cand=[r for r in seq if r['side']==side_needed]
    tp=np.nan; tplabel=None
    if is_long:
        for r in sorted([r for r in cand if r['price']>entry], key=lambda r:r['price']):
            if (r['price']-entry)/risk>=3: tp=r['price']; tplabel=r['label']; break
    else:
        for r in sorted([r for r in cand if r['price']<entry], key=lambda r:-r['price']):
            if (entry-r['price'])/risk>=3: tp=r['price']; tplabel=r['label']; break
    if np.isnan(tp):
        return dict(decision='ПРОПУСК', reason='нет свежего магнита RR>=3', dir='LONG' if is_long else 'SHORT', slom=slom)
    rr=abs(tp-entry)/risk
    atr_x = csize[i]/atr[i] if atr[i]>0 else 0
    out,itp,isl=E['resolve'](i,is_long,sl,tp)
    return dict(decision='ВХОД', dir='LONG' if is_long else 'SHORT',
                entry=round(entry,2), stop=round(sl,2), take=round(tp,2), RR=round(rr,2),
                tplabel=tplabel, out=out, lastlabel=lastlabel, atr_x=round(atr_x,2),
                struct='>'.join(labels), slom=slom, allstops={kk:round(vv,2) for kk,vv in sv.items()})

# самотест на прошлой неделе
if __name__=='__main__':
    E=build_engine('/home/claude/xau_1h_master.csv')
    print('Движок v3 собран. Самотест на неделе 7-15.04.2020:')
    print(f"Порог размера свечи (80-й перцентиль): {E['size_thr_pct']:.3f}% от цены")
    t0=pd.Timestamp('2020-04-07 13:00',tz='Europe/Kyiv').tz_convert('utc')
    t1=pd.Timestamp('2020-04-15 17:00',tz='Europe/Kyiv').tz_convert('utc')
    df=E['df']
    lo=df.index[df['time']>=t0][0]; hi=df.index[df['time']<=t1][-1]
    cnt_in=cnt_skip=0
    for i in range(lo,hi+1):
        if not E['TRIG'][i]: continue
        d=decide_point(E,i)
        tm=df['kyiv'].iloc[i].strftime('%m-%d %H:%M')
        if d['decision']=='ВХОД':
            cnt_in+=1
            print(f"  {tm} ВХОД {d['dir']:5s} вход {d['entry']} стоп {d['stop']} тейк {d['take']} RR {d['RR']} -> {d['out']} | {d['struct']}")
        else:
            cnt_skip+=1
    print(f'\nВходов: {cnt_in} | пропусков: {cnt_skip}')
