# -*- coding: utf-8 -*-
import csv, html, os

rows = list(csv.DictReader(open('lombard_top15.csv', encoding='utf-8-sig')))

NBSP = ' '  # narrow no-break space as thousands separator
def num(v, dec=1):
    if v in (None, '', 'n/a'): return None
    try: f = float(v)
    except ValueError: return None
    s = f'{f:,.{dec}f}'.replace(',', NBSP).replace('.', ',')
    return s
def esc(s): return html.escape(s or '')

def portfolio(r):
    """returns (display, is_estimate)"""
    if r['portfolio_30_06_2025_bn']:
        return num(r['portfolio_30_06_2025_bn']), False
    if r['portfolio_30_06_2025_est_bn']:
        return num(r['portfolio_30_06_2025_est_bn']), True
    return None, False

NA = '<span class="na">&mdash;</span>'
def cell(v, cls=''):
    inner = v if v is not None else NA
    return '<td class="' + cls + '">' + str(inner) + '</td>'

GROUP = 'М-Ломбард Group'

# ---------- table 1: prospect view ----------
t1 = []
for r in rows:
    p, est = portfolio(r)
    pts = num(r['retail_points'], 0)
    ppp = num(r['portfolio_per_point_mn_kzt'], 0)
    cl  = num(r['clients'], 0)
    krp = (r['headcount_band'] or '').split('(')
    krp = krp[1].rstrip(') ') if len(krp) > 1 else ''
    grp = ' <span class="grp">◆</span>' if r['group'] == GROUP else ''
    t1.append(f'''<tr{' class="hi"' if r['group']==GROUP else ''}>
      <td class="r dim">{r['rank']}</td>
      <td class="name">{esc(r['brand'])}{grp}</td>
      <td class="bin">{esc(r['bin'])}</td>
      {cell((p + ('<span class="est">°</span>' if est else '')) if p else None, 'r strong')}
      {cell(pts, 'r')}
      {cell(num(r['registered_filials'],0), 'r dim')}
      {cell(num(r['cities_count'],0), 'r')}
      {cell(cl, 'r')}
      <td class="krp">{esc(krp)}</td>
      {cell(ppp, 'r')}
    </tr>''')

# ---------- table 2: financial view ----------
t2 = []
for r in rows:
    t2.append(f'''<tr{' class="hi"' if r['group']==GROUP else ''}>
      <td class="r dim">{r['rank']}</td>
      <td class="name">{esc(r['brand'])}</td>
      {cell(num(r['assets_01_10_2024_bn']), 'r')}
      {cell(num(r['portfolio_01_10_2024_bn']), 'r')}
      {cell(num(r['equity_01_10_2024_bn']), 'r')}
      {cell(num(r['asset_share_01_04_2025_pct']), 'r')}
      {cell(num(r['net_profit_9m_2024_bn'],2), 'r')}
      {cell(num(r['net_profit_h1_2025_bn'],1), 'r')}
      {cell(num(r['net_profit_q1_2026_bn'],1), 'r strong')}
      {cell(num(r['taxes_paid_2025_bn'],2), 'r')}
    </tr>''')

HTML = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Ломбарды РК — топ-15</title>
<style>
@page {{ size: A4 landscape; margin: 11mm 10mm 13mm 10mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "DejaVu Sans", "Liberation Sans", sans-serif; font-size: 7.6pt;
        color: #16181d; margin: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
h1 {{ font-size: 15pt; margin: 0 0 1mm; letter-spacing: -.2px; }}
h2 {{ font-size: 9.5pt; margin: 0 0 2mm; padding-bottom: 1.2mm;
      border-bottom: 1.6px solid #16181d; letter-spacing: .3px; text-transform: uppercase; }}
.sub {{ color: #5c6370; font-size: 8pt; margin: 0 0 4mm; }}
.kpis {{ display: flex; gap: 3mm; margin: 0 0 4.5mm; }}
.kpi {{ flex: 1; border: 1px solid #dcdfe5; border-left: 2.5px solid #16181d;
        padding: 2mm 2.6mm; background: #fafbfc; }}
.kpi .v {{ font-size: 12pt; font-weight: 700; font-variant-numeric: tabular-nums; }}
.kpi .l {{ font-size: 6.6pt; color: #5c6370; text-transform: uppercase; letter-spacing: .4px; margin-top: .6mm; }}
.kpi .d {{ font-size: 6.6pt; color: #8a929e; margin-top: .4mm; }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
thead th {{ font-size: 6.5pt; text-transform: uppercase; letter-spacing: .3px; color: #5c6370;
           font-weight: 600; text-align: left; padding: 0 1.6mm 1.5mm; border-bottom: 1px solid #c8ccd4;
           vertical-align: bottom; line-height: 1.2; }}
thead th.r {{ text-align: right; }}
thead th .u {{ display: block; font-size: 5.9pt; color: #9aa1ac; text-transform: none; letter-spacing: 0; }}
tbody td {{ padding: 1.45mm 1.6mm; border-bottom: .5px solid #eceef2; vertical-align: baseline; }}
tbody tr:nth-child(even) td {{ background: #fafbfc; }}
tbody tr.hi td {{ background: #fff8e8; }}
td.r {{ text-align: right; }}
td.name {{ font-weight: 600; }}
td.bin {{ font-family: "DejaVu Sans Mono", monospace; font-size: 6.8pt; color: #5c6370; letter-spacing: -.3px; }}
td.strong {{ font-weight: 700; }}
td.dim {{ color: #8a929e; }}
td.krp {{ font-size: 6.7pt; color: #5c6370; }}
.na {{ color: #c4c9d1; }}
.est {{ color: #b8860b; font-weight: 700; }}
.grp {{ color: #b8860b; font-size: 6pt; }}
.notes {{ margin-top: 4mm; font-size: 6.9pt; color: #4a505a; line-height: 1.5; }}
.notes b {{ color: #16181d; }}
.page {{ page-break-before: always; }}
.two {{ display: flex; gap: 6mm; }}
.two > div {{ flex: 1; }}
.two h3, .full h3 {{ font-size: 8pt; margin: 0 0 1.6mm; letter-spacing: .2px; }}
.box {{ border: 1px solid #dcdfe5; background: #fafbfc; padding: 2.6mm 3mm; font-size: 7pt; line-height: 1.55; }}
.box p {{ margin: 0 0 1.8mm; }} .box p:last-child {{ margin: 0; }}
ul {{ margin: 0 0 0 3.6mm; padding: 0; }} li {{ margin-bottom: 1.1mm; line-height: 1.45; }}
.src {{ font-size: 6.4pt; color: #8a929e; margin-top: 3mm; line-height: 1.5; }}
.foot {{ position: fixed; bottom: -8mm; left: 0; right: 0; font-size: 6.2pt; color: #9aa1ac;
         display: flex; justify-content: space-between; }}
small.t {{ font-size: 6.4pt; color: #8a929e; font-weight: 400; text-transform: none; letter-spacing: 0; }}
</style></head><body>

<h1>Ломбардный рынок Казахстана — топ-15 сетей</h1>
<p class="sub">Prospect-лист и калибровка цены для retail-analytics (60&#8239;000&#8239;₸/мес за точку). Данные собраны 14.09.2026.</p>

<div class="kpis">
  <div class="kpi"><div class="v">373</div><div class="l">ломбардов в реестре</div><div class="d">АРРФР, 27.07.2026 · было 470 в 06.2025</div></div>
  <div class="kpi"><div class="v">453,8 млрд ₸</div><div class="l">портфель сектора</div><div class="d">НБК, 01.07.2026 · −5,7% за квартал</div></div>
  <div class="kpi"><div class="v">54,7 млрд ₸</div><div class="l">прибыль за Q1 2026</div><div class="d">×2,2 г/г · это квартал, не год</div></div>
  <div class="kpi"><div class="v">~⅔</div><div class="l">портфеля у топ-7</div><div class="d">и &gt;80% прибыли сектора</div></div>
  <div class="kpi"><div class="v">2,5×</div><div class="l">разрыв №7 → №8</div><div class="d">14,8 → 5,9 млрд ₸</div></div>
</div>

<h2>Сеть и масштаб <small class="t">— то, от чего считается цена</small></h2>
<table>
<thead><tr>
  <th class="r">#</th><th>Компания</th><th>БИН</th>
  <th class="r">Портфель<span class="u">млрд ₸ · 30.06.2025</span></th>
  <th class="r">Точек<span class="u">розничных</span></th>
  <th class="r">Филиалов<span class="u">в реестре</span></th>
  <th class="r">Городов<span class="u">присутствия</span></th>
  <th class="r">Клиентов<span class="u">заявлено</span></th>
  <th>Персонал<span class="u">КРП, чел.</span></th>
  <th class="r">Портфель/точку<span class="u">млн ₸</span></th>
</tr></thead>
<tbody>{''.join(t1)}</tbody>
</table>

<div class="notes">
<b>°</b> — портфель оценочный (факт 01.10.2024 × 1,298 рост сектора; проверено на топ-7, ошибка от −15% до +11%). Без значка — факт.
&nbsp;·&nbsp; <b>◆</b> — М-Ломбард и МК-Ломбард суть одна группа.
&nbsp;·&nbsp; <b>«Филиалов в реестре» ≠ точки продаж</b>: филиалы регистрируются по регионам, а не по отделениям — см. стр. 3.
&nbsp;·&nbsp; «—» = публично не раскрывается.
</div>

<div class="page"></div>
<h2>Финансы <small class="t">— регуляторная отчётность</small></h2>
<table>
<thead><tr>
  <th class="r">#</th><th>Компания</th>
  <th class="r">Активы<span class="u">млрд ₸ · 01.10.24</span></th>
  <th class="r">Портфель<span class="u">млрд ₸ · 01.10.24</span></th>
  <th class="r">Капитал<span class="u">млрд ₸ · 01.10.24</span></th>
  <th class="r">Доля активов<span class="u">% · 01.04.25</span></th>
  <th class="r">Прибыль 9М&#8201;2024<span class="u">млрд ₸</span></th>
  <th class="r">Прибыль H1&#8201;2025<span class="u">млрд ₸</span></th>
  <th class="r">Прибыль Q1&#8201;2026<span class="u">млрд ₸</span></th>
  <th class="r">Налоги 2025<span class="u">млрд ₸ · прокси выручки</span></th>
</tr></thead>
<tbody>{''.join(t2)}</tbody>
</table>

<div class="notes">
<b>Выручка по компаниям публично не раскрывается ни одним источником.</b> НБК публикует активы, портфель, капитал и прибыль — но не доход;
отчёты о прибылях и убытках подают только эмитенты облигаций. Вместо выручки приведены уплаченные налоги (КГД) как прокси масштаба.
&nbsp;·&nbsp; Прибыль за H1&#8201;2025 известна по топ-7, за Q1&#8201;2026 — по топ-3: НБК прекратил публиковать пофирменные данные после 01.10.2024.
<br><b>Ранжирование — по портфелю.</b> «Актив Ломбард» по активам занял бы 9-е место (6,0 млрд ₸), но его портфель в 6 раз меньше активов (0,9 млрд ₸),
а <b>собственный капитал отрицательный (−0,9 млрд ₸)</b> — по профилю это не ломбардная розница и для prospect-листа кандидат сомнительный.
</div>

<div class="page"></div>
<h2>Что нужно знать до контрактования</h2>
<div class="two">
  <div>
    <h3>⚠ 603 точки против 49 филиалов — это не ошибка данных</h3>
    <div class="box">
      <p>Филиалы ТОО регистрируются <b>по регионам, а не по отделениям</b>. Прямое подтверждение — проспект облигаций ТОО «Ломбард «GoldFinMarket» на KASE: там перечислены филиалы «по Жамбылскому региону», «по Северному региону» и т.д. — <b>семь на всю страну</b>. Розничные точки как филиалы не регистрируются вовсе.</p>
      <p><b>Вывод:</b> считать сеть по реестру нельзя. Используйте число точек, перекрёстно проверяя его колонкой «Персонал (КРП)» — это независимый сигнал Бюро нацстатистики.</p>
    </div>
    <h3 style="margin-top:4mm">Расхождение «заявлено / в реестре»</h3>
    <table>
    <thead><tr><th>Компания</th><th class="r">Точек</th><th class="r">Филиалов</th><th class="r">Персонал</th></tr></thead>
    <tbody>
      <tr><td class="name">М-Ломбард</td><td class="r">603</td><td class="r dim">49</td><td class="r">&gt;1000</td></tr>
      <tr><td class="name">МК-Ломбард</td><td class="r"><span class="na">не раскр.</span></td><td class="r dim">57</td><td class="r">&gt;1000</td></tr>
      <tr><td class="name">Birinshi Lombard</td><td class="r">420</td><td class="r dim">18</td><td class="r">&gt;1000</td></tr>
      <tr><td class="name">Астра-Ломбард</td><td class="r">400+</td><td class="r dim">21</td><td class="r">501–1000</td></tr>
      <tr><td class="name">Деньги населению</td><td class="r">324</td><td class="r dim">20</td><td class="r">251–500</td></tr>
      <tr><td class="name">МК-Золото Ломбард</td><td class="r">144</td><td class="r dim">29</td><td class="r">251–500</td></tr>
      <tr><td class="name">Сейф-Ломбард</td><td class="r">140+</td><td class="r dim"><b>9</b></td><td class="r">251–500</td></tr>
    </tbody></table>
  </div>
  <div>
    <h3>◆ Группы из нескольких ТОО</h3>
    <div class="box">
      <p><b>М-Ломбард + МК-Ломбард — одна группа.</b> Один директор (Алиев Г.&#8201;А.), одна форма собственности (иностранные юрлица), соседние адреса — Алматы, ул.&#8239;Попова, 33 и 29.</p>
      <p>Совокупно: <b>портфель 80,6 млрд ₸ (25% рынка)</b>, прибыль Q1&#8201;2026 — <b>23,4 млрд ₸ (43% прибыли всего сектора)</b>, 106 филиалов, 603 точки. Контракт на сеть — это два юрлица, не одно.</p>
      <p><b>МК-Золото Ломбард в эту группу не входит</b>, несмотря на префикс «МК»: другой директор (Осипова Л.&#8201;В.), другая форма собственности, другой адрес. До подтверждения — отдельный контрагент.</p>
      <p>Мультиюрлицо также у: Бөлім-Ломбард (2 ТОО), Ломбард Белый (3), Партнер (2), Астана Ломбард-7/8/9 (3), €lite (2), Серт (2).</p>
    </div>
    <h3 style="margin-top:4mm">Проверка цифр из брифа</h3>
    <div class="box">
    <ul>
      <li><b>Подтверждено:</b> 470 ломбардов (06.2025) · портфель 321,5 млрд ₸ · +40,2% г/г · прибыль Q1&#8201;2026 54,7 млрд ₸ ×2,2 · топ-7 &gt;⅔ портфеля · все 7 портфелей топ-7 поштучно.</li>
      <li><b>Правка:</b> ссылка KASE <i>MFLG</i> — это не Сейф-Ломбард, а ТОО «Ломбард «GoldFinMarket» (БИН 230140025888). Хост <i>old.kase.kz</i> отдаёт 502 — рабочий <i>kase.kz</i>.</li>
      <li><b>Правка:</b> регулятор — АРРФР, не НБК. В реестре сейчас 373 ломбарда против 470 в брифе.</li>
      <li><b>Уточнение:</b> Сейф-Ломбард заявляет уже 140+ отделений (сайт, 09.2026), не 130+.</li>
      <li><b>Не найдено:</b> SUPER Ломбард под этим именем в реестре АРРФР и отчётности НБК не значится — проверить юрлицо.</li>
    </ul>
    </div>
  </div>
</div>

<p class="src"><b>Источники по полям.</b> БИН, юр. адрес, ОКЭД, директор, форма собственности, филиалы, КРП, налоги — ba.prg.kz (агрегатор КГД / egov / Бюро нацстатистики).
Активы, портфель, капитал, прибыль на 2024 — Нацбанк РК, файл 110308 «Сведения о ломбардах РК» (пофирменный xlsx, 491 компания).
Портфель и прибыль 2025–2026 — ranking.kz и Kapital.kz. Доли рынка на 01.04.2025 — проспект облигаций ТОО «Ломбард «GoldFinMarket», KASE, 27.08.2025.
Точки, города, клиенты — сайты компаний и геокодированный справочник profinance.kz (сверен с реестром АРРФР).</p>

</body></html>'''

open('report.html', 'w', encoding='utf-8').write(HTML)
print('html written', len(HTML), 'bytes')
