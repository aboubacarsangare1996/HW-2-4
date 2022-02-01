import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

sns.set()


def get_dom_from_raw(raw_frame):
    """
    Функция, преобразующая срез по заявкам до какого-то времени в биржевой стакан.

    :param raw_frame: `DataFrame`, имеющий столбцы `buysell` (тип заявки B/S), `order_no` (номер заявки), `price` (цена заявки), `remaining_volume` (неисполненный объём заявки)

    :returns: `DataFrame` вида (buy - price - sell), представляющий собой биржевой стакан

    """

    buy = raw_frame[(raw_frame['buysell'] == 'B') & (raw_frame['remaining_volume'] > 0.0)]
    sell = raw_frame[(raw_frame['buysell'] == 'S') & (raw_frame['remaining_volume'] > 0.0)]
    buy = (buy[['price', 'remaining_volume']].groupby(by='price').sum()).rename(columns={'remaining_volume': 'buy'})
    sell = (sell[['price', 'remaining_volume']].groupby(by='price').sum()).rename(columns={'remaining_volume': 'sell'})

    dom = (pd.merge(buy, sell, how='outer', on='price')).sort_values(by='price', ascending=False).fillna(0)

    bid = buy.index.max()
    ask = sell.index.min()
    return dom, bid, ask


def get_spread_series(cnxn, security_code, end_time, start_time=None, freq=None):
    '''
    Основная функция для выполнения Д/З

    P.S. Для больших интервалов считает долго, поэтому лучше не задавать интервал 2 часа и частоту 1000ms - будет считать около часа
    Можно еще через pandas делать, вырезая сначала таблицу между для основного временного интервала,
    но я боялся потенциальной ошибки OUT OF MEMORY

    :param cnxn: соединение с базой данных
    :param security_code: тикер для ценной бумаги, например, 'AFLT'
    :param start_time: (optional) начало временного интервала, в формате 'HH:MI:SS.mss'
    :param end_time: конец временного интервала в том же формате
    :param freq: (optional), частота в миллисекундах

    :returns: `DataFrame`, определяющий биржевой стакан для ценной бумаги (buy volume - price - sell volume) и бид аск спред как `int` в слyчае заданы только `end_time` и `security_code` . Если, в добавок, даны `start_time` и `freq` то так же возвращает `Series` бид-аск спредов для временного интервала между `start_time` и `end_time`

    '''

    # считываем шаблон запроса для получения таблицы
    with open('template_get_remaining_volumes.sql', 'r', encoding='utf-8') as f:
        get_remaining_volumes = f.read()

    # печатаем полученные на вход аргументы
    print(security_code, start_time, end_time, f'{freq} ms')

    # дальше ветка: если нет start_time или частота нулевая
    if start_time is None or freq is None or freq == 0:
        # то формируем запрос на получение таблицы заявок до времени end_time и выполняем его
        query = get_remaining_volumes.format(security_code = security_code,
                                             time1 = '0:00:00',
                                             time2 = end_time)
        raw = pd.read_sql_query(query, cnxn)

        # превращаем эту таблицу в стакан
        dom, bid, ask = get_dom_from_raw(raw)
        # возвращаем бид-аск спред и стакан
        return (ask - bid) * 2 / (ask + bid), dom

    # теперь когда заданы все аргументы
    # проверяем то как задан формат времени
    if '.' in start_time:
        stime = datetime.strptime(start_time, '%H:%M:%S.%f')
    else:
        stime = datetime.strptime(start_time, '%H:%M:%S')

    if '.' in end_time:
        etime = datetime.strptime(end_time, '%H:%M:%S.%f')
    else:
        etime = datetime.strptime(end_time, '%H:%M:%S')

    # создаём массивы для времени и спредов
    times, spreads = [], []

    query = get_remaining_volumes.format(security_code = security_code,
                                         time1 = '0:00:00',
                                         time2 = start_time)
    new_raw = pd.read_sql_query(query, cnxn)
    dom, bid, ask = get_dom_from_raw(new_raw)

    # добавляем в массивы координаты
    times.append(stime)
    spreads.append((ask - bid) * 2 / (ask + bid))

    while True:
        time1 = stime
        time2 = stime + timedelta(milliseconds=freq)
        if time2 > etime:
            time2 = etime

        # print(time1.strftime('%H:%M:%S.%f'), time2.strftime('%H:%M:%S.%f'))
        query = get_remaining_volumes.format(security_code = security_code,
                                             time1 = time1.strftime('%H:%M:%S.%f'),
                                             time2 = time2.strftime('%H:%M:%S.%f'))

        raw1 = new_raw
        raw2 = pd.read_sql_query(query, cnxn)

        if not raw2.empty:
            new_raw = pd.concat([raw1, raw2], axis=0, ignore_index=True).\
                groupby(by=["buysell", "order_no", "price"], as_index=False).sum()

            dom, bid, ask = get_dom_from_raw(new_raw)
            spreads.append((ask - bid) * 2 / (ask + bid))
        else:
            new_raw = raw1
            spreads.append(spreads[-1])

        stime = time2
        times.append(stime)

        if stime == etime:
            return pd.Series(data=spreads, index=times), dom


def plot_bid_ask_spread_series(cnxn, security_code, start_time, end_time, freq, show_trend=True, save=False):
    """
    Построение графика динамики бид аск спреда

    :param cnxn: соединение с базой данных
    :param security_code: тикер для ценной бумаги, например, 'AFLT'
    :param start_time: начало временного интервала, в формате 'HH:MI:SS.mss'
    :param end_time: конец временного интервала в том же формате
    :param freq: частота в миллисекундах
    :param show_trend: (optional) отображать на графике тренд
    :param save:(optional) сохранить как PNG
    """
    bds, _ = get_spread_series(cnxn,
                               security_code=security_code,
                               start_time=start_time,
                               end_time=end_time,
                               freq=freq)

    # некоторые описательные статистики
    diff = abs(bds.max() - bds.min())  # разброс
    mean = bds.mean()  # среднее
    median = bds.median()  # квантиль 0.5 или просто медиана

    if show_trend:
        # линия тренда
        model = np.polyfit([i for i in range(1, len(bds) + 1)], bds.to_list(), 1)
        trend_func = np.poly1d(model)
        trend = trend_func([i for i in range(1, len(bds) + 1)])

    # отрисовка графика
    fig = plt.figure()
    plt.plot(bds.index, bds, label='BDS') # динамика бид аск спреда

    # отрисовка тренда
    if show_trend:
        plt.plot(bds.index, trend, 'r--', label='Trend') # линия тренда

    plt.title(f'Bid Ask Spread Dynamics {security_code} {start_time} {end_time} {freq} ms')

    # подписи осей и легенда
    plt.xlabel('Time')
    plt.ylabel('BDS')
    plt.legend()

    # корректировка корявого вывода оси времени
    if freq < 60*1000:
        fmt = DateFormatter('%H:%M:%S')
    else:
        fmt = DateFormatter('%H:%M')
    # корректировка корявого вывода оси времени
    plt.gcf().axes[0].xaxis.set_major_formatter(fmt)

    # показать график
    plt.show()

    # вывод некоторых описательных статистик
    # (так как график и так перегружен, то нет смысла пихать туда и надписи)
    # вообще говоря это делается через `plt.text()`
    print(f'Mean\t = {mean}')
    print(f'Median\t = {median}')
    print(f'Min Max Diff = {diff}')

    # сохранить график если нужно
    if save:
        directory = input('Directory:')
        filename = input('Name:')

        # проверяем на валидность
        if directory is None or directory == '':
            directory = ''
        elif not directory.endswith('/'):
            directory += '/'

        if not filename.endswith('.png'):
            filename += '.png'

        # сохраняем
        fig.savefig(directory + filename)
        print(f'\nSaved as {directory + filename}')


def plot_turnover(cnxn, security_code, agg='avg', show_trend=True, save=False):
    """
    :param cnxn: соединение с базой данных
    :param security_code: `tuple` тикеров ценных бумаг, например, ('AFLT', 'YNDX')
    :param agg (default='avg') как аггрегировать (avg = среднее, sum = сумма)
    :param show_trend: (optional) отображать на графике тренд
    :param save:(optional) сохранить как PNG
    """

    with open('template_get_turnover.sql', 'r', encoding='utf-8') as f:
        get_turnover = f.read()

    # печатаем полученные на вход аргументы
    print(security_code)

    query = get_turnover.format(security_code=str(security_code))
    table = pd.read_sql_query(query, cnxn)

    # как аггрегируем
    if agg.lower() == 'avg':
        column = 'avg_turnover'
    elif agg.lower() == 'sum':
        column = 'sum_turnover'

    # ОБОРОТ
    fig1 = plt.figure()
    for sec in security_code:
        filter_table = table[table['security_code'] == sec] # фильтруем по тикеру

        # некоторые описательные статистики
        diff = abs(filter_table[column].max() - filter_table[column].min()) # разброс
        mean = filter_table[column].mean() # среднее
        median = filter_table[column].median() # квантиль 0.5 или просто медиана

        print(sec)
        print(f'Mean\t = {mean}')
        print(f'Median\t = {median}')
        print(f'Min Max Diff = {diff}')
        print()

        if show_trend:  # линия тренда
            model = np.polyfit([i for i in range(1, len(filter_table) + 1)], filter_table[column].to_list(), 1)
            trend_func = np.poly1d(model)
            trend = trend_func([i for i in range(1, len(filter_table) + 1)])

        # график объема
        plt.plot(filter_table['order_hour'], filter_table[column], label=f"{sec}")

        if show_trend:
            plt.plot(filter_table['order_hour'], trend, '--', label=f"{sec} trend")  # линия тренда

    plt.title(f"{agg.capitalize()} Turnover Dynamics {' '.join(security_code)}")

    # подписи осей и легенда
    plt.xlabel('Hour')
    plt.ylabel('Turnover')
    plt.legend()

    # показать график
    plt.show()

    # сохранить график если нужно
    if save:
        directory = input('Directory:')
        filename = input('Name:')

        # проверяем на валидность
        if directory is None or directory == '':
            directory = ''
        elif not directory.endswith('/'):
            directory += '/'

        if not filename.endswith('.png'):
            filename += '.png'

        # сохраняем
        fig1.savefig(directory + filename)
        print(f'\nSaved as {directory + filename}')

