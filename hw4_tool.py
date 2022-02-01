# Импорт библиотек

# RUN IN CMD (TERMINAL) `python hw4_tool.py` in project directory

import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import psycopg2

from datetime import datetime, timedelta
import it4fin_functions as it4fin  # мои функции выделенные в отдельный модуль

# Создание соединения с базой данных
cnxn = psycopg2.connect(user='postgres',
                        database='postgres',
                        host='localhost',
                        port='5432',
                        password='12345')
# print(cnxn.get_dsn_parameters())

# проверочный запрос
# print(pd.read_sql_query("SELECT * FROM stock_orders LIMIT 1", cnxn))

# список тикеров
security_list = pd.read_sql_query("SELECT DISTINCT security_code FROM stock_orders", cnxn)
# print(security_list)

# Инициализация дашборда
hw4_tool = dash.Dash(__name__)

hw4_tool.layout = html.Div(
    children=[html.H1(children='Бид аск спред ценных бумаг компаний (Никита Царьков МФР 212)'),
              html.P(
                  children='Визуализация бид аск спреда для заданного периода времени для ценных бумаг компаний, торгующихся на Московской бирже, вывод тренда'),
              # фильтр по компаниям
              html.Div(
                  children=[html.Div(children='Выберите компанию'),
                            dcc.Dropdown(
                                id="security_code",
                                options=[
                                    {'label': sec, 'value': sec}
                                    for sec in np.sort(security_list['security_code'])
                                ],
                                value='YNDX'
                            )
                            ]
              ),
              # фильтр по времени (Time Picker нет, DatePicker не поддерживает время)
              html.Div(
                  children=[html.Div(children='Выберите начальное время'),
                            dcc.Input(
                                id='start_time',
                                placeholder='12:34:56.789 или 12:34:56',
                                type='text',
                                value='12:00:00'
                            )
                            ]
              ),
              # фильтр по времени (Time Picker нет, DatePicker не поддерживает время)
              html.Div(
                  children=[html.Div(children='Выберите конечное время'),
                            dcc.Input(
                                id='end_time',
                                placeholder='12:34:56.789 или 12:34:56',
                                type='text',
                                value='13:00:00'
                            )
                            ]
              ),

              # частота
              html.Div(
                  children=[html.Div(children='Выберите частоту'),
                            dcc.Input(
                                id='freq_str',
                                placeholder='100ms или 15s или 10min',
                                type='text',
                                value='15min'
                            )
                            ]
              ),

              # Пустой базовый график
              dcc.Graph(id="bas_graph")

              ])


@hw4_tool.callback(Output('bas_graph', 'figure'),
                   Input('security_code', 'value'),
                   Input('start_time', 'value'),
                   Input('end_time', 'value'),
                   Input('freq_str', 'value'))
def main_func(security_code, start_time, end_time, freq_str):
    freq = 0

    # чекаем формат записи частоты
    if freq_str is None:
        freq = 10 ** 15  # вариант чтобы сразу перейти в конец
    elif freq_str.endswith('ms'):
        freq = int(freq_str[:-2])
    elif freq_str.endswith('s'):
        freq = 1000 * int(freq_str[:-1])
    elif freq_str.endswith('min'):
        freq = 1000 * 60 * int(freq_str[:-3])

    bds, _ = it4fin.get_spread_series(cnxn,
                                      security_code=security_code,
                                      start_time=start_time,
                                      end_time=end_time,
                                      freq=freq)

    # линия тренда
    model = np.polyfit([i for i in range(1, len(bds) + 1)], bds.to_list(), 1)
    trend_func = np.poly1d(model)
    trend = trend_func([i for i in range(1, len(bds) + 1)])

    # честно говоря я очень сильно торопился
    figure = {"data":
                  [{"x": bds.index, "y": bds, "type": "line", "name": "bds", 'line': {'dash': 'solid'}},
                   {"x": bds.index, "y": trend, "type": "line", "name": "trend", 'line': {'dash': 'dash'}, 'marker': {'symbol': ''}}
                   ],
              "layout": {"title": f'Динамика бид аск спреда {security_code} {start_time} {end_time} {freq_str}'},
              'xaxis': {'anchor': 'Time', 'title': {'text': 'Time'}},
              'yaxis': {'anchor': 'Value', 'title': {'text': 'Value'}}
              }

    return figure


# Запуск дашборда
if __name__ == "__main__":
    hw4_tool.run_server()
