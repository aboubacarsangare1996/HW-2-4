-- RAW TAB нет обработки от нулей и отрицательных объёмов
SELECT buysell, -- тип заявки
       order_no, -- номер заявки, нужно понять оставшийся объем по этой заявке
       price, -- цена заявки
       SUM(CASE
                WHEN action = 0 THEN -volume -- снятие заявки
                WHEN action = 1 THEN volume -- постановка заявки
                WHEN action = 2 THEN -volume -- исполнение заявки
           END) remaining_volume -- остаточный объём
FROM stock_orders
WHERE security_code = '{security_code}' -- <== security_code
    AND order_time > '{time1}' -- <====== time_1 (start_time)
    AND order_time <= '{time2}' -- <======= time_2 (end_time)
GROUP BY buysell,
         order_no,
         price
HAVING SUM(CASE
                WHEN action = 0 THEN -volume -- снятие заявки
                WHEN action = 1 THEN volume -- постановка заявки
                WHEN action = 2 THEN -volume -- исполнение заявки
           END) != 0

