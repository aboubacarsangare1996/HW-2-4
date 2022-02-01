-- выделить по часам средний оборот и суммарный оборот
SELECT
    extract(HOUR from order_time) AS order_hour,
    security_code,
    avg(volume * trade_price) AS avg_turnover,
    sum(volume * trade_price) AS sum_turnover
FROM stock_orders
WHERE action = 2 and buysell = 'B'
    AND security_code in {security_code} -- <== security_code
GROUP BY order_hour, security_code
