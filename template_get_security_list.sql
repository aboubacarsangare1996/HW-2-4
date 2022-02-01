WITH tt as (
    SELECT DISTINCT security_code
    FROM stock_orders
)
SELECT
    security_code,
    CASE
        WHEN security_code ~ '.*P$'
            THEN 'preferred' -- насколько я понял, у привилегированных акций тикер заканчивается на 'P'
        WHEN security_code ~ '^\D.*\d$'
            THEN 'bond' -- у облигаций начинается на букву и заканчивается на цифру
        ELSE 'common' -- все остальное
    END     security_type
FROM tt