{{ config(
    materialized = 'incremental',
    unique_key = 'hourly'
) }}

with orders as (

    select
        date_trunc('hour', order_ts) as hourly,
        order_id,
        price_baht
    from {{ ref('staging_orders') }}

),

agg as (

    select
        hourly,
        count(order_id) as n_orders,
        sum(price_baht) as total_price_baht
    from orders
    group by hourly

)

select *
from agg
order by hourly
