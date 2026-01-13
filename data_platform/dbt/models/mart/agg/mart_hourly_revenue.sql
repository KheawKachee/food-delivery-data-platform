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
    
    {% if is_incremental() %}
    -- Use the full expression here, not the alias 'hourly'
    WHERE date_trunc('hour', order_ts) > (SELECT max(hourly) FROM {{ this }})
    {% endif %}
),

agg as (
    select
        hourly,
        count(order_id) as n_orders,
        sum(price_baht) as total_price_baht
    from orders
    group by 1
)

select * from agg