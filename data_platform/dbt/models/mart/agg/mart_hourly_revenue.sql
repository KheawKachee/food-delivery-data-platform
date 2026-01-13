{{ config(
    materialized = 'incremental',
    unique_key = 'hourly',
    incremental_strategy = 'merge'
) }}

with orders as (
    select
        date_trunc('hour', order_ts) as hourly,
        order_id,
        price_baht, 
        updated_at
    from {{ ref('staging_orders') }}
    
    {% if is_incremental() %}
    where date_trunc('hour', order_ts) >= (select coalesce(max(hourly), '1900-01-01'::timestamp) from {{ this }})
    {% endif %}
),

agg as (
    select
        hourly,
        count(order_id) as n_orders,
        sum(price_baht) as total_price_baht,
        max(updated_at) as updated_at
    from orders
    group by 1
)

select * from agg