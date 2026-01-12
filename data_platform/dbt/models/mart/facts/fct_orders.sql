{{ config(
    materialized = 'incremental',
    unique_key = 'order_id'
) }}

select
    o.order_id,
    o.order_ts,
    o.user_id,
    o.rider_id,
    o.distance_km,
    o.price_baht,
    (o.delivered_ts - o.food_ready_ts) as delivery_time
from {{ ref('staging_orders') }} o

{% if is_incremental() %}
where o.order_ts > (select max(order_ts) from {{ this }})
{% endif %}
