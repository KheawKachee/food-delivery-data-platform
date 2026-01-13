{{ config(
    materialized = 'incremental',
    unique_key = 'order_id',
    incremental_strategy = 'merge'
) }}

select
    o.order_id,
    o.order_ts,
    o.user_id,
    o.rider_id,
    o.distance_km,
    o.price_baht,
    (o.delivered_ts - o.food_ready_ts) as delivery_time,
    o.updated_at as updated_at
from {{ ref('staging_orders') }} o

{% if is_incremental() %}
where o.updated_at > (select coalesce(max(updated_at),'1900-01-01'::timestamp) from {{ this }})
{% endif %}