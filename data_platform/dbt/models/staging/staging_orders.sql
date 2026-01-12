{{ config(
    materialized = 'incremental',
    unique_key = 'order_id'
) }}

select
    (payload->>'order_id')::bigint       as order_id,
    (payload->>'user_id')::bigint        as user_id,
    (payload->>'rider_id')::bigint       as rider_id,
    (payload->>'order_ts')::timestamp    as order_ts,
    (payload->>'food_ready_ts')::timestamp as food_ready_ts,
    (payload->>'delivered_ts')::timestamp  as delivered_ts,
    (payload->>'distance_km')::numeric   as distance_km,
    (payload->>'price_baht')::numeric    as price_baht,
    nullif(payload->>'rider_rating','')::numeric as rider_rating
from {{ source('raw', 'raw_orders') }}

--- Incremental filter to only process new records
{% if is_incremental() %}
where (payload->>'order_ts')::timestamp >
      (select max(order_ts) from {{ this }})
{% endif %}

