{{ config(
    materialized = 'incremental',
    unique_key = 'order_id',
    incremental_strategy = 'merge'
) }}

WITH source_data AS (
    select
        (payload->>'order_id')::bigint       as order_id,
        (payload->>'user_id')::bigint         as user_id,
        (payload->>'rider_id')::bigint       as rider_id,
        (payload->>'order_ts')::timestamp    as order_ts,
        (payload->>'food_ready_ts')::timestamp as food_ready_ts,
        (payload->>'delivered_ts')::timestamp  as delivered_ts,
        (payload->>'distance_km')::numeric   as distance_km,
        (payload->>'price_baht')::numeric    as price_baht,
        nullif(payload->>'rider_rating','')::numeric as rider_rating,
        ingest_ts::timestamp                 as updated_at
    from {{ source('raw', 'raw_orders') }}

    {% if is_incremental() %}
    where ingest_ts::timestamp > (select coalesce(max(updated_at),'1900-01-01'::timestamp) from {{ this }})
    {% endif %}
),

deduplicated AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY order_id 
            ORDER BY updated_at DESC 
        ) AS rn
    FROM source_data
)

SELECT
    order_id,
    user_id,
    rider_id,
    order_ts,
    food_ready_ts,
    delivered_ts,
    distance_km,
    price_baht,
    rider_rating,
    updated_at
FROM deduplicated
WHERE rn = 1 
  AND order_id IS NOT NULL