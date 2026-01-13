{{ config(
    materialized = 'incremental',
    unique_key = 'order_id'
) }}

WITH source_data AS (
    SELECT
        (payload::jsonb->>'order_id')::bigint               AS order_id,
        (payload::jsonb->>'user_id')::bigint                AS user_id,
        (payload::jsonb->>'rider_id')::bigint               AS rider_id,
        NULLIF(payload::jsonb->>'order_ts','')::timestamp   AS order_ts,
        NULLIF(payload::jsonb->>'food_ready_ts','')::timestamp AS food_ready_ts,
        NULLIF(payload::jsonb->>'delivered_ts','')::timestamp  AS delivered_ts,
        (payload::jsonb->>'distance_km')::numeric           AS distance_km,
        (payload::jsonb->>'price_baht')::numeric            AS price_baht,
        NULLIF(payload::jsonb->>'rider_rating','')::numeric AS rider_rating
    FROM {{ source('raw', 'raw_orders') }}

    {% if is_incremental() %}
    WHERE (payload::jsonb->>'order_id')::bigint > (SELECT MAX(order_id) FROM {{ this }})
    {% endif %}
),

deduplicated AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY order_id 
            ORDER BY order_ts DESC 
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
    rider_rating
FROM deduplicated
WHERE rn = 1 
  AND order_id IS NOT NULL