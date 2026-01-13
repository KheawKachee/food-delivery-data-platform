{{ config(
    materialized = 'incremental',
    unique_key = 'order_id'
) }}

with orders as (

    select
        order_id,
        user_id,
        rider_id,
        order_ts,
        food_ready_ts,
        delivered_ts,
        distance_km,
        rider_rating
    from {{ ref('staging_orders') }}

),

users as (

    select
        user_id,
        zone as user_zone
    from {{ ref('staging_users') }}

),

riders as (

    select
        rider_id,
        zone as rider_zone
    from {{ ref('staging_riders') }}

),

base as (

    select
        o.order_id,
        o.order_ts,
        o.user_id,
        o.rider_id,

        -- core target
        (o.delivered_ts - o.food_ready_ts) as delivery_time,

        -- label for ops decision
        case
            when (o.delivered_ts - o.food_ready_ts) > interval '15 minutes'
            then 1 else 0
        end as is_delayed,

        -- core features
        o.distance_km,
        u.user_zone,
        r.rider_zone,

        -- temporal features , hour and dayofweek
        extract(hour from o.order_ts) as order_hour,
        extract(dow from o.order_ts) as order_dow,

        -- historical and not leakage
        avg(o.rider_rating) over (
            partition by o.rider_id
            order by o.order_ts
            rows between unbounded preceding and 1 preceding
        ) as avg_rider_rating_hist

    from orders o
    left join users u on o.user_id = u.user_id
    left join riders r on o.rider_id = r.rider_id
)

select *
from base

{% if is_incremental() %}
where order_id > (select max(order_id) from {{ this }})
{% endif %}