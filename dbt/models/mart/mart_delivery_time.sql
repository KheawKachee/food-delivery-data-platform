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

joined as (

    select
        o.order_id,
        o.order_ts,
        (o.delivered_ts - o.food_ready_ts) as delivery_time,
        o.distance_km,
        u.user_zone,
        r.rider_zone,
        avg(o.rider_rating) over (partition by o.rider_id) as avg_rider_rating
    from orders o
    left join users u on o.user_id = u.user_id
    left join riders r on o.rider_id = r.rider_id

)

select *
from joined
