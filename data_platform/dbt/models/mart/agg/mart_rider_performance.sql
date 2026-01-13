{{ config(
    materialized = 'incremental',
    unique_key = 'rider_id'
) }}

with orders as (

    select
        rider_id,
        order_id,
        rider_rating
    from {{ ref('staging_orders') }}
    {% if is_incremental() %}
    WHERE rider_id > (SELECT rider_id FROM {{ this }})
    {% endif %}
),

agg as (

    select
        rider_id,
        count(order_id)                as n_jobs,
        avg(rider_rating)              as avg_rider_rating
    from orders
    group by rider_id

),

riders as (

    select
        rider_id,
        zone as rider_zone
    from {{ ref('staging_riders') }}

)

select
    r.rider_id,
    r.rider_zone,
    coalesce(a.n_jobs, 0)           as n_jobs,
    coalesce(a.avg_rider_rating, 0) as avg_rider_rating
from riders r
left join agg a
    on r.rider_id = a.rider_id
