{{ config(
    materialized = 'incremental',
    unique_key = 'rider_id',
    incremental_strategy = 'merge'
) }}

with orders as (

    select
        rider_id,
        order_id,
        rider_rating,
        updated_at
    from {{ ref('staging_orders') }}
),

agg as (

    select
        rider_id,
        count(order_id)                as n_jobs,
        avg(rider_rating)              as avg_rider_rating,
        max(updated_at) as updated_at
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
    coalesce(a.avg_rider_rating, 0) as avg_rider_rating,
    a.updated_at
from riders r
left join agg a
    on r.rider_id = a.rider_id
{% if is_incremental() %}
where updated_at > (select coalesce(max(updated_at),'1900-01-01'::timestamp) from {{ this }})
{% endif %}
