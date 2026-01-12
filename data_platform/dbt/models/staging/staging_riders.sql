{{ config(materialized='incremental', unique_key='rider_id') }}

select
    (payload->>'rider_id')::bigint as rider_id,
    (payload->>'signup_date')::timestamp as signup_date,
    (payload->>'zone')::VARCHAR as zone
from {{ source('raw', 'raw_riders') }}
