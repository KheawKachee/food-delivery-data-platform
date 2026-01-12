{{ config(materialized='table', unique_key='rider_id') }}

select
    (payload->>'rider_id')::bigint as rider_id,
    (payload->>'signup_date')::timestamp as signup_date,
    (payload->>'zone')::varchar as zone,
    ingest_ts
from {{ source('raw', 'raw_riders') }}
