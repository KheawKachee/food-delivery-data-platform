{{ config(materialized='table', unique_key='rider_id') }}

select
    (payload::jsonb->>'rider_id')::bigint as rider_id,
    (payload::jsonb->>'signup_date')::timestamp as signup_date,
    (payload::jsonb->>'zone')::varchar as zone,
    ingest_ts::timestamp                     as updated_at
from {{ source('raw', 'raw_riders') }}
