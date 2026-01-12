{{ config(materialized='table', unique_key='user_id') }}

select
    (payload->>'user_id')::bigint      as user_id,
    (payload->>'signup_date')::timestamp as signup_date,
    (payload->>'zone')::varchar        as zone
from {{ source('raw', 'raw_users') }}
