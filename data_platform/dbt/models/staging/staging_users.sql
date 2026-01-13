{{ config(materialized='table', unique_key='user_id') }}

select
    (payload::jsonb->>'user_id')::bigint      as user_id,
    (payload::jsonb->>'signup_date')::timestamp as signup_date,
    (payload::jsonb->>'zone')::varchar        as zone
from {{ source('raw', 'raw_users') }}
