{{ config(materialized='table') }}

select
    user_id,
    zone as user_zone,
    signup_date
from {{ ref('staging_users') }}
