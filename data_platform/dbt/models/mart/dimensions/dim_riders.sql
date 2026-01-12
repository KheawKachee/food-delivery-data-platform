{{ config(materialized='table') }}

select
    rider_id,
    zone as rider_zone,
    signup_date
from {{ ref('staging_riders') }}
