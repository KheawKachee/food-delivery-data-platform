---drop and create---
drop schema if exists public cascade;
create schema public;


-- RAW LAYER (no business logic, no FKs)
CREATE TABLE raw_orders (
    order_id bigint PRIMARY KEY,
    payload JSONB NOT NULL,
    ingest_ts TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE raw_users (
    user_id bigint PRIMARY KEY,
    payload JSONB NOT NULL,
    ingest_ts TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE raw_riders (
    rider_id bigint PRIMARY KEY,
    payload JSONB NOT NULL,
    ingest_ts TIMESTAMP NOT NULL DEFAULT now()
);


---staging table---
CREATE TABLE staging_users (
    user_id bigint PRIMARY KEY,
    signup_date TIMESTAMP NOT NULL,
    zone VARCHAR(5) NOT NULL
);

CREATE TABLE staging_riders (
    rider_id bigint PRIMARY KEY,
    signup_date TIMESTAMP NOT NULL,
    zone VARCHAR(5) NOT NULL
);

CREATE TABLE staging_orders (
    order_id bigint PRIMARY KEY,
    user_id bigint NOT NULL,
    rider_id bigint NOT NULL,
    order_ts TIMESTAMP NOT NULL,
    food_ready_ts TIMESTAMP NOT NULL,
    delivered_ts TIMESTAMP NOT NULL,
    distance_km NUMERIC(6,2) CHECK (distance_km > 0),
    price_baht NUMERIC(10,2) CHECK (price_baht >= 0),
    rider_rating NUMERIC(3,1)
);


---mart table---
---delivery_time : hypothesis as avg_rider_rating have relation with these params--- 
CREATE TABLE delivery_time (
    order_id bigint PRIMARY KEY,
    order_ts TIMESTAMP NOT NULL,
    delivery_time INTERVAL NOT NULL,
    distance_km NUMERIC(6,2) CHECK (distance_km > 0),
    user_zone VARCHAR(5) NOT NULL,
    rider_zone VARCHAR(5) NOT NULL,
    avg_rider_rating NUMERIC(3,2) CHECK (avg_rider_rating BETWEEN 0 AND 5)
);

CREATE TABLE avg_rider_rating (
    rider_id bigint PRIMARY KEY,
    rider_zone VARCHAR(5) NOT NULL,
    avg_rider_rating NUMERIC(3,2) CHECK (avg_rider_rating BETWEEN 0 AND 5),
    n_jobs INT NOT NULL CHECK (n_jobs >= 0)
);


CREATE TABLE hourly_total_spends (
    hourly TIMESTAMP PRIMARY KEY,
    n_orders INT NOT NULL CHECK (n_orders >= 0),
    total_price_baht NUMERIC(12,2) CHECK (total_price_baht >= 0)
);



