---raw table---
drop schema if exists public cascade;
create schema public;
CREATE TABLE raw_orders (
    order_id BIGSERIAL PRIMARY KEY,
    payload JSONB NOT NULL,
    ingest_ts TIMESTAMP DEFAULT now()
);


---staging table---
CREATE TABLE stg_users (
    user_id INT PRIMARY KEY,
    signup_date TIMESTAMP NOT NULL,
    zone VARCHAR(5) NOT NULL
);

CREATE TABLE stg_riders (
    rider_id INT PRIMARY KEY,
    signup_date TIMESTAMP NOT NULL,
    zone VARCHAR(5) NOT NULL
);

CREATE TABLE stg_orders (
    order_id INT PRIMARY KEY,
    user_id INT NOT NULL,
    rider_id INT NOT NULL,
    order_ts TIMESTAMP NOT NULL,
    food_ready_ts TIMESTAMP NOT NULL,
    distance_km NUMERIC(6,2) CHECK (distance_km > 0),
    deliveried_ts TIMESTAMP NOT NULL,
    price_baht NUMERIC(10,2) CHECK (price_baht >= 0),
    rider_rating NUMERIC(3,1),

    -- Adding the Constraints
    FOREIGN KEY (user_id) REFERENCES stg_users(user_id),
    FOREIGN KEY (rider_id) REFERENCES stg_riders(rider_id)
);


---mart table---
---delivery_time : hypothesis as avg_rider_rating have relation with these params--- 
CREATE TABLE delivery_time (
    order_id INT NOT NULL PRIMARY KEY,
    order_ts TIMESTAMP,
    delivery_time INTERVAL NOT NULL,
    distance_km NUMERIC(6,2) CHECK (distance_km > 0),
    user_zone VARCHAR(5) NOT NULL,
    rider_zone VARCHAR(5) NOT NULL,
    avg_rider_rating NUMERIC(3,2) CHECK (avg_rider_rating >= 0)
);

CREATE TABLE avg_rider_rating (
    rider_id INT NOT NULL PRIMARY KEY,
    rider_zone VARCHAR(5) NOT NULL,
    avg_rider_rating NUMERIC(3,2) CHECK (avg_rider_rating >= 0),
    n_jobs INT NOT NULL
);

CREATE TABLE hourly_total_spends (
    hourly TIMESTAMP PRIMARY KEY,
    n_orders INT CHECK (n_orders >= 0),
    total_price_baht NUMERIC(12,2) CHECK (total_price_baht >= 0)
);



