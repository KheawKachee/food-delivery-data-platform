DROP SCHEMA public CASCADE;
CREATE SCHEMA public;

CREATE TABLE IF NOT EXISTS users (
    user_id INT PRIMARY KEY,
    signup_date TIMESTAMP NOT NULL,
    zone VARCHAR(5) NOT NULL
);

CREATE TABLE IF NOT EXISTS riders (
    rider_id INT PRIMARY KEY,
    signup_date TIMESTAMP NOT NULL,
    zone VARCHAR(5) NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INT PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(user_id),
    rider_id INT NOT NULL REFERENCES riders(rider_id),
    order_time TIMESTAMP NOT NULL,
    prep_time_minutes NUMERIC(6,2),
    distance_km NUMERIC(6,2),
    delivery_time_minutes NUMERIC(6,2),
    price_baht NUMERIC(10,2),
    rider_rating NUMERIC(3,1) CHECK (rider_rating BETWEEN 1 AND 5)
);



CREATE TABLE IF NOT EXISTS staging_users (
    user_id INT  ,
    signup_date TIMESTAMP NOT NULL,
    zone VARCHAR(5) NOT NULL
);

CREATE TABLE IF NOT EXISTS staging_riders (
    rider_id INT  ,
    signup_date TIMESTAMP NOT NULL,
    zone VARCHAR(5) NOT NULL
);

CREATE TABLE IF NOT EXISTS staging_orders (
    order_id INT  ,
    user_id INT NOT NULL  ,
    rider_id INT NOT NULL  ,
    order_time TIMESTAMP NOT NULL,
    prep_time_minutes NUMERIC(6,2),
    distance_km NUMERIC(6,2),
    delivery_time_minutes NUMERIC(6,2),
    price_baht NUMERIC(10,2),
    rider_rating NUMERIC(3,1) CHECK (rider_rating BETWEEN 1 AND 5)
);