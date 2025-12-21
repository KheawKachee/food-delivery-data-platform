airflow db migrate

airflow users create \
    --username admin \
    --firstname admin \
    --lastname admin \
    --role Admin \
    --email spiderman@superhero.org

airflow api-server --port 8080

airflow scheduler

airflow dag-processor

airflow triggerer