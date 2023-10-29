Install milvus
```bash
helm repo add milvus https://zilliztech.github.io/milvus-helm/

helm repo update

helm upgrade --install my-release \
    --set cluster.enabled=false \
    --set etcd.replicaCount=1 \
    --set pulsar.enabled=false \
    --set minio.mode=standalone \
    --set standalone.persistence.enabled=false \
    --set minio.persistence.enabled=false \
    --set etcd.persistence.enabled=false \
    milvus/milvus
```