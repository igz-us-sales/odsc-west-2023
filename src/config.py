from pydantic import BaseModel, BaseSettings, Field, PyObject


class AppConfig(BaseSettings):
    # Embeddings
    embeddings_model: str = "all-MiniLM-L6-v2"
    chunk_size: int  = 250
    chunk_overlap: int = 0
    embeddings_class: PyObject = "langchain.embeddings.HuggingFaceEmbeddings"
    
    # Vector store
    vector_store_class: PyObject = "langchain.vectorstores.Milvus"
    vector_store_connection_args: dict = {
        "host": "my-release-milvus.default.svc.cluster.local",
        "port": "19530"
    }


def get_embedding_function(config: AppConfig):
     return config.embeddings_class(model_name=config.embeddings_model)
    
def get_vector_store(config: AppConfig):
    return config.vector_store_class(
        embedding_function=get_embedding_function(config=config),
        connection_args=config.vector_store_connection_args
    )
