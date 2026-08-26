from pymilvus import MilvusClient, DataType

client = MilvusClient(
    uri="https://in01-8fb19767351ff95.aws-ap-southeast-1.vectordb.zillizcloud.com:19532",
    token="<REPLACE_WITH_ZILLIZ_TOKEN>"
)

col_name = "hermes_gemini_memory"
if client.has_collection(collection_name=col_name):
    client.drop_collection(collection_name=col_name)

schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)

# 1. Primary key
schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=200, is_primary=True)
# 2. Vector field for Gemini embeddings (3072 dimensions)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=3072)
# 3. Payload
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=15000)
schema.add_field(field_name="session_id", datatype=DataType.VARCHAR, max_length=100)
schema.add_field(field_name="timestamp", datatype=DataType.INT64)

# Create index
index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

print(f"Creating collection {col_name} for Gemini 3072D...")
client.create_collection(collection_name=col_name, schema=schema, index_params=index_params)
print("SUCCESS!")
