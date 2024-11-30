# SCV

블록 코딩 인터페이스를 이용한 딥러닝 개발 플랫폼 프로젝트의 후기입니다.

- 기간 : 2024. 10. 14. ~ 2024. 11. 19
- 팀 구성
  - BE (모델 관련 CRUD)
  - BE (Github oauth, Github import / export)
  - AI (블록-모델 변환, 유효성 검증 (Back-End), 데이터 전처리, )
  - FE
  - Infra (CPU 서버 구성, CI/CD)
  - **Me!**
- 역할
  - AI
    - 모델 테스트 API 작성
    - 모델 분석
      - 유사 모델 검색
      - confusion matrix
        - 가장 confidence 가 높은 이미지 추출
      - 시각화
        - feature activation
        - activation maximization
  - FE
    - 블록 유효성 검증 (Front-End)
  - Infra
    - GPU 서버 구성
    - GPU 서버 Deployments, Service 작성
    - FastAPI Docker 이미지 생성

## 기능 구현

### 1. 유사 모델 검색

크게 두 개의 고민 포인트가 있었다.

- 모델 간의 유사도를 어떻게 정의할 것인가?
- 전체 모델을 순회 하는 데에 너무 오래 걸리지 않았으면 좋겠다. $O(N)$

모델의 파라미터 개수나 레이어 구성과 같은 메타 데이터를 이용해서 클러스터링 기반의 유사도 검색을 시도할 수도 있었지만, 해당 Feature들로 유사한 모델을 검색한다는게 그리 설득력 있어 보이지 않았다.

그러다 CKA(Centered Kernel Alignment) 를 알게 되었고, 해당 metric이 데이터셋과 함께 레이어 내부에 학습된 파라미터 까지 고려할 수 있는 metric이라는 점이 매력적이어서 선택했다. (정확히는 공통된 데이터에 대해서 해당 레이어의 출력을 사용한다.)

구체적인 구현사항은 다음과 같다.

1. Milvus Vector db 에 Convolution layer의 CKA Vector를 저장한다.
2. Milvus Vector db 로 내적(Inner Product) 유사도를 측정해 가장 높은 유사도의 레이어를 반환하는 API를 작성한다.

다음은 Milvus Vector db의 Schema이다. CKA Vector를 만들기 위해 dataset은 100개를 사용한다. 즉, 100 \* `# Feature` 의 행렬을 $A$ 라 했을 때, $AA^T$가 Milvus에 저장된다. 이 행렬은 100 \* 100 크기이고, 이를 vector로 일렬로 쭉 펴서 (Flatten) 저장하기 때문에 10000 길이의 vector 가 된다.

```python
from pymilvus import connections, db, FieldSchema, CollectionSchema, Collection, DataType, MilvusClient
from dotenv import load_dotenv
import os


load_dotenv(verbose=True)
db_name = os.getenv("DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")
# milvus_host_name = os.getenv("MILVUS_HOST_NAME")
# milvus_port = os.getenv("MILVUS_PORT")

# conn = connections.connect(host=milvus_host_name, port=milvus_port)

# if not db_name in db.list_database():
#     database = db.create_database(db_name)

client = MilvusClient(
    uri="/data/scv_milvus.db"
)

# db.using_database(db_name)

id_field = FieldSchema(
    name="model_version_layer_id",
    dtype=DataType.VARCHAR,
    max_length=30,
    is_primary=True,
    description="model, version, layer id 를 concat 해서 사용")

model_version_field = FieldSchema(
    name="model_version_id",
    dtype=DataType.VARCHAR,
    max_length=30,
    description="model, version를 concat함. delete 요청에 사용")

accuracy_field = FieldSchema(
    name="test_accuracy",
    dtype=DataType.FLOAT,
    description="test 정확도")

layer_field = FieldSchema(
    name="layers",
    dtype=DataType.VARCHAR,
    max_length=1024,
    description="모델의 레이어 정보를 담은 JSON 파일"
)

vector_field = FieldSchema(
    name="cka_vec",
    dtype=DataType.FLOAT_VECTOR,
    dim=10000,
    description="cka 행렬 X 의 XX^T 를 취한 후에 Frobenius Norm으로 나눈 값")


schema = CollectionSchema(fields=[id_field, model_version_field, accuracy_field, layer_field, vector_field], description="collection with cka. pk is model_version_layer_id")

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="model_version_layer_id",
    index_type="INVERTED"
)

index_params.add_index(
    field_name="cka_vec",
    index_type="FLAT",
    metric_type="IP",
)

index_params.add_index(
    field_name="model_version_id",
    index_type="INVERTED"
)

client.create_collection(
    collection_name="cka_collection",
    schema=schema,
    index_params=index_params
)

client.release_collection(
    collection_name="cka_collection"
)
```

다음은 실제 FastAPI의 한 Endpoint 이다. 모델의 id, 버전 id, 레이어 id 를 받아야 레이어 하나가 결정되기 때문에 Path Variable로 받고 있다. 성능이 더 좋은 모델, 혹은 같은 데이터셋에만 해당하는 모델 등 SQL 조건을 걸어줄 수 있다. 실제로 마지막 `filter="model_version_id != '{}'".format(model_version_id)` 부분을 제외할 경우 항상 같은 레이어가 결과로 출력되는 것을 확인함으로써 본 기능이 잘 작동하고 있음을 검증할 수 있었다. (실제 CKA metric을 검증할 때도 같은 방법을 사용한다.)

```python
# 유사 모델 검색
@app.get("/{model_id}/{version_id}/{layer_id}/search", response_model=Model_Search_Response)
async def search_model(model_id: str, version_id: str, layer_id: str):
    model_version_layer_id = "{}_{}".format(f"model_{model_id}_v{version_id}", layer_id)
    model_version_id = f"model_{model_id}_v{version_id}"
    cached = redis.get(model_version_layer_id)
    if cached:
        print("응답이 캐싱되었습니다.")
        cached = json.loads(cached)
        cached["layers"] = deserialize_layers(cached["layers"])
        return cached

    model = client.get(
        collection_name=collection_name,
        ids=[model_version_layer_id]
    )

    if (len(model) == 0):
        raise InvalidModelId(model_version_layer_id)

    model = model[0]

    print("{} id로 가장 유사한 레이어를 검색합니다.".format(model_version_layer_id))

    results = client.search(
        collection_name=collection_name,
        data=[model["cka_vec"]],
        anns_field="cka_vec",
        output_fields=["model_version_layer_id", "test_accuracy", "cka_vec", "layers"],
        search_params={"metric_type": "IP"},
        limit=1,
        filter="model_version_id != '{}'".format(model_version_id)
        # 성능이 더 좋은 모델만 찾아주려면, 아무 것도 찾지 못했을 수 있음
        # filter="test_accuracy > {}".format(model[0]["test_accuracy"])
    )

    if (len(results[0]) == 0):
        raise LayerNotFound()

    results = dict(results[0][0])

    id_parse = results["id"].split("_")
    searched_model_version_id = id_parse[0] + id_parse[1] + id_parse[2]
    searched_layer_id = id_parse[1]
    searched_test_accuracy = results["entity"]["test_accuracy"]
    print(f"searched_model_version_id: {searched_model_version_id}")
    target = model["layers"]
    target_test_accuracy = model["test_accuracy"]
    searched = results["entity"]["layers"]

    gpt_description = await get_gpt_answer(target, searched, layer_id, searched_layer_id, target_test_accuracy,
                                           searched_test_accuracy)

    searched = deserialize_layers(searched)

    resp = {"model_version_id": searched_model_version_id, "layer_id": searched_layer_id,
            "gpt_description": gpt_description, "test_accuracy": searched_test_accuracy,
            "layers": serialize_layers(searched)}

    redis.set(model_version_layer_id, json.dumps(resp))
    redis.expire(model_version_layer_id, 3600)

    resp["layers"] = deserialize_layers(resp["layers"])
    return resp
```

다음은 유사 모델을 찾고 나서 두 모델의 구조를 비교하고 그를 바탕으로 성능 개선을 위한 조언을 GPT에게 물어보는 부분이다. 사실 이 부분은 기능 구현에 초점을 맞추었을 뿐 실제 도움이 되는 지에는 그리 집중하지 않았다. 시간이 된다면 프롬프트도 더 갈고 닦고, 실제 도움이 되는지 검증해보고 싶은 마음은 있다.

```python
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
api_key = os.getenv("API_KEY")

gpt_client = AsyncOpenAI(api_key = api_key)
gpt_model = "gpt-4o-mini"

prompt = """
-- Compare <target> model and <searched> model
-- first, compare them in their "Structures"
-- second, compare them in their "Performance", guess the reason why their performances were different.
-- third, with the point that <target> model's <target_layer>th layer(starts from 0) is turned out to be similar with <searched> model's <searched_layer>th layer(starts from 0) by the CKA(centered kernel alignment),
    tell me anything helpful to develop <target> model. guess the reason why two layers were similar.
-- There are no more information to give, so don't ask for further information
-- translate all to Korean.
"""

client = AsyncOpenAI(api_key = api_key)

async def get_gpt_answer(target_layer : str, searched_layer : str, target_layer_id, searched_layer_id, target_test_accuracy, searched_test_accuracy):

    chat_completion = await client.chat.completions.create(
        model= gpt_model,
        messages=[{"role": "user",
        "content": prompt + "<target> {} <target test accuracy> {} \n <searched> {} <searched model test accuracy> {}".format(target_layer, target_test_accuracy, searched_layer, searched_test_accuracy) }]
    )

    return chat_completion.choices[0].message.content
```

다음은 구체적으로 CKA vector가 Milvus에 저장되는 API Endpoint에서 사용하는 함수이다. MinIO에 데이터 셋이 `train`, `test`, `cka` 용도로 분류되어 저장되고 있다. 본 함수에서는 `cka` 데이터셋을 불러오고 (100개) 이를 레이어에 통과시키면서 해당 레이어가 Convolution일 때만 그 출력을 저장한다.

```python
async def save_cka_to_milvus(model, dataset, model_version_id, conv_idx, test_accuracy, layers, device):
    cka_dataset = load_dataset_from_minio(dataset, "cka")
    id_parse = model_version_id.split("_")
    model_id = id_parse[1]
    version_id = id_parse[2][1:]

# Milvus CKA 저장
    cka_matrix = defaultdict(list)
    with torch.no_grad():
        for index, data in enumerate(cka_dataset):
            input = data[0]
            label = data[1]
            input = input.to(device)
            label = label.to(device)
            x = input

            for i in range(0, len(model)):
                x = model[i](x)
                if i in conv_idx:
                    cka_matrix[i].append(torch.flatten(x).cpu().numpy())


    for i in cka_matrix.keys():
        mat = np.array(cka_matrix[i])
        n = mat.shape[0]
        # centering matrix
        H = np.eye(n) - np.ones((n, n)) / n
        cka = (H @ mat @ mat.T @ H).flatten()
        cka_vec = cka / np.linalg.norm(cka)
        async with httpx.AsyncClient() as client:
            res = await client.post(f"http://{fast_match_host_name}:{fast_match_port}/fast/v1/model/match/{model_id}/{version_id}/{i}",
                        json={
                            "test_accuracy": test_accuracy,
                            "layers" : serialize_layers(layers),
                            "cka_vec": cka_vec.tolist()
                        })
            print(res)
```

여기서 다음의 부분이 CKA vector를 만드는 부분에 해당한다.

```python
mat = np.array(cka_matrix[i])
n = mat.shape[0]
# centering matrix
H = np.eye(n) - np.ones((n, n)) / n
cka = (H @ mat @ mat.T @ H).flatten()
cka_vec = cka / np.linalg.norm(cka)
```
