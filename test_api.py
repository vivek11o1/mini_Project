"""Quick smoke test: python test_api.py"""

import warnings

warnings.filterwarnings("ignore", message=".*httpx.*deprecated.*")
warnings.filterwarnings("ignore", message=".*httpx2.*")

from fastapi.testclient import TestClient  # noqa: E402

from app import app

client = TestClient(app)

health = client.get("/health")
assert health.status_code == 200, health.text
assert health.json()["status"] == "ok"
print("health:", health.json())

topics = client.get("/topics")
assert topics.status_code == 200 and len(topics.json()["topics"]) >= 2
print("topics:", topics.json())

prediction = client.post("/predict", json={"text": "The government passes a new law in parliament"})
assert prediction.status_code == 200, prediction.text
assert prediction.json()["topic"] in topics.json()["topics"]
assert 0.0 <= prediction.json()["confidence"] <= 1.0
print("predict:", prediction.json())

batch = client.post(
    "/predict/batch",
    json={"texts": ["team wins the final match", "new smartphone launched today"]},
)
assert batch.status_code == 200, batch.text
assert len(batch.json()["predictions"]) == 2
print("batch:", batch.json())

assert client.get("/docs").status_code == 200
print("Swagger UI: /docs OK")
print("ALL TESTS PASSED")
