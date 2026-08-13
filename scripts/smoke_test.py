import sys
import time

import requests

BASE_URL = "http://localhost:8000"


def wait_for_health(timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def check_predict(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        r = requests.post(f"{BASE_URL}/predict", files=files, timeout=10)
    r.raise_for_status()
    body = r.json()
    assert body["label"] == "cat", f"expected cat, got: {body}"
    assert body["probabilities"]["cat"] > 0.6, f"low-confidence/random-looking prediction: {body}"
    return body


def main() -> None:
    if not wait_for_health():
        print("Smoke test FAILED: /health never returned ok")
        sys.exit(1)

    image_path = sys.argv[1] if len(sys.argv) > 1 else "scripts/sample_pet.jpg"
    try:
        result = check_predict(image_path)
    except Exception as exc:
        print(f"Smoke test FAILED: {exc}")
        sys.exit(1)

    print(f"Smoke test passed: {result}")
    sys.exit(0)


if __name__ == "__main__":
    main()
