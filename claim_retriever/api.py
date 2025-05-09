import requests
import time
from .config import FACT_CHECK_TOOLS_URL


def query_api(params, retry_delay=1, max_retries=5):
    for _ in range(max_retries):
        try:
            response = requests.get(FACT_CHECK_TOOLS_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if "error" in data and data["error"].get("code") == 503:
                print(f"Service unavailable. Retrying in {retry_delay} sec...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue

            return data.get("claims", [])

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return []

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")
            return []

        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    print("Max retries reached. Exiting...")
    return []
