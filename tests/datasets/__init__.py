from pathlib import Path

CACHE_DIR = Path(".pytest_cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
TEST_CACHE_DIR = str(CACHE_DIR)
