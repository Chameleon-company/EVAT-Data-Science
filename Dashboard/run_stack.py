import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from app_config import (
    API_HOST,
    API_PORT,
    DASHBOARD_PORT,
    DASHBOARD_ROOT,
    PRICE_API_HOST,
    PRICE_API_PORT,
    PRICE_API_BASE_URL,
)


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> int:
    api_base_url = f"http://{API_HOST}:{API_PORT}"
    env = os.environ.copy()
    env.setdefault("EVAT_API_BASE_URL", api_base_url)
    env.setdefault("PRICE_API_BASE_URL", PRICE_API_BASE_URL)

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "apis.congestion_prediction.app:app",
        "--host",
        API_HOST,
        "--port",
        str(API_PORT),
        "--reload",
    ]

    price_api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "apis.price_prediction.app:app",
        "--host",
        PRICE_API_HOST,
        "--port",
        str(PRICE_API_PORT),
        "--reload",
    ]

    dashboard_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path("dashboard.py")),
        "--server.port",
        str(DASHBOARD_PORT),
    ]

    print(f"Starting Congestion API on {api_base_url}")
    api_process = subprocess.Popen(api_cmd, cwd=DASHBOARD_ROOT, env=env)

    print(f"Starting Price API on {PRICE_API_BASE_URL}")
    price_api_process = subprocess.Popen(price_api_cmd, cwd=DASHBOARD_ROOT, env=env)

    print(f"Starting Streamlit Dashboard on http://127.0.0.1:{DASHBOARD_PORT}")
    dashboard_process = subprocess.Popen(dashboard_cmd, cwd=DASHBOARD_ROOT, env=env)

    def shutdown_handler(signum, frame):  # type: ignore[unused-argument]
        _terminate_process(dashboard_process)
        _terminate_process(api_process)
        _terminate_process(price_api_process)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        while True:
            if api_process.poll() is not None:
                print("Congestion API exited. Stopping dashboard.")
                _terminate_process(dashboard_process)
                _terminate_process(price_api_process)
                return api_process.returncode or 0

            if price_api_process.poll() is not None:
                print("Price API exited. Stopping dashboard.")
                _terminate_process(dashboard_process)
                _terminate_process(api_process)
                return price_api_process.returncode or 0

            if dashboard_process.poll() is not None:
                print("Dashboard exited. Stopping Congestion API.")
                _terminate_process(api_process)
                _terminate_process(price_api_process)
                return dashboard_process.returncode or 0

            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        _terminate_process(dashboard_process)
        _terminate_process(api_process)
        _terminate_process(price_api_process)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
