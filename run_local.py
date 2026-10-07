"""Run the local API and Streamlit UI together from one terminal."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parent
API_HEALTH_URL = "http://127.0.0.1:8000/health"


def api_is_ready() -> bool:
    try:
        with urlopen(API_HEALTH_URL, timeout=2) as response:
            return response.status == 200
    except (OSError, URLError, TimeoutError):
        return False


def wait_for_api(process: subprocess.Popen[bytes], timeout_seconds: int = 60) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(f"API đã dừng với mã lỗi {process.returncode}.", flush=True)
            return False
        if api_is_ready():
            return True
        time.sleep(1)
    return False


def start_service(name: str, command: list[str]) -> subprocess.Popen[bytes]:
    process_options: dict[str, object] = {}
    if os.name == "nt":
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        process_options["start_new_session"] = True
    print(f"Đang khởi động {name}...", flush=True)
    return subprocess.Popen(command, cwd=PROJECT_ROOT, **process_options)


def stop_service(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    processes: list[tuple[str, subprocess.Popen[bytes]]] = []
    try:
        if api_is_ready():
            print("API đã chạy sẵn tại http://127.0.0.1:8000.", flush=True)
        else:
            api = start_service(
                "API",
                [sys.executable, "-m", "uvicorn", "app.main:app", "--reload"],
            )
            processes.append(("API", api))
            if not wait_for_api(api):
                print("API không sẵn sàng sau 60 giây; hãy xem lỗi ở phía trên.", flush=True)
                return 1

        ui = start_service(
            "Streamlit",
            [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py"],
        )
        processes.append(("Streamlit", ui))
        print("API và giao diện đang chạy. Nhấn Ctrl+C để dừng.", flush=True)

        while True:
            for name, process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    print(f"{name} đã dừng với mã lỗi {exit_code}.", flush=True)
                    return exit_code or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Đang dừng API và giao diện...", flush=True)
        return 0
    finally:
        for _, process in reversed(processes):
            stop_service(process)


if __name__ == "__main__":
    raise SystemExit(main())
