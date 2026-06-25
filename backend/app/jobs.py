"""인메모리 작업(Job) 관리자.

학습은 수십 초~수 분 걸리므로 백그라운드 스레드에서 실행하고, 프론트는
job_id 로 진행상황(status/progress)을 폴링한다. 프로토타입 범위라 DB 없이
프로세스 메모리에만 보관한다(서버 재시작 시 초기화).
"""
import uuid
import threading
import traceback
from typing import Dict

from . import engine, engine_hanium

_jobs: Dict[str, Dict] = {}
_lock = threading.Lock()


def _set(job_id: str, **kw):
    with _lock:
        _jobs[job_id].update(kw)


def _run(job_id: str, params: Dict):
    def cb(phase, step, total, pv, pl):
        progress = 0.0
        if phase == "training" and total:
            progress = step / total
        elif phase == "backtesting":
            progress = 0.99
        _set(job_id, phase=phase, step=step, total=total,
             progress=round(progress, 4), last_pv=pv, last_profitloss=pl)

    try:
        _set(job_id, status="running", phase="starting", progress=0.0)
        # 엔진 선택: 'hanium'(신규 다중 알고리즘) / 'quantylab'(기존, 기본값)
        eng = engine_hanium if params.get("engine") == "hanium" else engine
        result = eng.run_experiment(params, progress_callback=cb)
        _set(job_id, status="done", phase="done", progress=1.0, result=result)
    except Exception as e:  # noqa: BLE001
        _set(job_id, status="error", phase="error",
             error=str(e), traceback=traceback.format_exc())


def start_job(params: Dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"id": job_id, "status": "queued", "phase": "queued",
                         "progress": 0.0, "params": params}
    t = threading.Thread(target=_run, args=(job_id, params), daemon=True)
    t.start()
    return job_id


def get_job(job_id: str) -> Dict | None:
    with _lock:
        return _jobs.get(job_id)
