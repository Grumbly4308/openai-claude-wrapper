"""Fine-tuning endpoints — 501 Not Implemented.

Claude models aren't fine-tunable through this path. We expose the
endpoints so clients can probe them without blowing up, but every write
returns a clear error.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from .deps import auth_dependency


router = APIRouter()


_UNSUPPORTED = HTTPException(
    status_code=501,
    detail={
        "error": {
            "message": "Fine-tuning is not supported: Claude models are not user-fine-tunable through this API. Use the Assistants API (/v1/assistants) for per-customer system prompts and tools.",
            "type": "not_implemented",
        }
    },
)


@router.post("/v1/fine_tuning/jobs", dependencies=[Depends(auth_dependency)])
async def create_fine_tune_job():
    raise _UNSUPPORTED


@router.get("/v1/fine_tuning/jobs", dependencies=[Depends(auth_dependency)])
async def list_fine_tune_jobs(limit: int = 20):
    return JSONResponse(content={"object": "list", "data": [], "has_more": False})


@router.get("/v1/fine_tuning/jobs/{job_id}", dependencies=[Depends(auth_dependency)])
async def get_fine_tune_job(job_id: str):
    raise HTTPException(status_code=404, detail="fine-tune job not found")


@router.post("/v1/fine_tuning/jobs/{job_id}/cancel", dependencies=[Depends(auth_dependency)])
async def cancel_fine_tune_job(job_id: str):
    raise _UNSUPPORTED


@router.get("/v1/fine_tuning/jobs/{job_id}/events", dependencies=[Depends(auth_dependency)])
async def list_fine_tune_events(job_id: str):
    return JSONResponse(content={"object": "list", "data": [], "has_more": False})
