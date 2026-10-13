import os
import asyncio
import time
import random
import httpx
import yaml

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# configure backend servers
API_KEY = "sk-abc123"
HEALTH_CHECK_INTERVAL = float(os.environ.get("BACKEND_HEALTH_CHECK_INTERVAL", "30"))
REQUEST_TIMEOUT = float(os.environ.get("BACKEND_REQUEST_TIMEOUT", "300"))
RECOVERY_WAIT_SECONDS = float(os.environ.get("BACKEND_RECOVERY_WAIT_SECONDS", "30"))

class Backend:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.healthy = True
        self.active_requests = 0
        self.lock = asyncio.Lock()

LOCAL_IP = os.getenv("LOCAL_IP", "localhost")
hosts_str = os.environ.get('WORKER_HOSTS', "")
worker_hosts = list(dict.fromkeys([*hosts_str.split(), LOCAL_IP]))

# Backend ports this balancer forwards to. Override with the BACKEND_PORTS env
# var (comma-separated) so a 4-instance backend (e.g. ports 8104-8107) can be
# served without the default 8100-8107. Defaults to the original 8100-8107.
backend_ports_env = os.environ.get('BACKEND_PORTS', ','.join(str(8100 + i) for i in range(8)))
BACKEND_PORTS = [int(p) for p in backend_ports_env.split(',') if p.strip()]

BACKENDS_DATA = [
    {"url": f"http://{ip}:{port}", "name": f"instance-{ip}-{port}", "healthy": True}
    for port in BACKEND_PORTS for ip in worker_hosts
]

backends_list = [Backend(b["name"], b["url"]) for b in BACKENDS_DATA]


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = "api_qwen2_5_siglip_3b_instruct"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False


class LoadBalancer:
    def __init__(self, backends: List[Backend]):
        self.backends = backends
    
    async def get_least_loaded_backend(self) -> Optional[Backend]:
        healthy_backends = [b for b in self.backends if b.healthy]
        
        if not healthy_backends:
            return None
        
        min_load = min(b.active_requests for b in healthy_backends)
        candidates = [b for b in healthy_backends if b.active_requests == min_load]
        
        return random.choice(candidates)

    def mark_unhealthy(self, backend: Backend):
        backend.healthy = False
    
    def mark_healthy(self, backend: Backend):
        backend.healthy = True

load_balancer = LoadBalancer(backends_list)


@asynccontextmanager
async def track_requests(backend: Backend):
    async with backend.lock:
        backend.active_requests += 1
    try:
        yield
    finally:
        async with backend.lock:
            backend.active_requests -= 1


async def health_check_task():
    async with httpx.AsyncClient() as client:
        while True:
            for backend in backends_list:
                # The Transformers backend performs synchronous generation in
                # the server process, so /health may not be served while a
                # request is active.  Do not mistake a busy worker for a dead
                # one.
                if backend.active_requests > 0:
                    continue
                try:
                    response = await client.get(f"{backend.url}/health", timeout=5.0)
                    if response.status_code == 200:
                        load_balancer.mark_healthy(backend)
                    else:
                        load_balancer.mark_unhealthy(backend)
                except Exception:
                    load_balancer.mark_unhealthy(backend)
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)


async def recover_backends(wait_seconds: float = RECOVERY_WAIT_SECONDS):
    """Actively recover workers instead of returning a burst of HTTP 503s."""
    deadline = time.monotonic() + wait_seconds
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            for backend in backends_list:
                # An in-flight request proves the process was reachable; its
                # health endpoint can be blocked by synchronous generation.
                if backend.active_requests > 0:
                    load_balancer.mark_healthy(backend)
                    continue
                try:
                    response = await client.get(f"{backend.url}/health", timeout=5.0)
                    if response.status_code == 200:
                        load_balancer.mark_healthy(backend)
                except Exception:
                    pass
            backend = await load_balancer.get_least_loaded_backend()
            if backend is not None:
                return backend
            await asyncio.sleep(1)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    health_check = asyncio.create_task(health_check_task())
    yield
    health_check.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    healthy_count = sum(1 for b in backends_list if b.healthy)
    return {
        "status": "healthy" if healthy_count > 0 else "unhealthy",
        "healthy_backends": healthy_count,
        "total_backends": len(BACKENDS_DATA)
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "api_qwen2_5_siglip_3b_instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization"
            }
        ]
    }


async def proxy_request(backend: Backend, path: str, request_data: Dict, stream: bool = False):
    url = f"{backend.url}{path}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    
    async with httpx.AsyncClient() as client:
        if stream:
            async with client.stream("POST", url, json=request_data, headers=headers, timeout=REQUEST_TIMEOUT) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
        else:
            response = await client.post(url, json=request_data, headers=headers, timeout=REQUEST_TIMEOUT)
            yield response

@app.get("/backends/status")
async def backends_status():
    return {
        "backends": [
            {
                "name": b.name,
                "url": b.url,
                "healthy": b.healthy,
                "active_requests": b.active_requests
            }
            for b in backends_list
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    backend = await load_balancer.get_least_loaded_backend()
    if not backend:
        backend = await recover_backends()
    if not backend:
        raise HTTPException(status_code=503, detail="No healthy backend available")
    
    request_data = request.model_dump(exclude_none=True)
    
    try:
        if request.stream:
            async def generate_with_tracking():
                async with track_requests(backend):
                    try:
                        async for chunk in proxy_request(backend, "/v1/chat/completions", request_data, stream=True):
                            yield chunk
                    except Exception:
                        load_balancer.mark_unhealthy(backend)
                        raise

            return StreamingResponse(generate_with_tracking(), media_type="text/event-stream")
        else:
            async with track_requests(backend):
                async for response in proxy_request(backend, "/v1/chat/completions", request_data, stream=False):
                    if response.status_code != 200:
                        # A request-specific 4xx/5xx does not mean the process
                        # or its /health endpoint is unavailable.
                        return JSONResponse(
                            content=response.json(), status_code=response.status_code)
                    return JSONResponse(content=response.json(), status_code=response.status_code)

    except httpx.TimeoutException:
        # A synchronous Transformers worker can remain alive and continue
        # computing after the proxy-side deadline.  Treating a slow long-
        # context request as a dead process empties a one-worker pool and
        # causes a 504 -> repeated-503 cascade.  Connectivity/health checks,
        # rather than request duration, determine worker health.
        raise HTTPException(status_code=504, detail="Backend timeout")

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Proxy error: {str(e)}")


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    print(f"Balancing {len(BACKENDS_DATA)} backend(s) on ports {BACKEND_PORTS} -> listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
