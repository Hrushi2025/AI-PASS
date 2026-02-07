from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.services.rag import RAGService
from app.services.agent import AgentEngine
from app.core.schemas import TaskInput, DecisionOutput
from app.services.metrics import compute_metrics


from pathlib import Path



app = FastAPI(title="AI-Pass Core MVP")

rag = RAGService()
agent = AgentEngine()

class RetrieveRequest(BaseModel):
    query: str
    doc_ids: list[str] = []
    top_k: int = 5

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), doc_id: str = "doc_1"):
    data = await file.read()
    return rag.ingest_pdf(data, doc_id=doc_id)

@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    return {"results": rag.retrieve(req.query, req.doc_ids, top_k=req.top_k)}

@app.post("/agent/run", response_model=DecisionOutput)
def run_agent(task: TaskInput):
    return agent.execute(task)

@app.get("/")
def root():
    return {"status": "AI-Pass running", "docs": "/docs"}

@app.get("/debug/paths")
def debug_paths():
    project_root = Path(__file__).resolve().parents[1]
    audit_dir = project_root / "data" / "audit_logs"
    return {
        "main_py": str(Path(__file__).resolve()),
        "project_root": str(project_root),
        "audit_dir": str(audit_dir),
        "audit_dir_exists": audit_dir.exists(),
        "audit_files": [p.name for p in audit_dir.glob("*.json")] if audit_dir.exists() else []
    }

@app.get("/metrics")
def metrics():
    return compute_metrics()
