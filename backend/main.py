import os
import io
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import Job, Candidate, Score

from pypdf import PdfReader
from docx import Document as DocxDocument
import requests

app = FastAPI(title="AI Resume Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "AI Resume Agent API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = getattr(db, "name", "?")
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# --------------------------
# File parsing
# --------------------------

def parse_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        out = []
        for page in reader.pages:
            text = page.extract_text() or ""
            out.append(text)
        return "\n".join(out).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parse failed: {e}")


def parse_docx(file_bytes: bytes) -> str:
    try:
        bio = io.BytesIO(file_bytes)
        doc = DocxDocument(bio)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DOCX parse failed: {e}")


def parse_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Text parse failed: {e}")


@app.post("/api/parse")
async def parse_resume(file: UploadFile = File(...)):
    ext = (file.filename or "").lower()
    data = await file.read()
    if ext.endswith(".pdf"):
        text = parse_pdf(data)
    elif ext.endswith(".docx"):
        text = parse_docx(data)
    elif ext.endswith(".txt") or ext.endswith(".md"):
        text = parse_text(data)
    else:
        # Fallback: try text decode
        text = parse_text(data)
    return {"filename": file.filename, "text": text}


# --------------------------
# Scoring via heuristic + optional OpenRouter LLM
# --------------------------

class ScoreRequest(BaseModel):
    job_description: str
    candidates: List[Dict[str, Any]]  # each {id, name, filename?, text}
    model: Optional[str] = None  # openrouter model id


def _tokenize(text: str) -> List[str]:
    import re
    return [
        t
        for t in re.sub(r"[^a-z0-9+#. ]", " ", (text or "").lower()).split()
        if len(t) > 2
    ]


def _unique_keywords(text: str) -> List[str]:
    common = {
        "and",
        "the",
        "with",
        "for",
        "you",
        "are",
        "our",
        "this",
        "that",
        "from",
        "your",
        "will",
        "have",
    }
    return list({t for t in _tokenize(text) if t not in common})


def _extract_years(text: str) -> int:
    import re
    years = [int(m.group(1)) for m in re.finditer(r"(\d{1,2})\s+years?", text or "")]
    if years:
        return max(years)
    if re.search(r"senior|lead|principal|staff", text or "", re.I):
        return 7
    if re.search(r"mid|intermediate", text or "", re.I):
        return 4
    if re.search(r"junior|entry", text or "", re.I):
        return 1
    return 0


def _infer_seniority(text: str) -> str:
    import re
    if re.search(r"principal|staff|lead|manager|architect", text or "", re.I):
        return "Senior"
    if re.search(r"senior", text or "", re.I):
        return "Senior"
    if re.search(r"mid|intermediate", text or "", re.I):
        return "Mid"
    if re.search(r"junior|entry", text or "", re.I):
        return "Junior"
    return "Unknown"


def _heuristic_score(jd: str, resume_text: str) -> Dict[str, Any]:
    jd_keywords = _unique_keywords(jd)
    rset = set(_tokenize(resume_text))
    matched = [k for k in jd_keywords if k in rset]
    skill = int(round((len(matched) / len(jd_keywords) * 100), 0)) if jd_keywords else 0
    years = _extract_years(resume_text)
    seniority = _infer_seniority(resume_text)
    seniority_score = 100 if seniority == "Senior" else 60 if seniority == "Mid" else 30 if seniority == "Junior" else 40
    exp_score = min(100, int(round(years / 12 * 100)))
    overall = int(round(skill * 0.7 + exp_score * 0.2 + seniority_score * 0.1))
    return {
        "skill_match": skill,
        "years": years,
        "seniority": seniority,
        "overall": overall,
        "top_skills": sorted(matched)[:10],
    }


def _llm_score(jd: str, candidates: List[Dict[str, Any]], model: Optional[str] = None):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None  # not configured
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://example.com"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "AI Resume Agent"),
    }
    model_id = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    system = (
        "You are an expert technical recruiter. Given a job description and candidate resume text, "
        "return a strict JSON array with items: {id, skill_match (0-100), years (int), seniority (Junior|Mid|Senior), overall (0-100), top_skills (array)}."
    )
    user_content = {
        "job_description": jd,
        "candidates": [
            {"id": c.get("id"), "name": c.get("name"), "text": c.get("text")[:4000] if c.get("text") else ""}
            for c in candidates
        ],
    }
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_content)},
        ],
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        # Expect either {results: [...]} or [...] directly
        results = parsed.get("results") if isinstance(parsed, dict) else parsed
        if not isinstance(results, list):
            return None
        by_id = {str(item.get("id")): item for item in results}
        return by_id, {"model": model_id}
    except Exception:
        return None


@app.post("/api/score")
def score_candidates(payload: ScoreRequest):
    jd = payload.job_description
    input_candidates = payload.candidates

    # Prepare for LLM
    llm_ready = [{"id": c.get("id"), "name": c.get("name"), "text": c.get("text") or c.get("raw_text") or ""} for c in input_candidates]

    llm_result = _llm_score(jd, llm_ready, model=payload.model)

    results = []
    for c in input_candidates:
        base = _heuristic_score(jd, (c.get("text") or c.get("raw_text") or ""))
        if llm_result:
            by_id, meta = llm_result
            override = by_id.get(str(c.get("id")))
            if override:
                base.update({
                    "skill_match": int(override.get("skill_match", base["skill_match"])) ,
                    "years": int(override.get("years", base["years"])) ,
                    "seniority": str(override.get("seniority", base["seniority"])) ,
                    "overall": int(override.get("overall", base["overall"])) ,
                    "top_skills": list(override.get("top_skills", base["top_skills"])) ,
                    "llm_model": meta.get("model"),
                })
        results.append({
            "id": c.get("id"),
            "name": c.get("name"),
            "filename": c.get("filename"),
            "scores": base,
        })

    # sort by overall desc
    results.sort(key=lambda x: x["scores"].get("overall", 0), reverse=True)
    return {"results": results}


# --------------------------
# Persistence endpoints
# --------------------------

class SaveRequest(BaseModel):
    job: Job
    candidates: List[Candidate]
    scores: List[Dict[str, Any]]  # tie candidates to scores via order or id


@app.post("/api/save")
def save_results(payload: SaveRequest):
    # Save job
    job_id = create_document("job", payload.job)

    id_map: Dict[str, str] = {}
    for c in payload.candidates:
        cid = create_document("candidate", c)
        id_map[c.name] = cid  # fallback mapping by name if no id provided

    saved_scores = []
    for s in payload.scores:
        # try to resolve candidate id
        cand_name = s.get("name")
        candidate_id = s.get("candidate_id") or (id_map.get(cand_name) if cand_name else None)
        if not candidate_id:
            # skip if cannot resolve
            continue
        score_doc = Score(
            job_id=job_id,
            candidate_id=candidate_id,
            skill_match=int(s.get("scores", {}).get("skill_match", 0)),
            years=int(s.get("scores", {}).get("years", 0)),
            seniority=str(s.get("scores", {}).get("seniority", "Unknown")),
            overall=int(s.get("scores", {}).get("overall", 0)),
            top_skills=list(s.get("scores", {}).get("top_skills", [])),
            llm_model=s.get("scores", {}).get("llm_model"),
            llm_metadata=None,
        )
        sid = create_document("score", score_doc)
        saved_scores.append(sid)

    return {"job_id": job_id, "score_ids": saved_scores}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
