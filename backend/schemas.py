"""
Database Schemas for the AI Resume Agent

Each Pydantic model corresponds to a MongoDB collection. Collection name is the
lowercase class name.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Job(BaseModel):
    title: Optional[str] = Field(None, description="Optional job title")
    description: str = Field(..., description="Full job description text")
    source: Optional[str] = Field(None, description="Source or reference for the JD")

class Candidate(BaseModel):
    name: str = Field(..., description="Candidate display name")
    filename: Optional[str] = Field(None, description="Uploaded filename if any")
    raw_text: str = Field(..., description="Extracted resume text")
    note: Optional[str] = Field(None, description="Import notes or parsing warnings")

class Score(BaseModel):
    job_id: str = Field(..., description="Reference to Job document id")
    candidate_id: str = Field(..., description="Reference to Candidate document id")
    skill_match: int = Field(..., ge=0, le=100)
    years: int = Field(..., ge=0)
    seniority: str = Field(...)
    overall: int = Field(..., ge=0, le=100)
    top_skills: List[str] = Field(default_factory=list)
    llm_model: Optional[str] = None
    llm_metadata: Optional[Dict[str, Any]] = None
