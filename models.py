from pydantic import BaseModel, Field
from typing import Optional, List


# Pydantic model do żądania
class RequestBody(BaseModel):
    request: str

class Affiliation(BaseModel):
    institution: Optional[str] = Field(default=None, nullable=True)
    department: Optional[str] = Field(default=None, nullable=True)
    role: Optional[str] = Field(default=None, nullable=True)
    country: Optional[str] = Field(default=None, nullable=True)
    start_date: Optional[str] = Field(default=None, nullable=True)
    end_date: Optional[str] = Field(default=None, nullable=True)


class Keyword(BaseModel):
    keyword: Optional[str] = Field(default=None, nullable=True)


class Publication(BaseModel):
    title: Optional[str] = Field(default=None, nullable=True)
    journal: Optional[str] = Field(default=None, nullable=True)
    doi: Optional[str] = Field(default=None, nullable=True)
    year: Optional[str] = Field(default=None, nullable=True)


class Education(BaseModel):
    degree: Optional[str] = Field(default=None, nullable=True)
    field: Optional[str] = Field(default=None, nullable=True)
    institution: Optional[str] = Field(default=None, nullable=True)
    country: Optional[str] = Field(default=None, nullable=True)
    start_date: Optional[str] = Field(default=None, nullable=True)
    end_date: Optional[str] = Field(default=None, nullable=True)


class ResearcherInfo(BaseModel):
    full_name: Optional[str] = Field(default=None, nullable=True)
    orcid_id: Optional[str] = Field(default=None, nullable=True)
    email: Optional[str] = Field(default=None, nullable=True)
    country: Optional[str] = Field(default=None, nullable=True)
    primary_affiliation: Optional[str] = Field(default=None, nullable=True)


class Scientist(BaseModel):
    researcher: ResearcherInfo
    affiliations: List[Affiliation] = Field(default_factory=list)
    keywords: List[Keyword] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    publications: List[Publication] = Field(default_factory=list)
