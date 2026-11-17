from pydantic import BaseModel, Field


class Source(BaseModel):
    text: str
    file: str
    page: int


class RAGResponse(BaseModel):
    answer: str = Field(description="Answer to the question based strictly on the provided context")
    sources: list[Source] = Field(default_factory=list, description="Chunks from the index that support the answer")
    refused: bool = Field(default=False, description="True when the context does not contain enough information to answer")
    refusal_reason: str | None = Field(default=None, description="Why the question was refused, if refused=True")
