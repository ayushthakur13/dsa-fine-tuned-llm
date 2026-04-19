"""FastAPI app scaffold for model generation and evaluation endpoints."""

from fastapi import FastAPI


app = FastAPI(title="DSA Solver API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
