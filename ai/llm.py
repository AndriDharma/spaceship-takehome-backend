"""Gemini on Vertex AI. The only model provider in this application.

Credentials resolve the same way in both environments. Locally a service
account file is read explicitly; on Cloud Run the file is absent and the
constructor falls through to Application Default Credentials, which the
attached service account supplies. No environment flag decides this - the
presence of the file does.
"""

import os
from functools import lru_cache
from typing import Optional

from google.oauth2 import service_account
from langchain_google_vertexai import ChatVertexAI

from core import config

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _credentials() -> Optional[service_account.Credentials]:
    path = config.SERVICE_ACCOUNT_FILE

    if path and os.path.exists(path):
        return service_account.Credentials.from_service_account_file(
            path,
            scopes=_SCOPES,
        )

    # Cloud Run: the metadata server answers, and passing credentials=None
    # lets the client library go and ask it.
    return None


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0, max_output_tokens: int = 2048) -> ChatVertexAI:
    """
    Cached per (temperature, max_output_tokens).

    Every call in this application is one of two kinds - deciding something, or
    writing an answer - and both want temperature 0. Deciding, because a router
    that picks a different tool for the same question is not a router.
    Answering, because the numbers in the answer come from the result set and
    there is nothing to be creative about.
    """
    return ChatVertexAI(
        model_name=config.GEMINI_MODEL,
        project=config.GCP_PROJECT_ID,
        location=config.VERTEX_REGION,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        credentials=_credentials(),
    )
