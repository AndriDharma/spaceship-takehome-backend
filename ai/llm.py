"""Gemini on Vertex AI. The only model provider in this application.

Uses ChatGoogleGenerativeAI from langchain-google-genai, pointed at Vertex by
passing a project and credentials. The older ChatVertexAI class does the same
job but is deprecated as of LangChain 3.2 and scheduled for removal in 4.0.

Credentials resolve the same way in both environments. Locally a service
account file is read explicitly; on Cloud Run the file is absent, credentials
are left unset, and the client falls through to Application Default
Credentials, which the attached service account supplies from the metadata
server. No environment flag decides this - the presence of the file does.
"""

import os
from functools import lru_cache
from typing import Optional

from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI

from core import config

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _credentials() -> Optional[service_account.Credentials]:
    path = config.SERVICE_ACCOUNT_FILE

    if path and os.path.exists(path):
        return service_account.Credentials.from_service_account_file(
            path,
            scopes=_SCOPES,
        )

    # Cloud Run: returning None lets the client library go and ask the
    # metadata server itself.
    return None


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0, max_tokens: int = 2048) -> ChatGoogleGenerativeAI:
    """
    Cached per (temperature, max_tokens).

    Every call in this application is one of two kinds - deciding something, or
    writing an answer - and both want temperature 0. Deciding, because a router
    that picks a different tool for the same question is not a router.
    Answering, because the numbers in the answer come from the result set and
    there is nothing to be creative about.

    Worth knowing: Gemini 3 defaults temperature to 1.0, so leaving it unset
    here would make both of those calls nondeterministic. It is passed
    explicitly for that reason rather than as a formality.
    """
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        credentials=_credentials(),
        project=config.GCP_PROJECT_ID,
        location=config.VERTEX_REGION,
        temperature=temperature,
        max_tokens=max_tokens,
    )
