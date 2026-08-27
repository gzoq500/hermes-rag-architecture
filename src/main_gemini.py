#!/usr/bin/env python3
"""Compatibility wrapper for the Hermes RAG gateway executable."""

from rag_gateway.__main__ import build_app, main


def create_application():
    """Return a configured ASGI application for legacy importers."""
    return build_app()


if __name__ == "__main__":
    main()
