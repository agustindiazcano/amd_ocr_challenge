# Agent Architecture

This document outlines the agent architecture for the **03 Enterprise RAG Engine**.

## Project Context
The objective of this phase of the AMD AI Challenge is to build a high-performance **Retrieval-Augmented Generation (RAG)** pipeline. This pipeline must be heavily optimized to run efficiently on **AMD ROCm hardware** under strict execution constraints:
- **Memory Limit:** Maximum of 48 GB VRAM.
- **Execution Time:** Less than 30 seconds per inference/query.

## Supported Agents
The RAG engine supports different LLM backends to power the conversational and retrieval capabilities. 
- [Gemini Implementation](gemini.md)
- [Claude Implementation](claude.md)
