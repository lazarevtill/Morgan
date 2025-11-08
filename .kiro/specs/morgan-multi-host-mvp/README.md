# Morgan Multi-Host MVP - Project Overview (V3 - Flexible Configuration)

> **Status**: Planning Complete - Flexible Multi-Host Deployment
> **Version**: 3.0.0
> **Last Updated**: 2025-11-02

---

## 📋 Project Summary

This directory contains the complete specification for deploying **Morgan V2 AI Assistant** across **any number of available hosts** (minimum 1, recommended 3-7) with flexible hardware configurations. The system automatically detects and optimizes for:

- **GPU Hosts**: NVIDIA GPUs with CUDA 12.4+ (any VRAM from 6GB to 12GB+)
- **CPU Hosts**: Intel i7 9th gen / i9 11th/13th gen (32GB-64GB RAM)
- **ARM64 Hosts**: Apple Silicon M1/M2/M3 (16GB+ RAM)

Morgan V2 includes:
- **Advanced RAG** (Retrieval-Augmented Generation) with hierarchical search
- **Emotion Detection** (11 specialized modules)
- **Empathy Engine** (5 modules for emotional support)
- **Learning & Adaptation** (6 modules for continuous improvement)
- **Multi-Stage Search** (coarse → medium → fine)
- **Distributed Inference** with automatic GPU/CPU allocation

---

## 🎯 MVP Goals

1. **Flexible Architecture**: Auto-detect and deploy on 1-7+ hosts
2. **GPU Optimization**: Automatically assign workloads based on detected VRAM (12GB/8GB/6GB)
3. **CPU Services**: Distribute RAG, emotion, empathy, learning across available CPU hosts
4. **High Availability**: Implement failover when multiple hosts available
5. **Simple Deployment**: Minimal configuration, maximum automation
6. **Optional API Gateway**: Kong (optional), or direct Consul DNS access for MVP

---

## 📝 Document Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | 2025-11-02 | Initial multi-host specification | Claude Code |
| 2.0.0 | 2025-11-02 | Updated for v2-0.0.1 branch with RAG, emotion, empathy, learning | Claude Code |
| 3.0.0 | 2025-11-02 | Updated for flexible host count, CUDA 12.4, Kong gateway option | Claude Code |
| 3.1.0 | 2025-11-02 | CLI-only interface (Click framework), removed WebUI from MVP | Claude Code |
| 3.2.0 | 2025-11-02 | PostgreSQL (code-only logic, no triggers/functions) + MinIO (S3 storage) | Claude Code |

---

## 📁 Document Structure

| Document | Purpose | Status |
|----------|---------|--------|
| **[requirements.md](./requirements.md)** | 13 functional requirements for flexible deployment | ✅ V3.2 |
| **[design.md](./design.md)** | Architecture for dynamic host allocation | ⏳ Pending |
| **[tasks.md](./tasks.md)** | Implementation tasks for flexible deployment | ⏳ Pending |
| **README.md** (this file) | Project overview and quick reference | ✅ V3.2 |

---

**Status**: ✅ Requirements updated for V3.2 (PostgreSQL + MinIO)
**Latest Changes**:
- PostgreSQL with code-only logic (no triggers/functions, all logic in Python services)
- MinIO for S3-compatible file storage (local or remote)
- CLI-only interface using Click framework
- Kong as optional gateway, dynamic hardware detection, flexible host count
**Next Steps**: Update design.md and tasks.md for PostgreSQL schema and MinIO architecture
