#!/usr/bin/env python3
"""Backup Qdrant collections"""
import requests
import json
from datetime import datetime
from pathlib import Path

QDRANT_URL = "http://localhost:6333"
BACKUP_DIR = Path("./backups") / f"qdrant_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

collections = [
    "morgan_conversations",
    "morgan_turns",
    "morgan_knowledge",
    "morgan_knowledge_hierarchical",
    "morgan_memory",
    "morgan_memories"
]

print("Creating Qdrant backups...")
print(f"Backup directory: {BACKUP_DIR}")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

for collection in collections:
    print(f"\nBacking up: {collection}")

    try:
        # Create snapshot
        resp = requests.post(f"{QDRANT_URL}/collections/{collection}/snapshots")
        data = resp.json()

        if data.get("status") == "ok":
            snapshot_name = data["result"]["name"]

            # Download snapshot
            snap_resp = requests.get(
                f"{QDRANT_URL}/collections/{collection}/snapshots/{snapshot_name}"
            )

            if snap_resp.status_code == 200:
                backup_file = BACKUP_DIR / f"{collection}_{snapshot_name}"
                backup_file.write_bytes(snap_resp.content)
                print(f"  ✅ Saved: {backup_file.name} ({len(snap_resp.content)} bytes)")
            else:
                print(f"  ⚠️  Could not download snapshot")
        else:
            print(f"  ⚠️  Snapshot creation failed")

    except Exception as e:
        print(f"  ❌ Error: {e}")

print(f"\n✅ Backup complete!")
print(f"Location: {BACKUP_DIR}")
