#!/usr/bin/env python3
"""
Restore Qdrant collections from snapshots

Usage:
    python3 scripts/restore_qdrant.py backups/qdrant_20251129_232004/
"""
import requests
import sys
from pathlib import Path

QDRANT_URL = "http://localhost:6333"


def restore_snapshots(backup_dir: Path):
    """Restore all snapshots from a backup directory"""
    if not backup_dir.exists():
        print(f"❌ Backup directory not found: {backup_dir}")
        return False

    snapshot_files = list(backup_dir.glob("*.snapshot"))
    if not snapshot_files:
        print(f"❌ No snapshot files found in: {backup_dir}")
        return False

    print(f"Restoring from: {backup_dir}")
    print(f"Found {len(snapshot_files)} snapshots")
    print()

    success_count = 0

    for snapshot_file in snapshot_files:
        # Extract collection name from filename
        # Format: morgan_conversations_morgan_conversations-*.snapshot
        parts = snapshot_file.name.split("_")
        collection = "_".join(parts[:2])  # morgan_conversations

        print(f"Restoring: {collection}")

        try:
            # 1. Upload snapshot
            upload_url = f"{QDRANT_URL}/collections/{collection}/snapshots/upload"

            with open(snapshot_file, "rb") as f:
                upload_resp = requests.put(
                    upload_url,
                    data=f,
                    headers={"Content-Type": "application/octet-stream"},
                )

            if upload_resp.status_code not in [200, 201]:
                print(f"  ❌ Upload failed: {upload_resp.text}")
                continue

            print(f"  ✅ Uploaded {snapshot_file.name}")

            # 2. Get snapshot name from upload response
            upload_data = upload_resp.json()
            if upload_data.get("status") == "ok":
                # Extract just the snapshot name (without collection prefix)
                snapshot_name = snapshot_file.name.replace(f"{collection}_", "")

                # 3. Recover from snapshot
                recover_url = f"{QDRANT_URL}/collections/{collection}/snapshots/recover"
                recover_resp = requests.put(
                    recover_url, json={"location": snapshot_name}
                )

                if recover_resp.status_code in [200, 201]:
                    print(f"  ✅ Restored {collection}")
                    success_count += 1
                else:
                    print(f"  ⚠️  Recovery response: {recover_resp.text}")
            else:
                print(f"  ⚠️  Upload response: {upload_data}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        print()

    print(f"✅ Restore complete! ({success_count}/{len(snapshot_files)} successful)")
    return success_count > 0


def verify_restore():
    """Verify that collections were restored successfully"""
    try:
        resp = requests.get(f"{QDRANT_URL}/collections")
        data = resp.json()

        if data.get("status") == "ok":
            collections = data["result"]["collections"]

            print("\n📊 Verification:")
            for c in collections:
                name = c["name"]
                detail_resp = requests.get(f"{QDRANT_URL}/collections/{name}")
                detail = detail_resp.json()

                if detail.get("status") == "ok":
                    points = detail["result"]["points_count"]
                    status = detail["result"]["status"]
                    print(f"  {name}: {points} points (status: {status})")

    except Exception as e:
        print(f"\n⚠️  Verification failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/restore_qdrant.py <backup_directory>")
        print("\nExample:")
        print("  python3 scripts/restore_qdrant.py backups/qdrant_20251129_232004/")
        sys.exit(1)

    backup_dir = Path(sys.argv[1])
    success = restore_snapshots(backup_dir)

    if success:
        verify_restore()
        print("\n✅ All done! Your data has been restored.")
    else:
        print("\n❌ Restore failed. Check the errors above.")
        sys.exit(1)
