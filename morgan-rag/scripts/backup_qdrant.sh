#!/bin/bash
# Qdrant Backup Script - Creates snapshots of all collections

BACKUP_DIR="./backups/qdrant_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating Qdrant backups..."
echo "Backup directory: $BACKUP_DIR"

# List of collections to backup
collections=(
    "morgan_conversations"
    "morgan_turns"
    "morgan_knowledge"
    "morgan_knowledge_hierarchical"
    "morgan_memory"
    "morgan_memories"
)

for collection in "${collections[@]}"; do
    echo "Backing up: $collection"

    # Create snapshot
    response=$(curl -s -X POST "http://localhost:6333/collections/$collection/snapshots")

    if echo "$response" | grep -q '"status":"ok"'; then
        # Extract snapshot name
        snapshot=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['result']['name'])" 2>/dev/null)

        if [ -n "$snapshot" ]; then
            # Download snapshot
            curl -s "http://localhost:6333/collections/$collection/snapshots/$snapshot" \
                -o "$BACKUP_DIR/${collection}_${snapshot}"
            echo "  ✅ Saved: ${collection}_${snapshot}"
        else
            echo "  ⚠️  Could not extract snapshot name"
        fi
    else
        echo "  ❌ Failed to create snapshot"
    fi
done

# Also backup docker volume
echo ""
echo "Creating docker volume backup..."
docker run --rm -v morgan-rag_qdrant_data:/data \
    -v "$(pwd)/backups:/backup" \
    alpine tar czf "/backup/qdrant_volume_$(date +%Y%m%d_%H%M%S).tar.gz" /data

echo ""
echo "✅ Backup complete!"
echo "Location: $BACKUP_DIR"
ls -lh "$BACKUP_DIR"
