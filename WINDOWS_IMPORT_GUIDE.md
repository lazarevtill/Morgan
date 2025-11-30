# Import Conversations on Windows

## Quick Import (PowerShell or CMD)

### Option 1: Using Batch File (Easiest)
```cmd
cd C:\Users\lazarev\Documents\GitHub\Morgan
import_windows.bat conversations.json
```

### Option 2: Direct Python (PowerShell)
```powershell
cd C:\Users\lazarev\Documents\GitHub\Morgan\morgan-rag
$env:PYTHONPATH = ".\local_libs;.;$env:PYTHONPATH"
python scripts\import_conversations_windows.py ..\conversations.json
```

### Option 3: Direct Python (CMD)
```cmd
cd C:\Users\lazarev\Documents\GitHub\Morgan\morgan-rag
set PYTHONPATH=.\local_libs;.;%PYTHONPATH%
python scripts\import_conversations_windows.py ..\conversations.json
```

---

## If You Get "ModuleNotFoundError"

The Windows-compatible script (`import_conversations_windows.py`) automatically handles paths, but if you still get errors:

### Check Python Installation
```powershell
python --version
# Should show Python 3.8+
```

### Verify Dependencies
```powershell
cd morgan-rag
python -c "import sys; sys.path.insert(0, 'local_libs'); import numpy, structlog; print('Dependencies OK')"
```

### Install Missing Dependencies
```powershell
cd morgan-rag
python -m pip install --target=.\local_libs `
  python-dotenv pydantic-settings `
  numpy structlog `
  sentence-transformers transformers `
  requests tqdm openai qdrant-client `
  aiohttp yarl
```

---

## Full Step-by-Step (Windows)

### 1. Open PowerShell
```powershell
# Press Win+X, select "Windows PowerShell"
```

### 2. Navigate to Morgan
```powershell
cd C:\Users\lazarev\Documents\GitHub\Morgan
```

### 3. Check Qdrant is Running
```powershell
curl http://localhost:6333/collections
```

If not running:
```powershell
cd morgan-rag
docker-compose up -d qdrant
```

### 4. Import Conversations
```powershell
# Using the batch file (easiest)
.\import_windows.bat .\conversations.json

# OR using PowerShell directly
cd morgan-rag
$env:PYTHONPATH = ".\local_libs;.;$env:PYTHONPATH"
python scripts\import_conversations_windows.py ..\conversations.json
```

### 5. Wait for Import
```
Importing 314 conversations...
============================================================
Processing conversation 10/314... (9 imported, 45 turns)
Processing conversation 20/314... (18 imported, 90 turns)
...
============================================================

✅ Import Complete!
   Imported: 314 conversations
   Total turns: 1,585

Your conversations are now searchable in Morgan! 🚀
```

---

## Troubleshooting Windows Issues

### Issue: "python: command not found"
**Solution**: Use `python3` or `py` instead:
```powershell
py scripts\import_conversations_windows.py ..\conversations.json
```

### Issue: "Cannot find path"
**Solution**: Use absolute path:
```powershell
python scripts\import_conversations_windows.py C:\Users\lazarev\Documents\GitHub\Morgan\conversations.json
```

### Issue: "PermissionError"
**Solution**: Run as Administrator:
```powershell
# Right-click PowerShell -> "Run as Administrator"
```

### Issue: "Connection refused to Qdrant"
**Solution**: Check Docker Desktop is running and Qdrant is up:
```powershell
docker ps | findstr qdrant
# Should show: morgan-qdrant ... Up ...

# If not running:
cd morgan-rag
docker-compose up -d qdrant
```

### Issue: Slow import on Windows
**Solution**: This is normal - Windows file I/O is slower. Expected time:
- 314 conversations: 15-30 minutes
- Shows progress every 10 conversations

---

## Environment Variables (PowerShell)

If you need to set variables manually:

```powershell
# Temporary (current session only)
$env:PYTHONPATH = "C:\Users\lazarev\Documents\GitHub\Morgan\morgan-rag\local_libs;C:\Users\lazarev\Documents\GitHub\Morgan\morgan-rag;$env:PYTHONPATH"
$env:QDRANT_URL = "http://localhost:6333"

# Permanent (for user)
[System.Environment]::SetEnvironmentVariable('PYTHONPATH', 'C:\Users\lazarev\Documents\GitHub\Morgan\morgan-rag\local_libs', 'User')
```

---

## WSL Alternative (Faster)

If you have WSL (Windows Subsystem for Linux), it's faster:

```bash
# In WSL terminal
cd /mnt/c/Users/lazarev/Documents/GitHub/Morgan/morgan-rag
export PYTHONPATH=./local_libs:$PYTHONPATH
python3 scripts/import_conversations.py /mnt/c/Users/lazarev/Documents/GitHub/Morgan/conversations.json
```

WSL advantages:
- Faster file I/O
- Native Python performance
- Same commands as Linux/Mac

---

## Verify Import Success

After import completes:

```powershell
# Check collections
curl http://localhost:6333/collections | python -m json.tool

# Should show:
# morgan_conversations: ~314 points
# morgan_turns: ~1585 points
```

---

## Quick Reference

### Import conversations.json
```cmd
import_windows.bat conversations.json
```

### Create backup
```powershell
cd morgan-rag
python scripts\backup_qdrant.py
```

### Check status
```powershell
curl http://localhost:6333/collections
```

---

## Files Created

- ✅ `import_windows.bat` - One-click import for Windows
- ✅ `scripts/import_conversations_windows.py` - Windows-compatible script
- ✅ Original script still works in WSL/Linux

**Use the Windows-specific files for best results on Windows!**
