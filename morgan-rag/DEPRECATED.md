# ⚠️ DEPRECATED - Old Morgan System

**This directory contains the old monolithic Morgan implementation and is now DEPRECATED.**

## Status

- **Deprecated Date:** December 8, 2025
- **Replacement:** New client-server architecture in `morgan-server/` and `morgan-cli/`
- **Maintenance:** No new features, critical bug fixes only
- **Removal:** Will be archived in a future release

## Why Deprecated?

The old Morgan system had several limitations:
- Monolithic architecture with tight coupling
- Difficult to deploy and scale
- Mixed concerns between CLI and core logic
- Limited API access for custom clients
- Complex configuration management

## Migration Path

**Please migrate to the new system:**

1. **Read the Migration Guide:** See [MIGRATION.md](../MIGRATION.md) in the root directory
2. **Install New Packages:**
   - Server: `cd ../morgan-server && pip install -e .`
   - Client: `cd ../morgan-cli && pip install -e .`
3. **Configure Server:** Set up environment variables or config files
4. **Start Services:** Use Docker Compose or manual deployment
5. **Use New CLI:** `morgan chat` instead of old CLI

## New System Benefits

- **Clean Architecture:** Separate client and server
- **Multiple Clients:** TUI, web, custom apps
- **Better Deployment:** Docker support, health checks, metrics
- **Enhanced Features:** Improved empathic engine, knowledge engine, personalization
- **Production Ready:** Structured logging, monitoring, graceful shutdown

## Documentation

- **[Migration Guide](../MIGRATION.md)** - Step-by-step migration instructions
- **[Server Documentation](../morgan-server/README.md)** - New server documentation
- **[Client Documentation](../morgan-cli/README.md)** - New client documentation
- **[Docker Documentation](../docker/README.md)** - Deployment guide

## Support

For migration assistance:
- Check the [Migration Guide](../MIGRATION.md)
- Review [Documentation Index](../DOCUMENTATION.md)
- Open a GitHub issue if you encounter problems

## Archive Notice

This directory is kept for reference during the migration period. It will be:
1. Moved to an `archive/` directory in a future release
2. Eventually removed from the main repository
3. Available in git history for reference

**Please do not use this code for new projects or deployments.**

---

**Use the new system:** `morgan-server/` and `morgan-cli/`
