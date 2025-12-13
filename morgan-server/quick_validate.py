#!/usr/bin/env python3
"""
Quick Production Readiness Validation

This script performs a quick validation without running the full test suite.
For full validation, use validate_production_readiness.py
"""

import asyncio
import sys
from pathlib import Path

import aiohttp


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")


def print_check(name: str, passed: bool, message: str = ""):
    symbol = "[OK]" if passed else "[FAIL]"
    color = Colors.GREEN if passed else Colors.RED
    print(f"{color}{symbol} {name}{Colors.RESET}")
    if message:
        print(f"     {message}")


def print_info(text: str):
    print(f"{Colors.BLUE}[INFO] {text}{Colors.RESET}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.RESET}")


async def check_service(url: str, name: str) -> bool:
    """Check if a service is accessible"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=3) as resp:
                return resp.status == 200
    except:
        return False


async def main():
    print(f"\n{Colors.BOLD}Morgan Server - Quick Validation{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")
    
    # Detect paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print_info(f"Project root: {project_root}")
    print_info(f"Morgan server: {script_dir}")
    print()
    
    results = {}
    
    # Check documentation
    print_header("DOCUMENTATION")
    docs = [
        (script_dir / "README.md", "Server README"),
        (project_root / "morgan-cli" / "README.md", "Client README"),
        (project_root / "docker" / "docker-compose.yml", "Docker Compose"),
        (project_root / "MIGRATION.md", "Migration Guide"),
        (project_root / "DOCUMENTATION.md", "Main Documentation"),
    ]
    
    for path, name in docs:
        exists = path.exists()
        results[name] = exists
        size_info = f"({path.stat().st_size} bytes)" if exists else f"Missing: {path}"
        print_check(name, exists, size_info)
    
    # Check services
    print_header("SERVICES")
    
    print_info("Checking Qdrant (Vector Database)...")
    qdrant_ok = await check_service("http://localhost:6333/healthz", "Qdrant")
    results["Qdrant"] = qdrant_ok
    print_check("Qdrant", qdrant_ok, 
                "Running at http://localhost:6333" if qdrant_ok 
                else "Not running. Start with: docker run -p 6333:6333 qdrant/qdrant")
    
    print_info("Checking Ollama (LLM)...")
    ollama_ok = await check_service("http://localhost:11434/api/tags", "Ollama")
    results["Ollama"] = ollama_ok
    print_check("Ollama", ollama_ok,
                "Running at http://localhost:11434" if ollama_ok
                else "Not running. Install from https://ollama.ai and run 'ollama serve'")
    
    print_info("Checking Morgan Server...")
    server_ok = await check_service("http://localhost:8080/health", "Morgan Server")
    results["Morgan Server"] = server_ok
    print_check("Morgan Server", server_ok,
                "Running at http://localhost:8080" if server_ok
                else "Not running. Start with: python -m morgan_server")
    
    # Check test structure
    print_header("TEST STRUCTURE")
    tests_dir = script_dir / "tests"
    tests_exist = tests_dir.exists()
    results["Tests Directory"] = tests_exist
    if tests_exist:
        test_files = list(tests_dir.rglob("test_*.py"))
        print_check("Tests Directory", True, f"Found {len(test_files)} test files")
    else:
        print_check("Tests Directory", False, "tests/ directory not found")
    
    # Summary
    print_header("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nTotal Checks: {total}")
    print(f"Passed: {Colors.GREEN}{passed}{Colors.RESET}")
    print(f"Failed: {Colors.RED}{total - passed}{Colors.RESET}\n")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}[SUCCESS] All checks passed!{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}[PARTIAL] Some checks failed{Colors.RESET}")
        print(f"{Colors.YELLOW}This is expected if services aren't running yet.{Colors.RESET}\n")
        
        print(f"{Colors.BOLD}To start services:{Colors.RESET}")
        if not results.get("Qdrant"):
            print("  docker run -p 6333:6333 qdrant/qdrant")
        if not results.get("Ollama"):
            print("  # Install from https://ollama.ai, then:")
            print("  ollama serve")
        if not results.get("Morgan Server"):
            print("  python -m morgan_server")
        print()
        
        print(f"{Colors.BOLD}To run full validation (including tests):{Colors.RESET}")
        print("  python validate_production_readiness.py")
        print()
        
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        sys.exit(1)
