#!/usr/bin/env python3
"""
Test script for Morgan services
"""
import asyncio
import aiohttp
import json
from typing import Dict, Any


async def test_service(session: aiohttp.ClientSession, name: str, url: str) -> Dict[str, Any]:
    """Test a service health endpoint"""
    try:
        async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {"service": name, "status": "healthy", "data": data}
            else:
                return {"service": name, "status": "unhealthy", "error": f"HTTP {resp.status}"}
    except Exception as e:
        return {"service": name, "status": "error", "error": str(e)}


async def main():
    """Test all Morgan services"""
    services = {
        "Core": "http://localhost:8000",
        "LLM": "http://localhost:8001",
        "TTS": "http://localhost:8002",
        "STT": "http://8003",
        "VAD": "http://localhost:8004",
    }
    
    print("=" * 80)
    print("Morgan Services Health Check")
    print("=" * 80)
    
    async with aiohttp.ClientSession() as session:
        tasks = [test_service(session, name, url) for name, url in services.items()]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            service = result["service"]
            status = result["status"]
            
            status_symbol = "✓" if status == "healthy" else "✗"
            status_color = "\033[92m" if status == "healthy" else "\033[91m"
            reset_color = "\033[0m"
            
            print(f"\n{status_color}{status_symbol} {service} Service: {status}{reset_color}")
            
            if "data" in result:
                print(f"  Version: {result['data'].get('version', 'unknown')}")
                print(f"  Uptime: {result['data'].get('uptime', 'unknown')}")
            elif "error" in result:
                print(f"  Error: {result['error']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())



