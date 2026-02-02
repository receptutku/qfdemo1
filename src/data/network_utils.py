"""
Network utility functions with comprehensive diagnostics.

Provides network classification, proxy detection, and health monitoring.
"""

import os
import json
import socket
import ssl
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.request import urlopen, Request
from urllib.error import URLError


PROXY_ENV_VARS = [
    "HTTP_PROXY", "http_proxy",
    "HTTPS_PROXY", "https_proxy",
    "ALL_PROXY", "all_proxy",
    "NO_PROXY", "no_proxy",
]

# Test endpoints for diagnostics
TEST_ENDPOINTS = {
    # Exchange APIs
    "kraken": "https://api.kraken.com/0/public/Time",
    "coinbase": "https://api.coinbase.com/v2/time",
    "okx": "https://www.okx.com/api/v5/public/time",
    "bybit": "https://api.bybit.com/v5/market/time",
    # General internet (should always work)
    "google": "https://www.google.com/generate_204",
    "cloudflare": "https://cloudflare.com/cdn-cgi/trace",
}


@dataclass
class EndpointTest:
    """Result of testing a single endpoint."""
    name: str
    url: str
    success: bool
    latency_ms: int
    error: Optional[str] = None
    error_type: Optional[str] = None  # "ssl", "timeout", "dns", "connection", "http"


@dataclass
class NetworkProfile:
    """Complete network diagnostic profile."""
    timestamp: str
    hostname: str
    proxy_vars: Dict[str, str]
    endpoints: Dict[str, EndpointTest]
    classification: str
    recommendation: str
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result["endpoints"] = {k: asdict(v) for k, v in self.endpoints.items()}
        return result
    
    def save(self, filepath: Path):
        """Save profile to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ExchangeHealth:
    """Health tracking for a single exchange."""
    name: str
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    last_error: Optional[str] = None
    unhealthy_until: Optional[datetime] = None
    
    def record_success(self):
        self.last_success = datetime.now(timezone.utc)
        self.consecutive_failures = 0
        self.total_requests += 1
        self.unhealthy_until = None
    
    def record_failure(self, error: str, cooldown_minutes: int = 10):
        self.last_failure = datetime.now(timezone.utc)
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_requests += 1
        self.last_error = error
        
        # Set cooldown if TLS/network error
        if any(x in error.lower() for x in ["ssl", "tls", "handshake", "certificate"]):
            self.unhealthy_until = datetime.now(timezone.utc) + timedelta(minutes=cooldown_minutes)
    
    @property
    def is_healthy(self) -> bool:
        """Check if exchange is currently healthy."""
        if self.unhealthy_until:
            if datetime.now(timezone.utc) < self.unhealthy_until:
                return False
        return self.consecutive_failures < 5
    
    @property
    def cooldown_remaining(self) -> Optional[int]:
        """Seconds remaining in cooldown, or None if not in cooldown."""
        if self.unhealthy_until:
            remaining = (self.unhealthy_until - datetime.now(timezone.utc)).total_seconds()
            return max(0, int(remaining))
        return None


def check_proxy_settings() -> Dict[str, str]:
    """Check for proxy environment variables."""
    found = {}
    for var in PROXY_ENV_VARS:
        value = os.environ.get(var)
        if value:
            found[var] = value
    return found


def print_proxy_warning():
    """Print warning if proxy settings are detected."""
    proxies = check_proxy_settings()
    
    if proxies:
        print("\n" + "=" * 60)
        print("⚠️  PROXY SETTINGS DETECTED")
        print("=" * 60)
        for var, value in proxies.items():
            if "@" in value:
                masked = value.split("@")[-1]
                value = f"***@{masked}"
            print(f"  {var} = {value}")
        print("\nThis may cause SSL errors with exchange APIs.")
        print("To disable: unset HTTP_PROXY HTTPS_PROXY ALL_PROXY")
        print("=" * 60 + "\n")
        return True
    return False


def test_endpoint(name: str, url: str, timeout: int = 10) -> EndpointTest:
    """Test a single endpoint and return detailed result."""
    start = time.time()
    
    try:
        req = Request(url, headers={"User-Agent": "btc-backtest/1.0"})
        resp = urlopen(req, timeout=timeout)
        latency = int((time.time() - start) * 1000)
        resp.close()
        return EndpointTest(name=name, url=url, success=True, latency_ms=latency)
    
    except ssl.SSLError as e:
        latency = int((time.time() - start) * 1000)
        return EndpointTest(
            name=name, url=url, success=False, latency_ms=latency,
            error=str(e), error_type="ssl"
        )
    
    except socket.timeout:
        latency = int((time.time() - start) * 1000)
        return EndpointTest(
            name=name, url=url, success=False, latency_ms=latency,
            error="Connection timed out", error_type="timeout"
        )
    
    except socket.gaierror as e:
        latency = int((time.time() - start) * 1000)
        return EndpointTest(
            name=name, url=url, success=False, latency_ms=latency,
            error=f"DNS resolution failed: {e}", error_type="dns"
        )
    
    except ConnectionRefusedError:
        latency = int((time.time() - start) * 1000)
        return EndpointTest(
            name=name, url=url, success=False, latency_ms=latency,
            error="Connection refused", error_type="connection"
        )
    
    except URLError as e:
        latency = int((time.time() - start) * 1000)
        error_str = str(e.reason)
        
        # Classify error type
        if "ssl" in error_str.lower() or "certificate" in error_str.lower():
            error_type = "ssl"
        elif "timed out" in error_str.lower():
            error_type = "timeout"
        else:
            error_type = "connection"
        
        return EndpointTest(
            name=name, url=url, success=False, latency_ms=latency,
            error=error_str, error_type=error_type
        )
    
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return EndpointTest(
            name=name, url=url, success=False, latency_ms=latency,
            error=str(e), error_type="unknown"
        )


def classify_network(results: Dict[str, EndpointTest]) -> Tuple[str, str]:
    """
    Classify network status based on test results.
    
    Returns: (classification, recommendation)
    """
    # Count successes by category
    exchange_tests = ["kraken", "coinbase", "okx", "bybit"]
    general_tests = ["google", "cloudflare"]
    
    exchange_success = sum(1 for k in exchange_tests if k in results and results[k].success)
    general_success = sum(1 for k in general_tests if k in results and results[k].success)
    
    ssl_errors = sum(1 for r in results.values() if r.error_type == "ssl")
    total_tests = len(results)
    
    # Classification logic
    if exchange_success >= 2:
        return (
            "✅ HEALTHY",
            "Network is working. You can proceed with data fetching."
        )
    
    if general_success == 0:
        return (
            "🚫 NO INTERNET",
            "Check your internet connection. Neither Google nor Cloudflare are reachable."
        )
    
    if ssl_errors >= total_tests - 1:
        return (
            "🔒 GENERAL TLS BROKEN",
            "SSL/TLS is broken for all endpoints. This is likely a device-level issue:\n"
            "  1. Check if corporate proxy/firewall is doing SSL inspection\n"
            "  2. Check if antivirus is intercepting HTTPS\n"
            "  3. Try: export REQUESTS_CA_BUNDLE=/path/to/certificates\n"
            "  4. Use a VPN or different network"
        )
    
    if general_success >= 1 and exchange_success == 0:
        return (
            "🚧 SELECTIVE API BLOCKING",
            "Exchange APIs are blocked but general internet works. Possible causes:\n"
            "  1. ISP/country blocking crypto exchanges\n"
            "  2. Corporate firewall blocking trading sites\n"
            "  3. DNS filtering (try: use 1.1.1.1 or 8.8.8.8)\n"
            "  → Recommended: Use a VPN"
        )
    
    if ssl_errors >= 2 and general_success >= 1:
        return (
            "🔍 DEVICE-LEVEL PROXY SUSPECTED",
            "Some endpoints have SSL issues while others work. Suggests:\n"
            "  1. Selective SSL inspection (work/school network)\n"
            "  2. Antivirus with HTTPS scanning enabled\n"
            "  → Try disabling antivirus HTTPS scanning\n"
            "  → Or use a VPN to bypass inspection"
        )
    
    if exchange_success >= 1:
        return (
            "⚠️ PARTIAL - SOME EXCHANGES WORK",
            f"Some exchanges are reachable ({exchange_success}/4). Use fallback mode:\n"
            "  python -m src.main live --fallback okx,bybit,kucoin,kraken_direct"
        )
    
    return (
        "❓ UNKNOWN ISSUE",
        "Mixed results. Check individual endpoint errors above for details."
    )


def run_full_diagnostics(save_path: Optional[Path] = None) -> NetworkProfile:
    """
    Run comprehensive network diagnostics.
    
    Returns NetworkProfile with classification and recommendations.
    """
    print("\n[Network Diagnostics] Testing endpoints...")
    
    # Test all endpoints
    results = {}
    for name, url in TEST_ENDPOINTS.items():
        print(f"  Testing {name}...", end=" ", flush=True)
        result = test_endpoint(name, url)
        results[name] = result
        
        if result.success:
            print(f"✓ {result.latency_ms}ms")
        else:
            print(f"✗ {result.error_type}: {result.error[:50]}...")
    
    # Classify
    classification, recommendation = classify_network(results)
    
    # Build profile
    profile = NetworkProfile(
        timestamp=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(),
        proxy_vars=check_proxy_settings(),
        endpoints=results,
        classification=classification,
        recommendation=recommendation,
    )
    
    # Save if path provided
    if save_path:
        profile.save(save_path)
        print(f"\n[Diagnostics] Saved to: {save_path}")
    
    return profile


def print_network_diagnostics():
    """Print network diagnostic info to console with full testing."""
    profile = run_full_diagnostics()
    
    # Print summary
    print("\n" + "=" * 60)
    print("NETWORK CLASSIFICATION")
    print("=" * 60)
    print(f"\nStatus: {profile.classification}")
    print(f"\nRecommendation:\n{profile.recommendation}")
    
    if profile.proxy_vars:
        print(f"\n⚠️  Proxy variables detected: {list(profile.proxy_vars.keys())}")
    
    print("=" * 60)
    
    return profile


def get_network_diagnostics() -> Dict:
    """Get basic network diagnostic info (legacy function)."""
    return {
        "hostname": socket.gethostname(),
        "proxy_vars": check_proxy_settings(),
    }
