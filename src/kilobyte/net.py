"""Tor-backed private networking for the web tools.

Privacy here is fail-closed: a caller that wants a request masked must check
``tor_available()`` first and refuse to send if it is False, rather than silently
falling back to a direct connection that would expose the real IP.

Routing is per-connection (a custom urllib opener), never a global ``socket.socket``
monkeypatch — the daemon holds other connections (llama-server, the RPC socket) that must
stay direct. DNS is resolved through Tor (``rdns``) so lookups do not leak locally either.
"""

from __future__ import annotations

import binascii
import http.client
import socket
import urllib.request

TOR_SOCKS_HOST, TOR_SOCKS_PORT = "127.0.0.1", 9050
TOR_CONTROL_HOST, TOR_CONTROL_PORT = "127.0.0.1", 9051
_COOKIE_PATH = "/var/lib/tor/control_auth_cookie"


def tor_available(timeout: float = 3.0) -> bool:
    """True if Tor's SOCKS port accepts a connection. Checked before every private request."""
    try:
        with socket.create_connection((TOR_SOCKS_HOST, TOR_SOCKS_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def _socks():
    import socks  # PySocks, imported lazily so this module loads even where it is absent
    return socks


class _SocksHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        socks = _socks()
        self.sock = socks.create_connection(
            (self.host, self.port), timeout=self.timeout,
            proxy_type=socks.SOCKS5, proxy_addr=TOR_SOCKS_HOST, proxy_port=TOR_SOCKS_PORT,
            proxy_rdns=True,
        )


class _SocksHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        socks = _socks()
        sock = socks.create_connection(
            (self.host, self.port), timeout=self.timeout,
            proxy_type=socks.SOCKS5, proxy_addr=TOR_SOCKS_HOST, proxy_port=TOR_SOCKS_PORT,
            proxy_rdns=True,
        )
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class SocksHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_SocksHTTPConnection, req)


class SocksHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_SocksHTTPSConnection, req)


def rotate_circuit() -> bool:
    """Ask Tor for a fresh circuit (a new exit IP) via the control port, cookie-authenticated.
    Returns True only if Tor acknowledged the NEWNYM signal."""
    try:
        cookie = open(_COOKIE_PATH, "rb").read()
    except OSError:
        return False
    try:
        with socket.create_connection((TOR_CONTROL_HOST, TOR_CONTROL_PORT), timeout=5) as s:
            s.sendall(b"AUTHENTICATE " + binascii.hexlify(cookie) + b"\r\n")
            if not s.recv(256).startswith(b"250"):
                return False
            s.sendall(b"SIGNAL NEWNYM\r\n")
            return s.recv(256).startswith(b"250")
    except OSError:
        return False


def exit_ip(timeout: float = 30.0) -> str | None:
    """Fetch the current Tor exit IP, or None if it cannot be determined."""
    try:
        opener = urllib.request.build_opener(SocksHTTPHandler, SocksHTTPSHandler)
        with opener.open("https://api.ipify.org", timeout=timeout) as resp:
            return resp.read(64).decode("ascii", "replace").strip()
    except Exception:
        return None
