#!/usr/bin/env python3
"""
Cross-platform interactive cluster shell via Paramiko – *refactored*.

Key improvements
----------------
• Dynamic PTY resize on SIGWINCH        (fixes garbled nano / readline)  
• TCP_NODELAY + Transport keep-alive    (snappier keystrokes, no stalls)  
• recv_ready-guarded reads              (no half-baked escape sequences)  
• Windows extended-key → ANSI mapping   (history & navigation on cmd/PS)  
• Still forwards Ctrl-C / Ctrl-D / Ctrl-Z cleanly
"""

from __future__      import annotations

try:
    import fcntl
except ModuleNotFoundError:
    import winfcntl as fcntl

import getpass
import os
import platform
import selectors
import signal
import shutil
import socket
import struct
import sys
from types           import ModuleType

import paramiko                          # SSH library
from dotenv           import load_dotenv # optional .env support

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
load_dotenv()
HOST = os.getenv("HOST")
USER = os.getenv("USER")
PASS = os.getenv("PASS")
PORT = int(os.getenv("PORT", 22))
KEEPALIVE = int(os.getenv("KEEPALIVE", 30))           # seconds

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _winsize() -> tuple[int, int]:
    """Return (cols, rows) of local TTY – falls back to 80×24."""
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.columns, size.lines

def _set_nodelay(chan: paramiko.Channel) -> None:
    """Disable Nagle so interactive keystrokes are not buffered."""
    transport = chan.get_transport()
    if transport and transport.sock:
        transport.sock.setsockopt(socket.IPPROTO_TCP,
                                  socket.TCP_NODELAY, 1)

# --------------------------------------------------------------------------- #
# Global channel pointer for signal handlers                                  #
# --------------------------------------------------------------------------- #
_global_chan: paramiko.Channel | None = None

def _sigint_handler(signum, frame):          # Ctrl-C → remote
    if _global_chan and not _global_chan.closed:
        try:
            _global_chan.send(b"\x03")       # ETX
        except Exception:
            pass                             # channel already gone

def _sigwinch_handler(signum, frame):        # Window resize
    if _global_chan and not _global_chan.closed:
        cols, rows = _winsize()
        try:
            _global_chan.resize_pty(width=cols, height=rows)
        except Exception:
            pass

signal.signal(signal.SIGINT,   _sigint_handler)
if platform.system() != "Windows":
    signal.signal(signal.SIGWINCH, _sigwinch_handler)

# --------------------------------------------------------------------------- #
# POSIX I/O loop                                                              #
# --------------------------------------------------------------------------- #
def _posix_shell(chan: paramiko.Channel) -> None:
    _set_nodelay(chan)
    sel = selectors.DefaultSelector()
    sel.register(chan, selectors.EVENT_READ)
    sel.register(sys.stdin, selectors.EVENT_READ)

    # raw mode for local TTY
    try:
        import termios, tty                       # noqa: E401
        raw_ok = sys.stdin.isatty()
    except ModuleNotFoundError:                   # on Windows / pipeless
        termios = tty = None                      # type: ignore
        raw_ok = False

    if raw_ok:
        orig = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin, termios.TCSANOW)
    else:
        orig = None

    try:
        while True:
            for key, _ in sel.select():
                if key.fileobj is chan:           # ← server → local
                    while chan.recv_ready():
                        data = chan.recv(32768)
                        if not data:
                            return
                        sys.stdout.buffer.write(data)
                    sys.stdout.flush()

                else:                             # ← local → server
                    if raw_ok:
                        data = os.read(sys.stdin.fileno(), 1024)
                    else:
                        data = sys.stdin.readline().encode()

                    if not data:                  # EOF
                        chan.send(b"\x04")
                        return
                    if data in (b"\x04", b"\x1a"):
                        chan.send(b"\x04")
                        return
                    chan.send(data)
    finally:
        if raw_ok and orig:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig)

# --------------------------------------------------------------------------- #
# Windows I/O loop (maps scan-codes to ANSI)                                  #
# --------------------------------------------------------------------------- #
_EXTENDED_MAP = {
    "H": "\x1b[A",   # Up
    "P": "\x1b[B",   # Down
    "K": "\x1b[D",   # Left
    "M": "\x1b[C",   # Right
    "G": "\x1b[H",   # Home
    "O": "\x1b[F",   # End
    "I": "\x1b[5~",  # PgUp
    "Q": "\x1b[6~",  # PgDn
    "S": "\x1b[3~",  # Del
    "R": "\x1b[2~",  # Ins
    ";": "\x1bOP",   # F1-F4
    "<": "\x1bOQ",
    "=": "\x1bOR",
    ">": "\x1bOS",
}

def _windows_shell(chan: paramiko.Channel) -> None:
    import msvcrt                              # Windows console helpers

    _set_nodelay(chan)
    chan.settimeout(0.0)

    while True:
        if chan.recv_ready():
            sys.stdout.buffer.write(chan.recv(32768))
            sys.stdout.flush()

        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):          # Extended key
                ext = msvcrt.getwch()
                if seq := _EXTENDED_MAP.get(ext):
                    chan.send(seq.encode())
                continue

            if ch == "\x03":                    # Ctrl-C
                chan.send(b"\x03")
                continue
            if ch == "\x1a":                    # Ctrl-Z → EOF
                chan.send(b"\x04")
                return
            chan.send(ch.encode())

        if chan.closed or chan.exit_status_ready():
            return

# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    if not HOST or not USER:
        sys.exit("Please set HOST and USER in environment or .env file.")

    password = PASS or getpass.getpass(f"{USER}@{HOST}'s password: ")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, port=PORT, username=USER, password=password)
    except paramiko.SSHException as exc:
        sys.exit(f"SSH failed: {exc}")

    # Enable SSH keep-alives
    client.get_transport().set_keepalive(KEEPALIVE)

    cols, rows = _winsize()
    chan = client.invoke_shell(term=os.getenv("TERM", "xterm-256color"),
                               width=cols, height=rows)
    chan.set_combine_stderr(True)
    chan.settimeout(0.0)

    global _global_chan
    _global_chan = chan

    try:
        if platform.system() == "Windows":
            _windows_shell(chan)
        else:
            _posix_shell(chan)
    finally:
        try:
            chan.close()
        finally:
            client.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Local interrupt – exiting]", file=sys.stderr)
