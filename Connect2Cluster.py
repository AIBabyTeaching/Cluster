#!/usr/bin/env python3
"""
Cross-platform interactive cluster shell via Paramiko.

Key features
------------
• Works unchanged on Linux, macOS *and* Windows – no unconditional termios import.
• Ctrl+C is forwarded (ASCII ETX 0x03) to the remote host via a SIGINT handler.
• Ctrl+D (POSIX) or Ctrl+Z (Windows) sends EOF to the remote side and closes.
• Disables TCP Nagle for snappier keystrokes.
• Only standard library + Paramiko + python-dotenv required.

Tested with Python ≥ 3.9 and Paramiko ≥ 3.4.
"""

from __future__ import annotations

import os
import sys
import socket
import selectors
import platform
import getpass
import signal
import paramiko
from types import ModuleType
from dotenv import load_dotenv

load_dotenv()

HOST  = os.getenv("HOST")
USER  = os.getenv("USER")
PASS  = os.getenv("PASS")          # optional – prompt if absent
PORT  = int(os.getenv("PORT", 22)) # default 22

# -----------------------------------------------------------------------------#
# Helper: turn off Nagle for lower latency                                     #
# -----------------------------------------------------------------------------#
def _set_nodelay(chan: paramiko.Channel) -> None:
    transport = chan.get_transport()
    if transport and transport.sock:
        transport.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

# -----------------------------------------------------------------------------#
# SIGINT => forward ^C to cluster, don't kill client                            #
# -----------------------------------------------------------------------------#
_global_chan: paramiko.Channel | None = None
def _sigint_handler(signum, frame):  # noqa: D401, unused-arg
    if _global_chan and not _global_chan.closed:
        try:
            _global_chan.send(b"\x03")  # ASCII ETX
        except Exception:
            pass  # ignore network errors

# Use the same handler on Windows and POSIX
signal.signal(signal.SIGINT, _sigint_handler)

# -----------------------------------------------------------------------------#
# POSIX shell – termios only imported if it actually exists                     #
# -----------------------------------------------------------------------------#
def _posix_shell(chan: paramiko.Channel) -> None:
    """
    Interactive loop for Linux / macOS.  Falls back gracefully if termios
    is unavailable, but raw mode (single-keystroke) is used when possible.
    """
    _set_nodelay(chan)
    sel = selectors.DefaultSelector()
    sel.register(chan, selectors.EVENT_READ)
    sel.register(sys.stdin, selectors.EVENT_READ)

    # Try to enter raw mode so we get immediate keystrokes; skip if not possible.
    termios: ModuleType | None
    tty:      ModuleType | None
    try:
        import termios as _termios
        import tty as _tty
        termios, tty = _termios, _tty
    except ModuleNotFoundError:
        termios = tty = None

    if termios and tty and sys.stdin.isatty():
        orig_tty = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin)
    else:
        orig_tty = None  # type: ignore[assignment]

    try:
        while True:
            for key, _ in sel.select():
                if key.fileobj is chan:
                    try:
                        data = chan.recv(32768)
                    except socket.timeout:
                        continue
                    if not data:
                        return  # remote side closed
                    sys.stdout.buffer.write(data)
                    sys.stdout.flush()

                elif key.fileobj is sys.stdin:
                    # In raw mode we can safely read small chunks; otherwise readline.
                    if termios and tty and sys.stdin.isatty():
                        data = os.read(sys.stdin.fileno(), 1024)
                    else:
                        data = sys.stdin.readline().encode()

                    if not data:
                        chan.send(b"\x04")  # EOF (Ctrl+D)
                        return
                    if data in (b"\x04", b"\x1a"):  # explicit EOF shortcut
                        chan.send(b"\x04")
                        return
                    chan.send(data)
    finally:
        if orig_tty is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_tty)

# -----------------------------------------------------------------------------#
# Windows shell                                                                 #
# -----------------------------------------------------------------------------#
def _windows_shell(chan: paramiko.Channel) -> None:
    import msvcrt  # Windows-only
    _set_nodelay(chan)
    chan.settimeout(0.0)

    while True:
        if chan.recv_ready():
            sys.stdout.buffer.write(chan.recv(32768))
            sys.stdout.flush()

        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            # Handle Ctrl+C locally *only* if SIGINT hasn't caught it yet
            if ch == "\x03":          # Ctrl+C
                chan.send(b"\x03")
                continue
            if ch == "\x1a":          # Ctrl+Z => EOF
                chan.send(b"\x04")
                return
            chan.send(ch.encode())

        if chan.closed or chan.exit_status_ready():
            return

# -----------------------------------------------------------------------------#
# Main                                                                          #
# -----------------------------------------------------------------------------#
def main() -> None:
    if not HOST or not USER:
        sys.exit("Please set HOST and USER in your environment or .env file.")

    password = PASS or getpass.getpass(f"{USER}@{HOST}'s password: ")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(HOST, port=PORT, username=USER, password=password)
    except paramiko.SSHException as exc:
        sys.exit(f"SSH connection failed: {exc}")

    chan = ssh.invoke_shell(width=120, height=30)
    chan.set_combine_stderr(True)
    chan.settimeout(0.0)

    global _global_chan   # noqa: PLW0603
    _global_chan = chan   # so SIGINT handler can reach it

    try:
        if platform.system() == "Windows":
            _windows_shell(chan)
        else:
            _posix_shell(chan)
    finally:
        try:
            chan.close()
        finally:
            ssh.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Normally SIGINT is swallowed, but catch any stray ones.
        print("\n[Local interrupt – exiting]", file=sys.stderr)
