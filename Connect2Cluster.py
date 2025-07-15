#!/usr/bin/env python3
"""
Cross-platform interactive cluster shell via Paramiko.

• Arrow-key, Home/End, PageUp/Down and F-keys work on Windows consoles
  (extended codes → ANSI sequences).
• Uses the local $TERM (or xterm-256color) when allocating the remote pty.
• Ctrl+C is forwarded to the remote shell; Ctrl+D/Z sends EOF and exits.
• No unconditional termios import, so it works on Windows without issues.
"""

from __future__ import annotations

import getpass
import os
import platform
import selectors
import signal
import socket
import sys
from types import ModuleType

import paramiko
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv("HOST")
USER = os.getenv("USER")
PASS = os.getenv("PASS")
PORT = int(os.getenv("PORT", 22))

# --------------------------------------------------------------------------- #
# Utility                                                                     #
# --------------------------------------------------------------------------- #
def _set_nodelay(chan: paramiko.Channel) -> None:
    transport = chan.get_transport()
    if transport and transport.sock:
        transport.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

_global_chan: paramiko.Channel | None = None
def _sigint_handler(signum, frame):  # noqa: D401
    if _global_chan and not _global_chan.closed:
        try:
            _global_chan.send(b"\x03")  # ETX
        except Exception:
            pass

signal.signal(signal.SIGINT, _sigint_handler)

# --------------------------------------------------------------------------- #
# POSIX event loop                                                            #
# --------------------------------------------------------------------------- #
def _posix_shell(chan: paramiko.Channel) -> None:
    _set_nodelay(chan)
    sel = selectors.DefaultSelector()
    sel.register(chan, selectors.EVENT_READ)
    sel.register(sys.stdin, selectors.EVENT_READ)

    # raw-mode if available
    try:
        import termios, tty  # noqa: E401
        raw_ok = sys.stdin.isatty()
    except ModuleNotFoundError:
        termios = tty = None  # type: ignore
        raw_ok = False

    if raw_ok:
        orig = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin)
    else:
        orig = None

    try:
        while True:
            for key, _ in sel.select():
                if key.fileobj is chan:
                    try:
                        data = chan.recv(32768)
                    except socket.timeout:
                        continue
                    if not data:
                        return
                    sys.stdout.buffer.write(data)
                    sys.stdout.flush()

                else:  # stdin
                    if raw_ok:
                        data = os.read(sys.stdin.fileno(), 1024)
                    else:
                        data = sys.stdin.readline().encode()

                    if not data:
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
# Windows event loop with extended-key translation                            #
# --------------------------------------------------------------------------- #
_EXTENDED_MAP = {
    "H": "\x1b[A",   # Up
    "P": "\x1b[B",   # Down
    "K": "\x1b[D",   # Left
    "M": "\x1b[C",   # Right
    "G": "\x1b[H",   # Home
    "O": "\x1b[F",   # End
    "I": "\x1b[5~",  # PageUp
    "Q": "\x1b[6~",  # PageDown
    "S": "\x1b[3~",  # Delete
    "R": "\x1b[2~",  # Insert
    ";": "\x1bOP",   # F1
    "<": "\x1bOQ",   # F2
    "=": "\x1bOR",   # F3
    ">": "\x1bOS",   # F4
}

def _windows_shell(chan: paramiko.Channel) -> None:
    import msvcrt

    _set_nodelay(chan)
    chan.settimeout(0.0)

    while True:
        if chan.recv_ready():
            sys.stdout.buffer.write(chan.recv(32768))
            sys.stdout.flush()

        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):      # extended key prefix
                ext = msvcrt.getwch()
                seq = _EXTENDED_MAP.get(ext)
                if seq:
                    chan.send(seq.encode())
                continue

            if ch == "\x03":                # Ctrl+C
                chan.send(b"\x03")
                continue
            if ch == "\x1a":                # Ctrl+Z -> EOF
                chan.send(b"\x04")
                return
            chan.send(ch.encode())

        if chan.closed or chan.exit_status_ready():
            return

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    if not HOST or not USER:
        sys.exit("Please set HOST and USER (env/.env).")

    password = PASS or getpass.getpass(f"{USER}@{HOST}'s password: ")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, port=PORT, username=USER, password=password)
    except paramiko.SSHException as exc:
        sys.exit(f"SSH failed: {exc}")

    term_type = os.getenv("TERM", "xterm-256color")
    chan = client.invoke_shell(
        term=term_type,
        width=120,
        height=30,
    )
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
