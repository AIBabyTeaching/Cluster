#!/usr/bin/env python3
"""
Interactive cluster shell via Paramiko.
"""

import paramiko, sys, select, platform, getpass, os
from dotenv import load_dotenv
load_dotenv()

HOST = os.getenv("HOST")
USER = os.getenv("USER")

def posix_shell(chan):
    import termios, tty
    old_tty = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin)
        while True:
            r, _, _ = select.select([chan, sys.stdin], [], [])
            if chan in r:
                data = chan.recv(1024)
                if not data:
                    break
                sys.stdout.buffer.write(data); sys.stdout.flush()
            if sys.stdin in r:
                chan.send(sys.stdin.read(1))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)

def windows_shell(chan):
    import msvcrt
    while True:
        if chan.recv_ready():
            sys.stdout.buffer.write(chan.recv(1024))
            sys.stdout.flush()
        if msvcrt.kbhit():
            chan.send(msvcrt.getwch())

def main():
    password = os.getenv("PASS") or getpass.getpass("Cluster password: ")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=password)
    chan = ssh.invoke_shell(width=120, height=30)

    try:
        if platform.system() == "Windows":
            windows_shell(chan)
        else:
            posix_shell(chan)
    finally:
        chan.close()
        ssh.close()

if __name__ == "__main__":
    main()
