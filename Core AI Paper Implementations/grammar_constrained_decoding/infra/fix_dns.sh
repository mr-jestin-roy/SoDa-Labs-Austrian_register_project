#!/bin/bash
# fix_dns.sh — run ON the A100 host (sudo needed).
# The instance's DHCP DNS (Monash internal 130.194.1.99 / 130.194.7.99) does not
# answer from Nectar, so nothing resolves. This adds public resolvers globally via
# systemd-resolved and keeps the Monash ones as fallback.
#
# Undo:  sudo rm /etc/systemd/resolved.conf.d/10-public-dns.conf
#        [ -e /etc/resolv.conf.bak-fixdns ] && sudo mv /etc/resolv.conf.bak-fixdns /etc/resolv.conf
#        sudo systemctl restart systemd-resolved
set -euo pipefail

check() { timeout 6 getent hosts "$1" >/dev/null 2>&1; }

if check pypi.org && check huggingface.co; then
  echo "DNS already works — nothing to do."; exit 0
fi

if systemctl is-active --quiet systemd-resolved; then
  sudo mkdir -p /etc/systemd/resolved.conf.d
  sudo tee /etc/systemd/resolved.conf.d/10-public-dns.conf >/dev/null <<'EOF'
[Resolve]
# Nectar blocks outbound UDP/53 but allows TCP/853, so use DNS-over-TLS.
DNS=1.1.1.1#cloudflare-dns.com 8.8.8.8#dns.google
DNSOverTLS=yes
FallbackDNS=130.194.1.99 130.194.7.99
Domains=~.
EOF
  # glibc must ask the local stub so the global servers are used.
  if [ "$(readlink -f /etc/resolv.conf)" != "/run/systemd/resolve/stub-resolv.conf" ]; then
    sudo cp -P /etc/resolv.conf /etc/resolv.conf.bak-fixdns
    sudo ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
  fi
  sudo systemctl restart systemd-resolved
else
  # No systemd-resolved: write resolv.conf directly (backup kept).
  sudo cp -P /etc/resolv.conf /etc/resolv.conf.bak-fixdns
  printf 'nameserver 1.1.1.1\nnameserver 8.8.8.8\nnameserver 130.194.1.99\n' | sudo tee /etc/resolv.conf >/dev/null
fi

for h in pypi.org huggingface.co files.pythonhosted.org astral.sh; do
  if check "$h"; then echo "ok   $h"; else echo "FAIL $h"; FAILED=1; fi
done
if [ "${FAILED:-0}" = 1 ]; then
  echo "DNS still failing — outbound UDP/TCP 53 may be blocked by the security group; check egress rules."
  exit 1
fi
echo "DNS fixed."
