# Server setup (Oracle VM)

HTTPS for the camera app and Snapdrop, on `guttedar.duckdns.org` (DuckDNS) with a Let's Encrypt certificate.

| Address | Goes to |
|---|---|
| `https://guttedar.duckdns.org` (443) | camera app (`rtmp-server` container, port 3200) |
| `https://guttedar.duckdns.org:8443` | Snapdrop (container, port 8080) |

A host nginx terminates TLS; the containers are unchanged.

1. Open ports 80, 443, 8443 in the Oracle security list and in iptables (`netfilter-persistent save`).
2. `apt install nginx certbot`, then `certbot certonly --webroot -w /var/www/html -d guttedar.duckdns.org`.
3. Copy `nginx/*.conf` to `/etc/nginx/conf.d/` and `nginx-reload.sh` to `/etc/letsencrypt/renewal-hooks/deploy/` (make it executable); `nginx -t && systemctl reload nginx`.

Port 80 must stay open for certificate renewal. Google sign-in needs the hostname under Firebase Authentication > Authorized domains.
