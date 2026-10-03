proxy_pass http://127.0.0.1:3200;
proxy_http_version 1.1;
proxy_set_header Host $host;
proxy_set_header X-Forwarded-Proto https;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection $conn_up;
proxy_buffering off;
proxy_read_timeout 3600s;
