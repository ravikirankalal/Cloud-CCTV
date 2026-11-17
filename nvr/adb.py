#!/usr/bin/env python3
# Shared connection to the Oracle 23ai Autonomous DB (the person-detection vector store).
# oracledb THIN mode (pure python, no Instant Client). Credentials come from files in the wallet dir
# (chmod 600), never hard-coded. Single source of the connection config for adb_setup.py and the poller.
import os
import oracledb

WDIR = os.environ.get("ADB_WALLET", "/home/ubuntu/nvr/adb-wallet")
DSN = os.environ.get("ADB_DSN", "nvrdetect_tp")


def connect():
    with open(os.path.join(WDIR, "admin.pw")) as f:
        pw = f.read().strip()
    with open(os.path.join(WDIR, "wallet.pw")) as f:
        wpw = f.read().strip()
    return oracledb.connect(user="ADMIN", password=pw, dsn=DSN,
                            config_dir=WDIR, wallet_location=WDIR, wallet_password=wpw)
