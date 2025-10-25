#!/usr/bin/env bash
# Shared config for scripts
set -euo pipefail
: "${BASE:=http://localhost:8085}"   # your port-forward / NodePort to kubefix
: "${NS:=demo}"                      # target namespace
: "${APP:=nginx}"                    # app label
export BASE NS APP

