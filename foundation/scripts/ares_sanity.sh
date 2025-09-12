#!/usr/bin/env bash
set -euo pipefail

bin=${1:-target/debug/ares}

echo "== ARES CLI Sanity Check =="
echo "Using binary: $bin"
echo

echo "-- Basic Status --"
$bin status || true
echo

echo "-- Detailed Neuromorphic Status --"
$bin status --detailed --subsystem neuromorphic || true
echo

echo "-- Quick NLP Query: 'show me system health' --"
$bin query "show me system health" || true
echo

echo "-- Toggle Learning --"
$bin learn toggle || true
echo

echo "-- Learning Stats --"
$bin learn stats || true
echo

echo "Sanity check completed."

