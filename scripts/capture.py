#!/usr/bin/env python3

from __future__ import annotations
import argparse
from pathlib import Path 
import sys 

ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(ROOT / "src"))

from src.edge_ids.traffic import ScapyPacketCollector 

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/packet_metadata.jsonl')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--iface', default=None)
    parser.add_argument('--bpf', default=None)
    args = parser.parse_args()

    collector = ScapyPacketCollector(ROOT / args.output)
    count = collector.capture(args.count, iface=args.iface, bpf_filter=args.bpf)
    print(f'Captured {count} packet summaries without storing payloads')

if __name__ == '__main__':
    main()