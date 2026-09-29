from __future__ import annotations
import json 
from dataclasses import asdict, dataclass 
from datetime import datetime, timezone 
from pathlib import Path 
from typing import Callable 
import pandas as pd 

@dataclass
class PacketSummary:
    timestamp_utc: str
    src_ip: str | None
    dst_ip: str | None
    src_port: int | None
    dst_port: int | None
    protocol: str
    packet_length: int

class CsvFlowCollector:
    def read(self, path: str | Path) -> pd.DataFrame:
        return pd.read_csv(path)

class ScapyPacketCollector:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def capture(self, packet_count:int = 100, iface: str | None = None, bpf_filter: str | None = None) -> int:
        try:
            from scapy.all import IP, TCP, UDP, sniff 
        except ImportError as exc:
            raise RuntimeError("Scapy is not installed.") from exc 

        def summarize(packet):
            src_ip = dst_ip = None
            src_port = dst_port = None 
            protocol = "OTHER"
            if IP in packet:
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                protocol = str(packet[IP].proto)
            if TCP in packet:
                protocol = 'TCP'
                src_port = int(packet[TCP].sport)
                dst_port = int(packet[TCP].dport)
            elif UDP in packet:
                protocol = 'UDP'
                src_port = int(packet[UDP].sport)
                dst_port = int(packet[UDP].dport)

            record = PacketSummary(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                packet_length=len(packet)
            )
            with self.output_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(asdict(record)) + '\n')

        sniff(count=packet_count, iface=iface, filter=bpf_filter, prn=summarize, store=False)
        return packet_count 
    