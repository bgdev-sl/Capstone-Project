# Generate demo dataset

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--output", default="data/demo_flows.csv")
    args = parser.parse_args()

    X, y = make_classification(
        n_samples=args.rows,
        n_features=18,
        n_informative=10,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        weights=[0.70, 0.30],
        class_sep=1.25,
        random_state=42,
    )

    columns = [
        "Flow_Duration",
        "Packet_Length_Mean",
        "Packet_Length_Std",
        "Flow_Bytes_s",
        "Flow_Packets_s",
        "Fwd_Packets_s",
        "Bwd_Packets_s",
        "Fwd_Header_Length",
        "Bwd_Header_Length",
        "SYN_Flag_Count",
        "ACK_Flag_Count",
        "RST_Flag_Count",
        "Protocol_Type",
        "Src_Port",
        "Dst_Port",
        "Packet_Size_Max",
        "Packet_Size_Min",
        "Active_Mean",
    ]
    frame = pd.DataFrame(X, columns=columns)
    # Make the synthetic data more network-flow-like without pretending to reproduce CICIoT2023.
    frame["Protocol_Type"] = np.where(frame["Protocol_Type"] > 0, "TCP", "UDP")
    frame["Src_Port"] = np.abs(frame["Src_Port"] * 1000).astype(int) % 65535
    frame["Dst_Port"] = np.abs(frame["Dst_Port"] * 1000).astype(int) % 65535
    frame["Label"] = np.where(y == 0, "Benign", "DDoS")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Created {len(frame)} demo flow rows at {output}")

if __name__ == "__main__":
    main()