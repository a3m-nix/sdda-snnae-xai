# Data

The Edge-IIoTset and CICIoT2023 datasets are publicly available from their original sources and are not redistributed
here.

1. **PCAP to features.** The bidirectional packet features are extracted from the original PCAP files with TShark and
   sampled with the scripts in `scripts/data` of the companion repository https://github.com/a3m-nix/snn-ae-ddos-energy.
2. **Expected input.** The notebook `src/pipeline/3-4-paper3_xai_v4_3x10_final_multicell_v2.ipynb` reads
   `data/dataset-from-pcap/ciciot2023_bidir.parquet` and `data/dataset-from-pcap/edge-iiotset_bidir.parquet`
   (paths relative to the repository root).
3. **Balanced samples.** From these files the notebook builds the artifact-aware 33-feature balanced samples
   `results/paper3-faithfulness-v4-3x10/data/ciciot2023_balanced.npz` and `edge_iiotset_balanced.npz`. The feature lists are in
   `results/paper3-faithfulness-v4-3x10/metadata/`. To verify a rebuild, compare with the SHA-256 of the files used in the article:

| File | SHA-256 |
|---|---|
| `ciciot2023_balanced.npz` | `7413c9be5ff6de4244e751a540061ccccc9092a6ca6bf70971342e886f401021` |
| `edge_iiotset_balanced.npz` | `3f5f9781e3b01cf27f80ff4d9cbe2e0c3e6ad5f9873bef16e0c075caf17f0d3e` |
