# Environment

All experiments ran in a Proxmox container on an Intel Xeon Silver 4416+ server (30 vCPUs and 32 GB RAM allocated),
with TensorFlow on CPU, 4 intra-op and 4 inter-op threads, and oneDNN optimizations disabled for determinism.

Main versions: Python 3.10.12, NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.1, scikit-learn 1.5.2, TensorFlow 2.12.0.
The full package list of the environment is in `requirements-freeze.txt`.
