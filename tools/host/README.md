# Optional host tools

`host_info.sh` reads host/model state. `install_memory_policy.sh` is the original,
explicit privileged installer for `90-qwen-memory.conf`. Neither is called by
profile discovery or serving. No host policy is installed by this layout release.

Host/network/power/storage changes must follow the user's host agreements and live
state checks. Do not run the installer merely to adopt a new repository layout.
The real-quant temporary wrapper remains under `quant/with_quant_swappiness.sh`
to preserve the existing lock and restore behavior.
