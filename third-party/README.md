# Third-party code

`bash third-party/setup.sh {model|corpus|extraction|all}` clones each project below at the pinned commit into this
folder, applies our patch where there is one (`patches/`) and builds it. Nothing here is vendored; every project keeps
its own license.

| Folder | Project and commit | Used for | Setup section | License | Our changes |
|---|---|---|---|---|---|
| `LitePT/` (+ `litept`, `libs` links) | [prs-eth/LitePT](https://github.com/prs-eth/LitePT) `f0cc769` | Scene encoder backbone of the model | `model` | MIT | none |
| `mano/` | [MANO](https://mano.is.tue.mpg.de) model files, downloaded by you | Hand model everywhere (`MANO_RIGHT.pkl`, `MANO_LEFT.pkl`) | — | MANO license | — |
| (pip) | [manopth](https://github.com/hassony2/manopth) `4f1dcad`, [chumpy](https://github.com/mattloper/chumpy) `580566e` | MANO layer; reading the MANO pickles | `pyproject.toml` | GPL-3.0; MIT | none |
| `uni3d/` | [baaivision/Uni3D](https://github.com/baaivision/Uni3D) `64e03c3` + `BAAI/Uni3D` uni3d-g weights | Object-shape embeddings of a corpus | `corpus` | MIT | none |
| `mega-sam/` | [mega-sam/mega-sam](https://github.com/mega-sam/mega-sam) `a27b4e6` with submodules | Camera motion in the object-pose step | `extraction` | Apache-2.0; DROID-SLAM and lietorch BSD-3 | `megasam.patch`: no torch_scatter, current PyTorch tensor-type calls, sm_90 build targets, Eigen vectorization off in lietorch's CPU code |
| `sam-3d-objects/` | [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) `81a8237` + gated weights | Object mesh and keyframe poses | `extraction` | SAM License | `sam3d.patch`: dtype guards for bf16 decoding, MoGe depth, mask and intrinsics returned, preset attention backend, `CUDA_HOME` outside conda; the layout post-optimization runs its manual alignment when the initial IoU is zero and keeps the unoptimized pose unless the IoU improves |
| `kaolin/` | [NVIDIAGameWorks/kaolin](https://github.com/NVIDIAGameWorks/kaolin) `9921a79` | Mesh distances in the optimizer and relation features | `extraction` | Apache-2.0 | none (built for your GPU architectures) |
| `wilor/` | [rolpotamias/WiLoR](https://github.com/rolpotamias/WiLoR) `fcb9113`, own environment `wilor/.venv` | Hand detection and reconstruction | `extraction` | CC-BY-NC-ND-4.0 | none |

Models downloaded on first use from Hugging Face or PyTorch Hub: Molmo2-8B (`allenai/Molmo2-8B`, served with vLLM),
SAM 3, MoGe-2 / MetricAnything (`yjh001/metricanything_student_pointmap`), CoTracker3 and DINOv3
(`facebook/dinov3-vitl16-pretrain-lvd1689m`).

Notes:

- **WiLoR** runs in its own Python 3.11 environment (`wilor/.venv`, created by the setup script) and reads a copy of
  `MANO_RIGHT.pkl` without chumpy objects (`wilor/WiLoR/mano_data/MANO_RIGHT.pkl`, written by the setup script).
  `load_wilor()` uses paths relative to `wilor/WiLoR`; our runner changes into that folder.
- **SAM 3D Objects** uses the MetricAnything student as its depth model: the setup script writes that model id into
  `sam-3d-objects/checkpoints/hf/pipeline.yaml`.
- **Kaolin** is built from source for the architectures in `TORCH_CUDA_ARCH_LIST` (default `8.0;9.0;10.0`). On GPUs
  with compute capability 10 or higher the model computes fingertip-to-surface distances with libigl on the CPU.
- **LitePT** runs its pointrope operator in PyTorch when the CUDA extension is not built (`libs/pointrope/setup.py`
  builds it).
- **The released model** needs only the `model` section; LitePT's own ScanNet weights are needed only to train from
  scratch (`model.litept_weights_path`, default `checkpoints/litept_scannet/model_best.pth`, from the LitePT release).
- **Robot retargeting** clones SPIDER separately: `robotics/third_party.sh`.
