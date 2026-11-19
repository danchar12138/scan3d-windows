# Third-party software

Scan3D's own code is MIT-licensed (see [LICENSE](LICENSE)). It drives other programs and
models through their command lines and loads a few browser libraries. None of the downloaded
programs or model weights is stored in this repository.

## Programs downloaded by `setup.ps1`

| Component | Version | Used for | License |
|---|---|---|---|
| [Spirula Studio](https://github.com/harry7557558/spirula-studio) | 2026.11.12 | frame extraction, masking, camera poses, geometry priors, Gaussian training | GPL-3.0 |
| [COLMAP](https://github.com/colmap/colmap) (CUDA build) | 4.2.1 | dense reconstruction, meshing, decimation, photo texturing | BSD-3-Clause |
| Python (embeddable) | 3.12.10 | runs Scan3D's code | PSF |

Scan3D starts these as separate programs and does not link to, copy or modify their code.
If you redistribute a bundle that includes them, follow their licenses (for Spirula Studio,
GPL-3.0).

## Neural network models (downloaded on demand)

Models are fetched into `tools/models` only when you turn on the feature that needs them.
Each has its own license, shown in the Models panel and in the node it belongs to. Check the
linked source before you use results commercially.

| Stage | Model | License |
|---|---|---|
| Masking (people) | SAM 3 | [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE) |
| Masking (alternative) | SAM 2.1 + Grounding DINO | Apache-2.0 |
| Masking (subject only) | BiRefNet | [MIT](https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE) |
| Normal prior | MoGe-2 | see the [model page](https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx) |
| Dense seed | RoMa v2 (uses DINOv3) | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/adc254450203739c8149213a7a69d8d905b4fcfa/LICENSE.md) and RoMa v2 MIT |
| Pose features | ALIKED + LightGlue | see the [COLMAP release](https://github.com/colmap/colmap/releases/tag/3.13.0) |

The licence you accept by turning a feature on is recorded locally in
`tools/accepted-licences.json` (not part of the repository).

## Browser libraries (in `app/web/vendor`)

Unmodified release files, all MIT. Versions, sources and checksums are in
[app/web/vendor/README.md](app/web/vendor/README.md); the license files are next to the code.

| Library | Version |
|---|---|
| three.js | 0.186.1 |
| Spark (`@sparkjsdev/spark`) | 2.3.1 |
| qrcode-generator | 1.4.4 |
