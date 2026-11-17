# Vendored browser libraries

Everything the web UI loads is a local file under `app/web/`; nothing is fetched from the
internet at runtime. The files below are unmodified copies taken from the npm registry
tarballs on 2026-11-16. To update one, download the new tarball, replace the files, and update
the version and checksums here.

Only permissive licences (MIT). No GPL / AGPL code.

| Library | Version | Licence | Source (npm tarball) | Tarball SHA-1 (as published by the registry) |
|---|---|---|---|---|
| three.js | 0.186.1 | MIT | https://registry.npmjs.org/three/-/three-0.186.1.tgz | `6d50f70c2c437f844179bbb56d6f5b774e1ca38a` |
| Spark (`@sparkjsdev/spark`) | 2.3.1 | MIT | https://registry.npmjs.org/@sparkjsdev/spark/-/spark-2.3.1.tgz | `05258fb134917dee00ba9c25927f6550890733cf` |
| qrcode-generator | 1.4.4 | MIT | https://registry.npmjs.org/qrcode-generator/-/qrcode-generator-1.4.4.tgz | `63f771224854759329a99048806a53ed278740e7` |

## Files

| File here | Path inside the tarball | SHA-256 |
|---|---|---|
| `three/three.module.js` | `package/build/three.module.js` | `9052042d676cb0fdc1ddfefe193053f34b7ac0513a616fdac4535d49987812ea` |
| `three/three.core.js` | `package/build/three.core.js` | `9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6` |
| `three/LICENSE` | `package/LICENSE` | `8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc` |
| `three/addons/controls/OrbitControls.js` | `package/examples/jsm/controls/OrbitControls.js` | `3d79d07ecb686b4e5d93232eedab255331c1beef711e13164eaa1f68655a5f2b` |
| `three/addons/loaders/GLTFLoader.js` | `package/examples/jsm/loaders/GLTFLoader.js` | `131c0f78c01d19368ae495caa65b3adaa10487810a36a05bb5901b769a35ac16` |
| `three/addons/utils/BufferGeometryUtils.js` | `package/examples/jsm/utils/BufferGeometryUtils.js` | `9fb63427ce6641fa14fd0baff9cc4d1b5f9c3d85fd084bf2e90e803c44ec1797` |
| `three/addons/utils/SkeletonUtils.js` | `package/examples/jsm/utils/SkeletonUtils.js` | `b1632a703206c3d830de9fcbe515696770d04b71a15ee6b50afa6d2c3298c86f` |
| `three/addons/postprocessing/Pass.js` | `package/examples/jsm/postprocessing/Pass.js` | `444b409c235ead986893c472e720da1b779a56985c7d10b279c7944b52bd61c5` |
| `spark/spark.module.js` | `package/dist/spark.module.js` | `2de375d5e489692f976abe3199c8435ecc00f97fabaf35e489eb7d368e1f6f60` |
| `spark/LICENSE` | `package/LICENSE` | `51829693e5dccd9ca1daa093991faac3aaa93238eb8fd5f5cb4130af85791d64` |
| `qrcode-generator/qrcode.js` | `package/qrcode.js` | `18ae399f81182bc9de916e9c77b195df20cc58d6f2d55a62b085a299f1bf1780` |

`app/tests/test_server.py` checks these SHA-256 values against the files on disk.

## What each one is for

- **three.js**: draws `mesh.glb` and `collision.glb` (GLTFLoader), camera control (OrbitControls).
  `BufferGeometryUtils.js` and `SkeletonUtils.js` are imports of GLTFLoader; `Pass.js` is an
  import of Spark.
- **Spark**: real-time Gaussian splat rendering on WebGL2, as a three.js object. Copyright
  World Labs Technologies, Inc. The one file contains its WebAssembly part (compiled from the
  project's own Rust code, MIT) and its worker script inline, plus a bundled copy of `fflate`
  (MIT, Copyright Arjun Barrett); it starts the worker from a blob and fetches nothing.
- **qrcode-generator**: the QR code in the "用手机访问" panel. Copyright Kazuhiko Arase; the
  MIT licence is stated in the header of `qrcode.js` (the package ships no separate licence
  file). "QR Code" is a registered trademark of DENSO WAVE INCORPORATED.

## Things to know

- The page is found through the import map in `app/web/index.html` (`three`, `three/addons/`,
  `@sparkjsdev/spark`). The server's Content-Security-Policy allows that import map by its
  hash, WebAssembly compilation (`'wasm-unsafe-eval'`) and blob workers; nothing else inline.
- Spark needs no SharedArrayBuffer and no service worker, so it runs on a plain-http LAN
  address (not a "secure context"). This was the reason to prefer it over
  `@mkkellogg/gaussian-splats-3d` 0.4.7, which by default hands a SharedArrayBuffer to its
  worker and never finishes loading on such a page (it works with
  `sharedMemoryForWorkers: false`, but has had no release since January 2025).
- The libraries contain user-agent checks of their own (Spark: default sizes for its
  level-of-detail system and for its own first-person controls; GLTFLoader: a workaround for
  image decoding in Safari and old Firefox; three.js: one framebuffer workaround for the
  Oculus browser). scan3d's code does no user-agent sniffing, turns
  Spark's level-of-detail off and passes explicit values for every Spark setting that would
  otherwise depend on the user agent (see `viewer3d.js`), and does not use Spark's controls.
- `spark.module.js` ends with a `sourceMappingURL` comment; the map file is not vendored
  (browsers ask for it only with developer tools open, and get a 404).
