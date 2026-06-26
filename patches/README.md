# Patches

These patch files modify upstream third-party projects fetched by
`scripts/setup_stella.ps1`.

## stella_vslam

- Patch: `stella_vslam-windows.patch`
- Upstream: https://github.com/stella-cv/stella_vslam
- Commit: `8ac1be4d1fa20e4148b478d7b30788abcbb1d9fe`
- Licenses:
  - `LICENSE.stella_vslam`
  - `LICENSE.stella_vslam.original`
  - `LICENSE.stella_vslam.fork`

## stella_vslam_examples

- Patch: `stella_vslam_examples-windows.patch`
- Upstream: https://github.com/stella-cv/stella_vslam_examples
- Commit: `defc69eecc36e51cdda22885bb86954f08ad6887`
- License: `LICENSE.stella_vslam_examples`

The upstream source trees and build outputs are not distributed in this
repository. They are downloaded or built under `third_party/`, which is ignored
by git.
