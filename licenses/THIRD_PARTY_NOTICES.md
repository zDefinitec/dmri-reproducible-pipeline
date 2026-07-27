# Third-party notices

This file is a guide, not a replacement for the original notices. Exact
per-file hashes, versions, evidence limits, modifications, and origins are in
`THIRD_PARTY_ATTRIBUTION.json`.

- Henrique direct-DKI helper: downloaded `main` snapshot from
  `RafaelNH/CamCAN-dMRI-study`, unchanged, under CC BY 4.0. No commit is
  asserted because the source snapshot contains no Git or archive metadata.
- NODDI v1.05 retained source: upstream notice plus complete Artistic License
  2.0. The retained subset is modified: `CreateROI.m` is a compatibility
  rewrite and `NODDI_erfi.m` is a package-authored mathematical replacement.
  The replacement is not silently claimed under the upstream Artistic
  licence, and the historical unlicensed File Exchange bytes are absent.
- MATLAB NIfTI runtime: 72 files are byte-identical to official SPM12 r7487
  commit `50b4fd3bf376f062965b5e1b20ab29af4aa1e6f3`; ten legacy files and the
  distribution notices are covered by the bundled NIfTI MATLAB notice. Both
  origins state GPL 2.0 or later. The unused, unlicensed `make.m` and all
  precompiled MEX files are excluded.
- Historical JHU atlas: image/XML bytes from FSL `data_atlases` tag
  `fsl-5_0_4`; package filenames and provenance metadata are documented. The
  current official FSL licence places JHU atlases under the main
  non-commercial terms at their owners' request.

Original and complete texts:

- `HENRIQUE-CC-BY-4.0.txt`
- `NODDI-LICENSE.txt`
- `ARTISTIC-2.0.txt`
- `NIFTI-MATLAB-LICENSE.txt`
- `GPL-2.0.txt`
- `SPM12-r7487-LICENCE.txt`
- `FSL-JHU-LICENSE.md`

There is no project-wide licence grant in this distribution.
