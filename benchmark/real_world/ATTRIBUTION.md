# Data and third-party attribution

The benchmark driver code follows the paper repository's MIT license (`licenses/benchmark-MIT.txt`). Dataset-derived material and third-party source retain their own terms; the repository license does not relicense them.

## MOTChallenge material

Data/cache references and example frames derive from the supplied recoded MOT16-04 and MOT20-03 excerpts. Original material is credited to the MOTChallenge authors and contributors. Official source/terms, checked 27 October 2026:

- [MOT16 official archive and license](https://motchallenge.net/data/MOT16/).
- [MOT20 official archive and license](https://motchallenge.net/data/MOT20/).
- [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-nc-sa/3.0/) and [legal code](https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode).

Both official dataset pages state CC BY-NC-SA 3.0. Dataset-derived content under `data/` is distributed under those same terms: retain attribution/license links, indicate modifications, use noncommercially, and share adaptations alike. Original contributors retain copyright; no endorsement is implied. The website notes that conditions of original third-party sequence sources can additionally apply.

Modifications: supplied 999-frame MP4 excerpts at 24 fps; supplied converted five-column GT; cached RF-DETR Nano detections at 768 px / confidence 0.2; JSON gzip compression; diagnostic box/point/text overlays and panel resizing in annotated previews. Clean native PNGs are unchanged decoded pixels; no AI generation, enhancement or color editing was applied. Videos and checkpoint weights are not included. Original class/visibility/ignore metadata and exact original conversion commands were not supplied, so this package makes no official filtering/evaluation claim.

Please cite the dataset papers:

Anton Milan, Laura Leal-Taixé, Ian Reid, Stefan Roth, Konrad Schindler. **MOT16: A Benchmark for Multi-Object Tracking** (2016). [arXiv:1603.00831](https://arxiv.org/abs/1603.00831).

Patrick Dendorfer, Hamid Rezatofighi, Anton Milan, Javen Shi, Daniel Cremers, Ian Reid, Stefan Roth, Konrad Schindler, Laura Leal-Taixé. **MOT20: A benchmark for multi object tracking in crowded scenes** (2020). [arXiv:2003.09003](https://arxiv.org/abs/2003.09003).

## pyppbox and SORT

The optional tracker dependency is [pyppbox](https://github.com/rathaumons/pyppbox), version 4.0.0, source commit `f3f1e8e671d7cad7e0df2f2469ae0f5ee6c27258`, by Ratha SIV / UMONS-Numediart. Its source is AGPL-3.0-or-later; see `licenses/pyppbox-AGPL-3.0.txt` and the upstream notices copied verbatim to `licenses/pyppbox-NOTICE.txt`. The optional `patches/pyppbox-imgsz.patch` changes its configuration/adapter to accept input resolution before optimization and follows those upstream source terms. It does not change Centroid or SORT tracking logic. Corresponding measured source snapshots and the full recorded patch are preserved in the collection provenance, with original headers.

The bundled upstream SORT implementation is by Alex Bewley, [SORT repository](https://github.com/abewley/sort), GPL-3.0-or-later as stated in its source. Its license is copied to `licenses/SORT-GPL-3.0.txt`. No full pyppbox checkout is vendored in this portable package.

RF-DETR is an optional external dependency from [Roboflow](https://github.com/roboflow/rf-detr); no model code or checkpoint is redistributed here. Solver and numerical dependencies retain their upstream licenses. Their measured versions and binary identities are in the archived environment records.
