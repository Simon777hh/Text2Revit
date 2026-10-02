# Third-party notices

The project's evaluation license applies only to the author's original material.
It does not replace any third-party license.

* `resplan_utils.py` is derived from ResPlan's MIT-licensed helper code, copyright
  2025 The ResPlan Authors. The full grant is in `third_party/ResPlan-MIT.txt`.
  [Upstream license](https://github.com/m-agour/ResPlan/blob/main/LICENSE).
* ResPlan data has a separate CC BY 4.0 grant covering the upstream authors'
  contributions. Raw/prepared datasets are not distributed in this repository.
  Consult that grant and the upstream takedown information when obtaining data.
* RPLAN datasets and any optional external toolbox must be obtained separately
  under their own terms. This repository grants no rights to those assets.
* The offline installer includes the public OpenAI CLIP text encoder. OpenAI's
  CLIP software is MIT licensed, copyright 2021 OpenAI; see
  `third_party/CLIP-MIT.txt` and the
  [upstream license](https://github.com/openai/CLIP/blob/main/LICENSE).
  Obtain the encoder from `openai/clip-vit-large-patch14` and consult its model card.
* Python/Conda, PyTorch, NumPy, Transformers, Shapely, Matplotlib and their
  transitive dependencies retain their original licenses and notices. The packed
  runtime includes installed package license metadata. Their presence does not
  make them subject to the Text2Revit evaluation license.
* Autodesk Revit is a separately licensed prerequisite. API reference assemblies
  are build dependencies and are not copied into the add-in payload.

Archived HouseDiffusion/DPFM experiments and their licenses remain outside this
repository. They are not part of the current training or packaged backend.
