# Reserved singing references

The [reservation manifest](../tests/benchmarks/thresholds/singing_reference_reservation_v1.json)
records eight NUS-48E singer/song pairs and two accompaniment recordings before
the v2 development experiments. Twenty files were downloaded and SHA-256 hashed
without decoding audio, parsing phone annotations, or running candidate models.
Raw recordings and annotations stay outside Git. These are reserved inputs,
not evaluation results.

| Singer | Global NUS song number |
|---|---|
| MCUR | 10 |
| MPOL | 11 |
| MPUR | 02 |
| NJAT | 15 |
| PMAR | 08 |
| SAMF | 09 |
| VKOW | 19 |
| ZHIY | 03 |

Selection used directory metadata and the published global song assignments:
lexically ordered unused singers, then the smallest unused song not already
selected. Every pair excludes the four development singers and global songs
01, 04, 05, and 07. The accompaniment reservations are the US Army Strings
performance of Brahms's Hungarian Dance No.5 and Lena Orsa's *Pistachio Ice Cream
Ragtime*. Neither recording is one of the two Kevin MacLeod recordings used in
development. Dataset/song separation is relative to the recorded development
ledger; other downstream consumption must be checked before evaluation.

[NUS-48E](https://smcnus.comp.nus.edu.sg/) is distributed for research use with
citation of Duan et al. (2013). The hashed accompaniment license texts identify
the Brahms performance as public domain and Pistachio as CC-BY-NC3.0. The
manifest includes the original download URLs, file hashes, and local paths;
paths are locations of retained assets, not portable guarantees of availability.

Training overlap remains uncertain. The pinned base CTC model card names no
fine-tuning dataset. The larger model documents speech training, but we do not
have an exhaustive recording audit. The exact RVC RMVPE checkpoint lacks a
complete training manifest, and the RMVPE paper cannot establish the provenance
of different distributed weights. Jamendo training is documented for CPJKU;
complete track-level exclusions for it and Demucs are not established. Therefore
these files are unused in the known development ledger, not verified absent
from every model's training data.

Fresh pitch references are **not yet secured**. The reviewed
[iKala record](https://zenodo.org/records/3532214) describes human F0 contours
but restricts file access. TONAS and VocalNotes also need access/terms resolved.
NUS phone boundaries cannot replace F0 annotations, and an F0-unvoiced frame
does not establish vocal absence. A usable licensed F0 partition remains a
prerequisite for fresh #21 qualification.

Before opening the reserved annotations or running inference, freeze a separate
evaluation protocol and candidate SHA, verify the hashes and consumption ledger,
and state the unresolved training-overlap limits. The current development runs
use only previously consumed cached inputs. No reserved file contributes to
model fitting, threshold selection, mapping choice, or the reported development
metrics. Study-specific windows, RSA criteria and neural fitting are downstream.
