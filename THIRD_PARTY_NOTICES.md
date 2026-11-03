# Third-party code provenance and license notice

## CleanRL (SAC continuous-action reference)

Project: https://github.com/vwxyzjn/cleanrl
Reference file: https://github.com/vwxyzjn/cleanrl/blob/004f8a086a892a2a180f4dd332b90d83a968aa7a/cleanrl/sac_continuous_action.py
License: MIT, with upstream notices - https://github.com/vwxyzjn/cleanrl/blob/004f8a086a892a2a180f4dd332b90d83a968aa7a/LICENSE

The complete upstream CleanRL license record, including attribution and license notices for earlier SAC implementations used by CleanRL, is preserved verbatim at [licenses/CleanRL_LICENSE.txt](licenses/CleanRL_LICENSE.txt). Those are CleanRL's upstream source acknowledgments; the reference implementation directly consulted for ELE680 was CleanRL, not necessarily each upstream project.

The ELE680 SAC actor and algorithm implementation were guided by this reference. A targeted source comparison on 26 October 2026 detected a three-line exact match in the tanh-squashed action/log-probability block of `src/networks.py` (the conceptual formula is also standard SAC); this is credited as CleanRL-derived or adapted code rather than presented as original mathematical novelty. The rest of the brief comparison found no identical three-line blocks over 90 characters in `src/sac_update.py`, `src/train.py`, `src/replay_buffer.py` or `src/evaluate.py`. This limited automated spot-check is **not** a comprehensive plagiarism certificate.

The full upstream MIT license notice is retained below for transparency and compliance when code is derived from CleanRL:

MIT License

Copyright (c) 2019 CleanRL developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Literature origins

The study additionally credits Haarnoja et al. (Soft Actor-Critic), Ba et al.
(Layer Normalization), He et al. (residual networks), Bjorck et al.
(deeper DRL with spectral normalization), and Lee et al. (SimBa) in the report's
BibTeX and in `docs/SOURCE_ATTRIBUTION.md`. Their code is not claimed to be
the exact implementation copied in this project.
