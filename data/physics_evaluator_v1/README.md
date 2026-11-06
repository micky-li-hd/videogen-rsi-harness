# Physics evaluator input contract

The evaluator consumes measurements extracted from every decoded frame. It does not claim to obtain these measurements from raw pixels. Each tracked object supplies a monotonically increasing `samples` array with:

- `frame`, `center`, and `visible`;
- `signed_distances`, where positive means a gap, zero means contact, and negative means penetration;
- material metrics such as `area`, `aspect_ratio`, `deformation`, `volume_proxy`, and `components`.

The trace also declares event anchors such as collision, first contact, release and stop. Rubric hard rules bind exact object/event IDs and fixed thresholds. Missing tracks, masses, distances, events or metrics produce `unknown`; they never become passes.

Supported rules cover full-frame tracking, gravity direction, acceleration range, velocity continuity, momentum, restitution, contact, penetration, hover, frictional stopping, sliding/stopping, rigid/soft/liquid boundaries, event order and event time windows.

Joint checks may bind hard-rule IDs and VLM-check IDs. Any hard or visual failure fails the check; all channels must pass for a pass; incomplete or conflicted evidence yields `unknown`. A VLM cannot override a failed numerical boundary.

PhyGenBench rubrics in `data/phygenbench/prompt-rubrics.json` are reference-assisted adaptations of OpenGVLab/PhyGenBench at revision `f8642cb796f3bcb01f0b7c1b2ec53b75d357c739`. They are not the official multi-model PhyGenEval metric and the references are expectations, not observations.
