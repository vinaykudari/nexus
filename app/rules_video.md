UI Component Animation Rules

Goal

- Given a screenshot/crop of a UI component and an instruction to animate, produce a short looping video that can replace the selection 1:1.

Canvas & Duration

- Preserve the exact canvas width, height, and aspect ratio of the input.
- Duration 2–4 s; seamless loop; 24–30 fps; no audio.
- No cropping, reframing, added borders, or padding.

Motion

- Animate only what the instruction requires: hover, press, shimmer, progress, micro‑transitions.
- Keep layout and hierarchy intact; avoid disruptive reflows.
- Easing: use standard cubic easing; natural inertia; subtle amplitudes.
- Motion should enhance clarity and accessibility, not distract.

Visual Consistency

- Maintain original style unless instructed to restyle; otherwise apply minimal, tasteful updates.
- Preserve brand elements; keep edges crisp; avoid artifacts and banding.
- Respect accessible contrast during motion.

Output Contract

- Return a single video in the first candidate with MIME type video/mp4.
- Keep the same canvas size as the input image.
- If video cannot be produced, return the first frame as a still image with the requested edits.
