# Vendored game

The Chrome Dino game code, run in real Chrome by `src/environment/chrome_game.py`.

- Source: https://github.com/wayou/t-rex-runner at commit `5455bfa408ec6b707c7300ff194b7390733a766d`
- The game code is extracted from Chromium (`index.js` header: Copyright 2014 The Chromium Authors, BSD-style license); the repo is BSD-3-Clause (see `LICENSE`).
- Copied unchanged: `index.html`, `index.js`, `index.css`, and the sprites under `assets/`. Unused demo GIFs were not copied.

`chrome://dino` itself cannot be automated (it is a sandboxed error page), which is why a local copy is used.
