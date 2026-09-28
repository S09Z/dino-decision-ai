# Dashboard latency

Measured 2026-09-28 with `python -m src.dashboard.loadtest` on a temporary DB: a writer adds episode rows at a fixed rate while WebSocket clients listen; delay is commit → arrival, per row and client; the server checks for new rows every *check* ms.

| Check | Clients | Rows/s | Delivered | p50 | p95 | max | REST p50 | REST p95 |
|---|---|---|---|---|---|---|---|---|
| 100 ms | 1 | 20 | 200/200 | 53 ms | 103 ms | 121 ms | 5 ms | 18 ms |
| 100 ms | 5 | 20 | 1000/1000 | 51 ms | 103 ms | 135 ms | 5 ms | 16 ms |
| 100 ms | 20 | 20 | 4000/4000 | 51 ms | 98 ms | 133 ms | 6 ms | 19 ms |
| 50 ms | 1 | 20 | 200/200 | 24 ms | 58 ms | 69 ms | 5 ms | 17 ms |
| 50 ms | 5 | 20 | 1000/1000 | 25 ms | 58 ms | 84 ms | 6 ms | 25 ms |
| 50 ms | 20 | 20 | 4000/4000 | 30 ms | 78 ms | 91 ms | 10 ms | 3764 ms |

Machine: Windows 11, 16 cores, while three DQN variants trained in real time beside it.

## Findings

- **Every row reached every client** (up to 4,000/4,000 with 20 clients).
- **100ms checks (the default, `create_app(ws_interval=0.1)`):** p50 ≈ 51ms, p95 ≈ 98–103ms,
  flat from 1 to 20 clients; REST stays fast (p95 < 20ms). The delay is almost all the
  wait for the next check (on average half the interval).
- **50ms checks** halve the latency (p50 25–30ms) but do not scale: at 20 clients GET
  /metrics/latest's p95 jumps to 3.8s, because each client polls the DB on the server's
  event loop. Kept 100ms. With many clients, one shared poller broadcasting to all of them
  would remove that cost.
- **Live check (Phase 6 exit criterion):** `dino-ai dashboard` on the real DB during round 1
  showed all three variants `training` and streamed 30 finished episodes and 41 status
  updates in 45s. (That check overlapped an accidental second copy of round 1, six games in
  all, when training slowed to 12.3–12.7 steps/s, so it does not isolate the dashboard's
  own cost. Once the copy was stopped, all three were back at 13.3 steps/s by step 45k.)
