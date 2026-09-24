# Hugging Face Spaces deployment preparation

Create a public Space with **Docker** SDK and CPU Basic hardware. The root README YAML sets `sdk: docker` and `app_port: 7860`; the Dockerfile listens on that port, uses one Uvicorn worker and pre-downloads only the pinned model snapshot during build. Runtime is offline, so a missing snapshot or changed revision fails visibly. The cache lives under `/home/user/.cache/huggingface` in the image. It is part of the built image and is not persistent writable storage; rebuilding redownloads it. No Hugging Face token or other secret is needed for this public model. If a token later becomes necessary for an approved model, add it as a Space Secret, never in the repository or browser.

The example configuration is in `.env.example`. Set `TOKEN_TRAIL_ALLOWED_ORIGINS` to exact site origins, without paths. CORS controls browsers only; it is not authentication. `TOKEN_TRAIL_TIMEOUT_SECONDS` defaults to 30 and `TOKEN_TRAIL_RATE_LIMIT_PER_MINUTE` to 12. Model identity is fixed in source, not an environment setting. Before using the public site, test an OPTIONS preflight and POST from both `https://crunchycodes.net` and its intended GitHub Pages origin.

For GitHub-driven deployment, create a separate GitHub repository for this project and a Space repository under the Hugging Face account. Push the reviewed main branch to the Space's Git remote, or configure a GitHub Actions mirror with a Hugging Face write token stored only as a GitHub secret. Do not use a browser token. Space rebuilds from its repository's Dockerfile. The exact Space URL and remote are intentionally unset until the account owner creates them. No deployment has been performed.

CPU Basic currently offers 2 vCPU and 16 GB RAM with no hourly hardware charge, but compute Space creation requires a paid plan; this account is Pro. Free hardware sleeps after inactivity, causing a wake and possible cold start. CPU Upgrade offers 8 vCPU and 32 GB at a paid hourly rate and usually stays running unless sleep is configured. Check current prices before changing hardware. ZeroGPU is currently Gradio-only and cannot run this Docker/FastAPI design directly; using it would require a different integration and is not the initial target. A dedicated GPU is unnecessary until CPU measurements show a clear need.

After deployment, record build success, `/health` readiness, cold and warm response times, one fixed-prompt benchmark, memory use, CORS behaviour, rate-limit behaviour, and a model-failure test. Do not claim the Space works until those checks pass on the actual Space.

Current platform references: [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker), [Space configuration](https://huggingface.co/docs/hub/spaces-config-reference), [CPU hardware and sleep](https://huggingface.co/docs/hub/spaces-gpus), and [ZeroGPU compatibility](https://huggingface.co/docs/hub/spaces-zerogpu). Hosting rules and prices can change; check these before deployment.

## Local container check

On 24 September 2026, the image built successfully on a local arm64 Docker host (about 1.81 GB). A running container returned `200` from `/health` with `ready: true` and `200` from a three-token `/v1/traces` request with `probability_basis: raw_softmax_logits`. This checks the Docker packaging on arm64; the Hugging Face Linux amd64 build and Space runtime remain untested.
