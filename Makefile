.PHONY: tos \
        env-backend env-frontend envs \
        install-backend install-frontend install \
        run-backend run-frontend \
        up down logs reset \
        freeze-backend freeze-frontend freeze \
        delete-backend delete-frontend delete-envs \
        grader-load test test-report

CONDA ?= $(shell which conda)

# ─── Conda ToS ───────────────────────────────────────────────────────────────

# Run once before creating environments.
tos:
	$(CONDA) tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
	$(CONDA) tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ─── Environment setup ───────────────────────────────────────────────────────

env-backend:
	$(CONDA) create -n creeping-crawler-backend python=3.11 -y
	$(CONDA) run -n creeping-crawler-backend pip install -r backend/requirements.txt
	$(CONDA) run -n creeping-crawler-backend python -m playwright install --with-deps chromium

env-frontend:
	$(CONDA) create -n creeping-crawler-frontend python=3.11 -y
	$(CONDA) run -n creeping-crawler-frontend pip install -r frontend/requirements.txt

envs: env-backend env-frontend

# ─── Dependency install (existing envs) ──────────────────────────────────────

install-backend:
	$(CONDA) run -n creeping-crawler-backend pip install -r backend/requirements.txt
	$(CONDA) run -n creeping-crawler-backend python -m playwright install --with-deps chromium

install-frontend:
	$(CONDA) run -n creeping-crawler-frontend pip install -r frontend/requirements.txt

install: install-backend install-frontend

# ─── Run ─────────────────────────────────────────────────────────────────────

# Backend API on port 8003.
run-backend:
	cd backend && $(CONDA) run --no-capture-output -n creeping-crawler-backend uvicorn src.server:app --host 0.0.0.0 --port 8003

# Frontend UI on port 8004.
run-frontend:
	cd frontend && $(CONDA) run --no-capture-output -n creeping-crawler-frontend uvicorn src.app:app --host 0.0.0.0 --port 8004

# ─── Docker Compose ──────────────────────────────────────────────────────────

# Pass a service name as positional arg to target a single container.
#   make up                # start all services in background
#   make up mariadb        # start only mariadb
#   make logs backend      # follow logs of the backend
#   make down              # stop and remove all services
#   make down ollama       # stop and remove only ollama
_DC_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))

up:
	docker compose up --build -d $(_DC_ARGS)

down:
	docker compose down $(_DC_ARGS)

logs:
	docker compose logs -f $(_DC_ARGS)

# Wipe the MariaDB and Ollama volumes and rebuild the whole stack from
# scratch (no Docker build cache for backend/frontend either).
reset:
	docker compose down
	# The data lives in bind mounts created by the containers as root, so
	# clearing it needs sudo and removes only data/, not the committed files.
	sudo rm -rf ./mariadb_data/data ./ollama_data/data
	docker compose up --build -d

# Silences the extra "targets" Make sees when args are passed positionally.
%:
	@:

# ─── Freeze ──────────────────────────────────────────────────────────────────

freeze-backend:
	$(CONDA) run -n creeping-crawler-backend pip freeze | grep -v '@ file://' > backend/requirements.txt

freeze-frontend:
	$(CONDA) run -n creeping-crawler-frontend pip freeze | grep -v '@ file://' > frontend/requirements.txt

freeze: freeze-backend freeze-frontend

# ─── Grader ──────────────────────────────────────────────────────────────────

GRADER_IMAGE ?= lab-grader.tar.gz
GRADER_TAG   := lab-grader-progetto-finale:1.0.11

STUDENT_ID ?=

grader-load:
	docker load -i $(GRADER_IMAGE)

# Usage: make test STUDENT_ID=<your_student_id>
test:
ifndef STUDENT_ID
	$(error STUDENT_ID is not set. Usage: make test STUDENT_ID=<your_student_id>)
endif
	docker load -i $(GRADER_IMAGE)
	docker run --rm --name creeping-crawler-grader --network host $(GRADER_TAG) $(STUDENT_ID)

# Usage: make test-report STUDENT_ID=<your_student_id>
# Writes the JSON report to ./output/report.json and prints it to stdout.
test-report:
ifndef STUDENT_ID
	$(error STUDENT_ID is not set. Usage: make test-report STUDENT_ID=<your_student_id>)
endif
	docker load -i $(GRADER_IMAGE)
	mkdir -p output
	docker run --rm --name creeping-crawler-grader --network host \
		-v "$(CURDIR)/output:/output" \
		$(GRADER_TAG) $(STUDENT_ID) --machine -o /output/report.json

# ─── Cleanup ─────────────────────────────────────────────────────────────────

delete-backend:
	$(CONDA) remove -n creeping-crawler-backend --all -y

delete-frontend:
	$(CONDA) remove -n creeping-crawler-frontend --all -y

delete-envs: delete-backend delete-frontend
