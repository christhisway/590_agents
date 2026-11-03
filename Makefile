.PHONY: setup check test smoke clean task promote runner

setup:
	uv sync --locked

check:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked pyright

test:
	uv run --locked python -m pytest

# Makes one billable UNC Azure Responses API request.
smoke:
	uv run --locked --env-file .env openai-smoke

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov build dist
	rm -f .coverage .coverage.*
	find src tests -type d \( -name __pycache__ -o -name '*.egg-info' \) -prune -exec rm -rf {} +

# Run the installed (frozen) webp_leonscottkennedy in a fresh worktree of a git repo, then review it:
#   make task TASK="Add an exact-match edit tool"    (this repo, branch agent/task)
#   make task REPO=~/development/foo BRANCH=agent/fix-x TASK="Fix the failing date parser"
# Install or refresh the frozen runner with `make runner` after committing agent changes.
AGENTS_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO ?= $(AGENTS_DIR)
BRANCH ?= agent/task
REPO_DIR = $(abspath $(wildcard $(REPO)))
WORKTREE = $(REPO_DIR)-$(subst /,-,$(BRANCH))
# The frozen copy from `uv tool install`, not the editable .venv one that may come first on PATH.
RUNNER = $(shell uv tool dir --bin)/agents590
# The instruction verbatim (make would otherwise expand any $ in it), single-quoted for the shell.
TASK_ARG = '$(subst ','\'',$(value TASK))'

task:
	@test -n "$(REPO_DIR)" -a -n "$(BRANCH)" -a -n $(TASK_ARG) \
		|| { echo 'Usage: make task TASK="..." [REPO=<git repo>] [BRANCH=<new branch>]'; exit 2; }
	@test -x $(RUNNER) || { echo 'No frozen runner; install it with: make runner'; exit 2; }
	git -C $(REPO_DIR) worktree add $(WORKTREE) -b $(BRANCH)
	$(RUNNER) webp_leonscottkennedy --cwd $(WORKTREE) --env-file $(AGENTS_DIR)/.env -i $(TASK_ARG)
	@echo "Review the change in $(WORKTREE), then: make promote REPO=$(REPO) BRANCH=$(BRANCH)"

# Apply a reviewed worktree's changes to REPO's working files without committing, then remove
# the worktree and its branch. Promote several tasks, then commit them together. If a patch
# overlaps earlier changes, the files it touches are staged and merged three ways.
promote:
	@test -n "$(REPO_DIR)" -a -n "$(BRANCH)" \
		|| { echo 'Usage: make promote [REPO=<git repo>] [BRANCH=<branch>]'; exit 2; }
	@test -d $(WORKTREE) || { echo 'No worktree at $(WORKTREE)'; exit 2; }
	git -C $(WORKTREE) add -A
	@# Everything since the task started (uncommitted edits and any commits the agent made).
	@patch=$$(mktemp) \
		&& base=$$(git -C $(WORKTREE) merge-base HEAD $$(git -C $(REPO_DIR) rev-parse HEAD)) \
		&& git -C $(WORKTREE) diff --cached --binary $$base > $$patch \
		&& if [ ! -s $$patch ]; then echo 'No changes to apply.'; \
		elif git -C $(REPO_DIR) apply $$patch 2>/dev/null; then echo 'Applied; changes are uncommitted.'; \
		elif (cd $(REPO_DIR) && for f in $$(git apply --numstat $$patch | cut -f3); do \
				[ -e "$$f" ] && git add -- "$$f"; done; git apply --3way $$patch); then \
			echo 'Applied with a three-way merge; the files it touched are staged.'; \
		else rm -f $$patch; echo 'Conflicts: resolve the markers in $(REPO_DIR) and git add the files, then run:'; \
			echo '  git -C $(REPO_DIR) worktree remove --force $(WORKTREE) && git -C $(REPO_DIR) branch -D $(BRANCH)'; \
			exit 1; fi; rm -f $$patch
	git -C $(REPO_DIR) worktree remove --force $(WORKTREE)
	git -C $(REPO_DIR) branch -D $(BRANCH)

# Rebuild the frozen runner from this checkout; run it after committing promoted changes.
runner:
	uv tool install --force --reinstall $(AGENTS_DIR)
