.PHONY: setup check test smoke clean task promote

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
# Install the frozen runner once with `uv tool install --reinstall .`; promote refreshes it.
AGENTS_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO ?= $(AGENTS_DIR)
BRANCH ?= agent/task
REPO_DIR = $(abspath $(wildcard $(REPO)))
WORKTREE = $(REPO_DIR)-$(subst /,-,$(BRANCH))
# The instruction verbatim (make would otherwise expand any $ in it), single-quoted for the shell.
TASK_ARG = '$(subst ','\'',$(value TASK))'

task:
	@test -n "$(REPO_DIR)" -a -n "$(BRANCH)" -a -n $(TASK_ARG) \
		|| { echo 'Usage: make task TASK="..." [REPO=<git repo>] [BRANCH=<new branch>]'; exit 2; }
	git -C $(REPO_DIR) worktree add $(WORKTREE) -b $(BRANCH)
	agents590 webp_leonscottkennedy --cwd $(WORKTREE) --env-file $(AGENTS_DIR)/.env -i $(TASK_ARG)
	@echo "Review the change in $(WORKTREE), then: make promote REPO=$(REPO) BRANCH=$(BRANCH)"

# Commit the reviewed worktree, merge it into REPO's checked-out branch, and remove the
# worktree. Reinstalls the frozen runner when the agent worked on this repository.
promote:
	@test -n "$(REPO_DIR)" -a -n "$(BRANCH)" \
		|| { echo 'Usage: make promote [REPO=<git repo>] [BRANCH=<branch>] [MSG="..."]'; exit 2; }
	git -C $(WORKTREE) add -A
	git -C $(WORKTREE) diff --cached --quiet \
		|| git -C $(WORKTREE) commit -m "$${MSG:-Agent task on $(BRANCH)}"
	git -C $(REPO_DIR) merge $(BRANCH)
	git -C $(REPO_DIR) worktree remove $(WORKTREE)
	git -C $(REPO_DIR) branch -d $(BRANCH)
	if [ "$(REPO_DIR)" = "$(AGENTS_DIR)" ]; then uv tool install --force --reinstall $(AGENTS_DIR); fi

