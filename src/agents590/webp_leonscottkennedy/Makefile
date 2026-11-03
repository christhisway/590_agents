# Leon's self-contained task workflow, kept here so the shared Makefile stays untouched.
# Run from anywhere with -f (relative REPO paths resolve against your current directory):
#   make -f src/agents590/webp_leonscottkennedy/Makefile task TASK="Add an exact-match edit tool"
#   make -f src/agents590/webp_leonscottkennedy/Makefile promote
#   make -f src/agents590/webp_leonscottkennedy/Makefile runner
# task runs the installed (frozen) agent in a fresh worktree of a git repo (this repo and
# branch agent/task by default); review the worktree, then promote it. Refresh the frozen
# runner with `runner` after committing agent changes.
.PHONY: task promote runner

LEON_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
AGENTS_DIR := $(abspath $(dir $(LEON_MAKEFILE))/../../..)
LEON := make -f $(LEON_MAKEFILE)
REPO ?= $(AGENTS_DIR)
BRANCH ?= agent/task
REPO_DIR = $(abspath $(wildcard $(REPO)))
WORKTREE = $(REPO_DIR)-$(subst /,-,$(BRANCH))
# The frozen copy from `uv tool install`, not the editable .venv one that may come first on PATH.
RUNNER = $(shell uv tool dir --bin)/agents590
# The instruction verbatim, passed to recipes through the environment: make would otherwise
# expand any $ in it and split a multi-line instruction into separate shell commands.
export LEON_TASK := $(value TASK)

task:
	@test -n "$(REPO_DIR)" -a -n "$(BRANCH)" -a -n "$$LEON_TASK" \
		|| { echo 'Usage: $(LEON) task TASK="..." [REPO=<git repo>] [BRANCH=<new branch>]'; exit 2; }
	@test -x $(RUNNER) || { echo 'No frozen runner; install it with: $(LEON) runner'; exit 2; }
	git -C $(REPO_DIR) worktree add $(WORKTREE) -b $(BRANCH)
	$(RUNNER) webp_leonscottkennedy --cwd $(WORKTREE) --env-file $(AGENTS_DIR)/.env -i "$$LEON_TASK"
	@echo "Review the change in $(WORKTREE), then: $(LEON) promote REPO=$(REPO_DIR) BRANCH=$(BRANCH)"

# Apply a reviewed worktree's changes to REPO's working files without committing, then remove
# the worktree and its branch. Promote several tasks, then commit them together. If a patch
# overlaps earlier changes, the files it touches are staged and merged three ways.
promote:
	@test -n "$(REPO_DIR)" -a -n "$(BRANCH)" \
		|| { echo 'Usage: $(LEON) promote [REPO=<git repo>] [BRANCH=<branch>]'; exit 2; }
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
