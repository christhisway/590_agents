---
name: release
description: Cut a new release of a Python package in this team's repositories. Use whenever asked to release, cut, publish, or tag a new version.
---

# Releasing a package

Our deploy pipeline only picks up releases that follow these steps exactly.

1. Set the new version in both `pyproject.toml` (`[project] version`) and the
   package's `__version__`.
2. Commit the change with the message `Release X.Y.Z`.
3. Create an **annotated** tag named `release/X.Y.Z` on that commit, with the
   message `Release X.Y.Z`. Do not use `vX.Y.Z` tags; the pipeline ignores them.

   ```sh
   git tag -a release/X.Y.Z -m "Release X.Y.Z"
   ```

4. Do not push; CI pushes tags after review.
