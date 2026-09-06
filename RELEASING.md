# Releasing 0.4.0

The version is set in `pyproject.toml` and `src/dspy_toon/__init__.py`.
Keep both values in sync. The changelog entry remains unreleased until publication.

Build into a version-specific directory to avoid uploading older local archives:

```bash
uv build --out-dir dist/0.4.0
uv run --no-project --with twine python -I -m twine check --strict dist/0.4.0/*.whl dist/0.4.0/*.tar.gz
```

The wheel contains only `dspy_toon`, package metadata and license notices. The
source archive adds the build configuration and README. Both use an explicit
file selection, so research artifacts stay out of pip installations.

The existing Publish to PyPI workflow builds and checks artifacts when run
manually. Publishing a GitHub release also enables its PyPI upload job. Confirm
the `pypi` environment and trusted publisher configuration before publishing a
release. No package has been uploaded as part of this preparation.

## Optional fresh Git history

GitHub already identifies this repository as independent, not as a fork.
The inherited contributors come from its commit history.

The local `codex/0.4.0-snapshot` branch is intended as a single root commit of
the prepared working tree. It records the snapshot under the maintainer's
configured Git identity; it does not claim sole authorship of inherited work.
The original MIT attribution remains in `NOTICE` and in both package archives.

Keep the history backup outside the repository. Review the snapshot before
replacing the remote default branch. A public replacement requires a force
push with a lease against the freshly verified remote `main` commit. Existing
clones will need to reconcile their history or clone again. Old tags and other
branches can still retain earlier commits; the old history is not erased from
GitHub by changing `main` alone.

GitHub calculates its contributor graph from the default branch and may take
time to refresh it. A fresh history changes that graph; it does not remove
historical attribution or guarantee that every GitHub view immediately changes.

Do not publish a release or replace remote history until the maintainer has
approved those actions.
