# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Existing documentation structure

This repo uses its existing documentation rather than a separate `CONTEXT.md` layout:

- `README.md` — project overview, citation, roadmap, and high-level usage pointer.
- `CHANGELOG.md` — release notes and project updates.
- `docs/README.md` — local ReadTheDocs/Sphinx build instructions.
- `docs/source/index.rst` — documentation entrypoint.
- `docs/source/tutorials/` — user-facing tutorials for installation, tasks, datasets, models, attacks, defenses, configs, and wrappers.
- `docs/source/modules/` — generated module documentation.
- `docs/source/notes/` — project notes including contributing, changelog, and scalability.

## Before exploring, read these

Read the most relevant existing docs for the area you are about to touch:

- For installation, usage, supported modalities, or extension work, read the relevant file under `docs/source/tutorials/`.
- For API/module behavior, read `docs/source/modules/` and then verify against the source code.
- For contribution or release context, read `docs/source/notes/`, `CHANGELOG.md`, and `README.md`.

If documentation is missing or stale, verify behavior in code and keep any documentation changes consistent with the implementation.

## Keep docs and code aligned

When changing public behavior, update the corresponding docs in the existing structure. Do not introduce a new domain-doc layout unless the project explicitly adopts one later.

## Vocabulary

Use the project's existing terms from the docs and code, especially around backdoor learning, multimodal tasks, attacks, defenses, poison datasets, configs, and training pipelines. Avoid inventing synonyms when the docs already use a term.
