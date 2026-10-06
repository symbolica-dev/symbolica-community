# Community development guidance

## Responsibility and commit authorship

At the start of each new task, ask who is responsible for the work and establish
their commit author name and email before committing. Do not infer responsibility
from the shell's existing Git or JJ configuration. If the user has already named
the responsible person for the current task, use that answer without asking
again. Set the author explicitly with JJ; changing the committer alone is not
sufficient. Ben Ruijl's established identity is `Ben Ruijl <ben@ruijl.ch>`.

## Feynkit dependency source

Always source Feynkit and its companion workspace crates from the `feynkit`
branch of `https://github.com/alphal00p/gammaloop`. Use that Git URL and
`branch = "feynkit"` in published Cargo dependencies and registry patches, and
commit the exact resolved revision in `Cargo.lock`. Keep these crates on one
source and revision so their native types and Symbolica kernel remain shared.
Vakint is a separate dependency with its own independently maintained pin.

Before bumping Feynkit, fetch `origin/feynkit`. Consolidate any required fixes
from forks or task branches onto that latest head with JJ, rebasing or replaying
and linearizing commits as necessary. Resolve conflicts while preserving both
the upstream behavior and required fixes. Run relevant checks, advance and push
the `feynkit` bookmark to `alphal00p/gammaloop`, then update Community's lockfile
and verify the packaged integration. Do not publish a Community dependency on a
Feynkit fork or a temporary feature branch to avoid consolidation.

Remove obsolete fork-based patch tables when the upstream branch already supplies
the required changes. Verify that transitive Feynkit dependencies resolve to the
same upstream branch and locked revision. Local path overrides are development
tools only; keep machine-specific paths and configuration out of commits.

## Isolation and verification

Use JJ and an isolated workspace for each independent task. Preserve other
workspaces' edits, runtime pairings, and running notebook sessions. Build the
complete Community extension into a private environment so all modules share
one Symbolica kernel; do not copy individual native extensions into a shared
installation. Publishing source changes and deploying a shared notebook server
are separate actions.

Run focused Rust and Python checks for the affected behavior, keep generated
stubs consistent with the bindings, and preserve existing test assertions when
resolving conflicts. Never store or print license keys or authentication tokens.
