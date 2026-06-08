# CLAUDE.md

## Language

All documentation and output **must be in English**. This includes:

- Source code comments and docstrings
- README and any files under `docs/` and `examples/`
- Commit messages and pull request descriptions
- CLI output, help text, log and error messages
- Any response or summary produced while working in this repository

Write new content in English and translate existing non-English content to
English when you touch it.

See also [AGENTS.md](AGENTS.md).

## Karpathy's best practices for AI-assisted coding

Andrej Karpathy's guidance for code you *professionally care about* (as opposed to
throwaway "vibe coding"). Follow this rhythm when working in this repo:

1. **Load full context first.** Pull everything relevant into context before changing
   anything. If the project is small, just include all of it.
2. **One concrete, incremental change at a time.** Describe the single next step;
   don't sprawl across the codebase in one go.
3. **Ask for approaches before code.** Prefer a few high-level options with pros/cons
   over jumping straight to an implementation; pick one deliberately.
4. **Review and learn every diff.** Read the generated code, check API docs, ask for
   explanations, push back. Don't merge code you don't understand.
5. **Test, then commit.** Verify (manually and with tests), then `git commit` so each
   step is small and revertible. Repeat.

Two guiding principles:

- **Keep the AI on a tight leash.** Small, concrete, verifiable increments beat large
  free-roaming changes that drift and produce massive, inscrutable diffs.
- **Make verification fast and easy.** Optimize for tight generation→review feedback
  loops; the faster a human can verify, the faster the loop can spin.

Mindset: treat the AI as an *over-eager junior intern* — encyclopedic but prone to
confident mistakes. Stay slow, defensive, careful, and paranoid; always take the
inline learning opportunity rather than blindly delegating.

Sources: [Karpathy's AI-assisted coding rhythm (X, 2025)](https://x.com/karpathy/status/1915581920022585597)
· ["Software Is Changing (Again)" talk notes](https://catalaize.substack.com/p/andrej-karpathy-software-is-changing).
