# Repository agent guidance

## CodeGraph

Use CodeGraph for every nontrivial code change to trace relevant code paths, callers, callees, and dependencies before editing. It is especially useful when a change crosses modules. This workflow applies to application code, benchmark tools, and tests; a documentation-only edit does not need a graph query.

- Run CodeGraph commands from the repository root so they use this repository's index.
- Before using the graph, run `codegraph status .`. If `.codegraph/` is absent, initialize it with `codegraph init .`; do not reinitialize an existing index.
- Before editing, use `codegraph explore <topic>` to map the relevant area. Follow up on important symbols with `codegraph node <symbol>`, `codegraph callers <symbol>`, `codegraph callees <symbol>`, or `codegraph impact <symbol>` as appropriate; use `codegraph query <search>` when you need to locate a symbol by name.
- After source changes, run `codegraph sync .`, then `codegraph affected <changed-source-files...>` to identify likely tests. Run the relevant tests directly; the graph is a navigation aid and is not a substitute for test execution.
- Treat graph results as a navigation aid, not proof of complete coverage. Dynamic calls, lambdas, generated code, and test relationships may be missed; inspect the source and run the relevant tests directly.
- Keep the local `.codegraph/` database out of commits. Commit only intentional, portable project configuration for CodeGraph.

Typical sequence:

```sh
codegraph status .
codegraph explore "<feature or behavior>"
codegraph impact <symbol>          # when changing an existing symbol
# make the source changes
codegraph sync .
codegraph affected <changed-source-files...>
# run the relevant tests directly
```
