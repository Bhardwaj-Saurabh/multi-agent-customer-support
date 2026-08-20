# Resubmission Notes

This document maps each reviewer concern from the first review to its evidence in this submission.

## 1. `.env` included ✅

`.env` is now tracked in the repository with all deployment values:

| Key | Value |
|---|---|
| `RETURNS_KB_ID` | `H6RAQJICLX` |
| `SHIPPING_KB_ID` | `QNWPLCCB4B` |
| `WARRANTY_KB_ID` | `HRGNMKHXHB` |
| `AGENTCORE_RUNTIME_ARN` | `arn:aws:bedrock-agentcore:us-east-1:044138078595:runtime/udacity_agentcore_runtime-XnGpv7EvJU` |
| `GUARDRAIL_ID` | `mlrnpoy3z3bm` |
| `GUARDRAIL_VERSION` | `2` |

The three AWS credential lines are intentionally blank — they are short-lived lab session
tokens and are never committed. All other values reflect the live deployment.

## 2. Test runs at each step ✅ (screenshots in `docs/screenshots/`)

| Screenshot | Command | Result |
|---|---|---|
| `test_task2.png` | `python tests/test_agent.py task2` | 40/40 (100%) |
| `test_task3.png` | `python tests/test_agent.py task3` | 20/20 (100%) |
| `test_task5.png` | `python tests/test_agent.py task5` | 25/25 (100%) |
| `test_all.png` | `python tests/test_agent.py all` | **120/120 (100%)** — includes Task 4 (15/15) and Task 6 (20/20) sections |

Note on Task 4 / Task 6 standalone runs: the starter's SDK-compatibility patch for the
`bedrock-agentcore` clients is registered when `agent_orchestrator` is imported
(see `_register_agentcore_compat_methods()` in the starter code). The provided
`tests/test_agent.py` only imports that module inside the Task 2 suite, so `task4`/`task6`
verification is captured from the `all` run where the patch is active — visible in
`test_all.png` with both sections passing.

Note on `test_5_parallel_retrieval`: the `tests/test_agent.py` distributed with this
starter does not contain a test by that name (Task 5 comprises `test_5_1`–`test_5_3`,
all passing). Parallel retrieval is demonstrated live instead — see
`orchestrator_test_scenario2.png`, which shows `search_all_policies` dispatching all
three Knowledge Base retrievers in parallel and each returning passages.

## 3. End-to-end run: `python src/agent_orchestrator.py test` ✅

Three scenarios captured (`orchestrator_test_scenario{1,2,3}.png`):

1. **Return request** (`ORD-27176`) → Orchestrator → Inventory → Refund → Communication
2. **Policy question** (premium return policy) → Orchestrator → Policy (3 parallel KB retrievers) → Communication
3. **Math question** → Orchestrator answers directly → Communication

## 4. AWS console deployment screenshots ✅

Console screenshots are included in `docs/screenshots/`:

| Screenshot | Shows |
|---|---|
| `kb1.png`, `kb2.png`, `kb3.png` | Each Knowledge Base detail page: Available status, service role, and synced S3 data source |
| `agentcoreruntime.png` | AgentCore Runtime `udacity_agentcore_runtime` (READY, PUBLIC network, MCP protocol) |
| `memory.png` | AgentCore Memory (SESSION_SUMMARY strategy, 7-day expiry) |
| `guardrail.png` | Guardrail `udacity-agentcore-guardrail` v2 with its policies |
| `tracemap.png` | CloudWatch X-Ray Trace Map: fully connected OrchestratorAgent → workers graph |
| `traces.png` | X-Ray trace list for the live runs |

They capture:

- **Bedrock → Knowledge Bases**: `novamart-returns-policy-kb`, `novamart-shipping-policy-kb`,
  `novamart-warranty-policy-kb` — each detail page shows the Titan Embed Text v2 embedding
  model, the S3 Vectors store (`udacity-agentcore-vectors-044138078595-52743820`,
  indexes `returns/shipping/warranty-policy-index`), and a synced S3 data source
  (`policies/returns/`, `policies/shipping/`, `policies/warranty/`).
- **Bedrock AgentCore → Runtimes**: `udacity_agentcore_runtime` in READY state
  (PUBLIC network, MCP protocol) with environment variables including the guardrail reference.
- **Bedrock AgentCore → Memory**: `udacity_agentcore_memory` (SESSION_SUMMARY strategy, 7-day expiry).
- **Bedrock → Guardrails**: `udacity-agentcore-guardrail` version 2 with content, PII,
  topic, and word policies.
- **CloudWatch → X-Ray Trace Map**: the connected service graph
  (OrchestratorAgent → routing tools → all four worker agents, PolicyAgent fanning out
  to the three retrievers and Bedrock model nodes).

## 5. Extras

- `docs/guardrail-validation.md` — adversarial validation of the guardrail
  (10 live `ApplyGuardrail` test cases: prompt injection, denied topics, PII, profanity,
  violence — 9/9 adversarial inputs intercepted, benign control passes).
