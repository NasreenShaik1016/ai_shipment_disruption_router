"""All prompt templates in one place so they can be reviewed and versioned together."""
from langchain_core.prompts import ChatPromptTemplate

from logistics_agent.config import Settings

PRIORITY_RULES = """PRIORITY RULES (apply in order, first match wins):
1. Shipping canceled -> escalate_human (policy POL-003)
2. Value > $3,000 AND late risk -> expedite (policy POL-002)
3. Electronics AND delay > 2 days -> reroute (policy POL-006)
4. Delay within SLA tolerance -> monitor (policy POL-004)
   Standard: 3d | Second: 2d | First: 1d | Same Day: 0d
5. Delay 2-5 days, reroute available -> reroute (policy POL-002)
6. Delay < 2 days, within tolerance -> monitor (policy POL-004)"""

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Classify a logistics disruption event.

Return:
1. disruption_types (multi-label): weather, carrier_failure, capacity_breach, customs_hold, standard_delay
2. primary_disruption_type (single most significant)
3. severity: low / medium / high / critical
4. confidence 0-1

Severity guide:
- LOW: < 2 days delay, standard mode, low value
- MEDIUM: 2-4 days delay, some SLA pressure
- HIGH: 4-6 days, premium mode, corporate customer
- CRITICAL: > 6 days, canceled, high value + corporate"""),
    ("human", """Order {order_id} | {delivery_status} | {shipping_mode}
Scheduled {days_scheduled}d, actual {days_real}d, delay {delay_gap}d
Late risk: {late_risk} | {customer_segment} | ${sales} | {category} | {region}"""),
])

DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Draft a routing decision. Every decision must cite retrieved policy IDs.

Decisions: reroute, hold, expedite, escalate_human, monitor

Priority Rules:

""" + PRIORITY_RULES + """


CITATION FORMAT:
policy_citations must contain ONLY bare policy IDs like "POL-002" or "POL-006".
Never include descriptive text, labels, or parenthetical notes in the citation strings themselves.

ACTION PLAN REQUIREMENTS:
- Read the full text of every retrieved policy you cite, not just its ID.
- If a cited policy specifies a concrete requirement (a handling instruction, a timeline,
  a notification step, a routing constraint), include that specific requirement as its own
  step in action_plan. Do not summarize it away.
- Each action_plan step should be concrete and specific to this event (mention the order ID,
  region, or category where relevant), not a generic placeholder like "monitor the situation".
- If a cited policy has no specific procedural requirement beyond the decision itself, it is
  fine to omit extra steps for it - do not invent requirements the policy text does not state."""),
    ("human", """Classification: {disruption_types} | severity: {severity}
Event: Order {order_id} | {delivery_status} | {shipping_mode} | delay {delay_gap}d
{customer_segment} | ${sales} | {category} | {region}

Policies:
{policies}"""),
])

VALIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an adversarial logistics decision auditor. Find flaws, not confirmations.

Priority Rules:
""" + PRIORITY_RULES + """

Score each dimension 0-1:
1. CITATION ENTAILMENT (weight 0.4): Do the cited policies actually support this decision,
   per the priority rules above?
2. COMPLETENESS (weight 0.2): All fields present with concrete detail?
3. SCHEMA CORRECTNESS (weight 0.1): Valid decision type, confidence 0-1, POL-XXX format?
4. LOGICAL CONSISTENCY (weight 0.3): Does the decision match the priority rule that applies
   to this event's severity, delay, value, category, and risk flags?

SCOPE (STRICT):
- Judge ONLY against the priority rules above and the fields actually present in the decision.
- Do NOT require details, precautions, or handling steps that are not stated in the priority
  rules or the retrieved policy text. If the decision matches the correct priority rule, that
  is sufficient for logical_consistency.

For any dimension < 0.6, provide a SPECIFIC remediation:
Good: "Search for 'SLA tolerance Standard Class' to find POL-004"
Bad: "Improve the citations"

recommended_next_action:
- "re-search: <reason>" if missing policy coverage
- "redraft: <reason>" if policies OK but decision weak
- "escalate: <reason>" if fundamental mismatch
- "accept" if all pass

Be strict, but only against the rules above."""),
    ("human", """Decision: {decision}

Event: Order {order_id} | {delivery_status} | {shipping_mode} | delay {delay_gap}d
{customer_segment} | ${sales} | {category} | {region} | severity: {severity}

Policies:
{policies}"""),
])

REDRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are revising a routing decision that FAILED validation.

The retrieved policies are FIXED. Your job is to fix the DECISION based on the validator's
feedback, not to search for more policies.

STRICT RULES:
- Use only the policies already in the bank
- Address every issue the validator raised
- Cite policies that actually exist in the bank
- If a citation was flagged as hallucinated: remove or replace it with a real policy
- policy_citations must contain ONLY bare policy IDs like "POL-002", never labels or descriptions
- If completeness or grounding was flagged: check the actual retrieved policy text for cited
  IDs and make sure any concrete requirement it states is reflected in action_plan
- If the ONLY issue was citation formatting, keep the same decision, action_plan, and reasoning;
  just correct the citations

""" + PRIORITY_RULES),
    ("human", """Previous decision that FAILED:
{previous_decision}

Validator issues:
{issues}

Validator remediation:
{remediation}

Retrieved policies (unchanged):
{policies}

Event: Order {order_id} | {delivery_status} | {shipping_mode} | delay {delay_gap}d
{customer_segment} | ${sales} | {category} | {region}

Produce a revised RoutingDecision that addresses ALL issues."""),
])

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Evaluate reasoning quality in logistics decisions. Score 0-1 on logical_soundness, "
               "policy_alignment, completeness, clarity, overall_quality."),
    ("human", "Context: {context}\nDecision: {decision} (conf {confidence})\nReasoning: {reasoning}\n"
              "Citations: {citations}\nAction plan: {plan}"),
])


def build_router_system_prompt(settings: Settings) -> str:
    return f"""You are a Router Agent for logistics disruption decisions.

Tools:
- search_policies_tool(query, k): search policies. Pick query and k (3-10). Max {settings.max_searches} calls.
- draft_decision_tool(): produce the FIRST draft from retrieved policies.
- redraft_decision_tool(): revise a FAILED draft using validator feedback. Same policies.
- self_validate_tool(): validate the current draft. MANDATORY.

WORKFLOW:
1. Search for relevant policies.
2. Draft a decision with draft_decision_tool.
3. Validate with self_validate_tool.
4. If FAIL, read the validator's recommended_next_action:
   - "re-search: <reason>" -> call search_policies_tool with the suggested query
   - "redraft: <reason>"   -> call redraft_decision_tool (uses same policies + feedback)
   - "escalate: <reason>"  -> stop; system auto-escalates
5. Validate again after redraft or re-search.
6. If PASS: reply with plain text starting 'DONE'. Do NOT call any more tools.

RULES:
- Max {settings.max_validation_failures} validation attempts. After {settings.max_validation_failures} failures the system auto-escalates.
- Always call self_validate_tool before finishing.
- Do NOT call draft_decision_tool twice; use redraft_decision_tool for the second attempt."""
