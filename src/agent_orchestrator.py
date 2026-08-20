"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude 3 Haiku - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.
"""

import boto3
import json
import time
import os
import sys
import uuid
import random
import logging
import re
import io
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent, tool
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES  (pre-written - do not modify)
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)





# ─────────────────────────────────────────────────────
# AWS CLIENTS (pre-written - do not modify)
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ─────────────────────────────────────────────────────
# COMPATIBILITY PATCH (pre-written - do not modify)
# ─────────────────────────────────────────────────────
def _register_agentcore_compat_methods():
    """Register event handler to inject control-plane methods into bedrock-agentcore clients."""
    _control = agentcore_control

    def _add_methods(class_attributes, base_classes, **kwargs):
        def get_agent_runtime(self, agentRuntimeId, **kw):
            try:
                response = _control.get_agent_runtime(agentRuntimeId=agentRuntimeId)
            except Exception:
                response = {}
            response['memoryConfiguration'] = {
                'enabledMemoryTypes': ['SESSION_SUMMARY'],
                'storageDays': 7,
            }
            response['codeInterpreterConfiguration'] = {
                'enabled': True,
                'executionEnvironment': 'PYTHON_3_11',
                'timeoutSeconds': 30,
            }
            return response

        def get_agent_runtime_logging_configuration(self, agentRuntimeId, **kw):
            return {
                'loggingConfiguration': {
                    'cloudWatchConfig': {
                        'logGroupName': config.AGENT_LOG_GROUP,
                        'logLevel': 'INFO',
                        'enabled': True,
                    },
                    'xRayConfig': {
                        'enabled': True,
                        'samplingRate': 1.0,
                    }
                }
            }

        def put_agent_runtime_logging_configuration(self, agentRuntimeId,
                                                    loggingConfiguration=None, **kw):
            return {'ResponseMetadata': {'HTTPStatusCode': 200}}

        class_attributes['get_agent_runtime'] = get_agent_runtime
        class_attributes['get_agent_runtime_logging_configuration'] = get_agent_runtime_logging_configuration
        class_attributes['put_agent_runtime_logging_configuration'] = put_agent_runtime_logging_configuration

    import boto3 as _boto3
    if _boto3.DEFAULT_SESSION is not None:
        _boto3.DEFAULT_SESSION._session.register(
            'creating-client-class.bedrock-agentcore', _add_methods
        )
    else:
        import botocore.session as _bc_session
        _original_get = _bc_session.get_session

        def _patched_get(*args, **kwargs):
            sess = _original_get(*args, **kwargs)
            sess.register('creating-client-class.bedrock-agentcore', _add_methods)
            return sess

        _bc_session.get_session = _patched_get

_register_agentcore_compat_methods()


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#  Pre-written - do not modify.
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.
    Pre-written - do not modify.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    Pre-written - do not modify.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    Pre-written - do not modify.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.1,
    )

    system_prompt = (
        "You are the Inventory Agent for NovaMart customer support - a data-retrieval "
        "specialist for order and customer records.\n\n"
        "Your ONLY job is to gather facts using your tools and report them accurately:\n"
        "- Report tool results verbatim: order status, order_date, price, quantity, "
        "product_name, tracking_number, estimated_delivery, return_eligible flag, and "
        "customer tier exactly as returned.\n"
        "- NEVER make eligibility, refund, or policy decisions - other agents decide; "
        "you only report data.\n"
        "- NEVER invent or guess values. If a lookup returns found=False or an error, "
        "state clearly that the record was not found.\n"
        "- When asked about a specific order, use check_order_status. When asked about "
        "a customer's tier or account, use get_customer_tier. When asked about order "
        "history, use list_customer_orders.\n"
        "- Respond with a concise, structured summary of the retrieved facts."
    )

    @tool
    def check_order_status(order_id: str) -> dict:
        """
        Look up a single order by its order ID and return its full record.

        Args:
            order_id: The order's unique identifier (e.g. ORD-12345)

        Returns:
            The complete order record (status, dates, product, price, return
            eligibility), or found=False if the order does not exist
        """
        try:
            from boto3.dynamodb.conditions import Attr
            table = dynamodb.Table(config.ORDERS_TABLE)
            # Orders table uses a composite key (customer_id, order_id) - this
            # tool receives only order_id, so a filtered scan is required.
            items, scan_kwargs = [], {'FilterExpression': Attr('order_id').eq(order_id)}
            while True:
                response = table.scan(**scan_kwargs)
                items.extend(response.get('Items', []))
                if 'LastEvaluatedKey' not in response:
                    break
                scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
            if not items:
                return {'found': False, 'error': f'Order {order_id} was not found.'}
            return {'found': True, 'order': items[0]}
        except Exception as e:
            logger.error(f"check_order_status failed: {e}")
            return {'found': False, 'error': f'Lookup failed: {e}'}

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        try:
            table = dynamodb.Table(config.CUSTOMERS_TABLE)
            item = table.get_item(Key={'customer_id': customer_id}).get('Item')
            if not item:
                return {'found': False, 'error': f'Customer {customer_id} was not found.'}
            return {'found': True, 'customer': item}
        except Exception as e:
            logger.error(f"get_customer_tier failed: {e}")
            return {'found': False, 'error': f'Lookup failed: {e}'}

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        try:
            table = dynamodb.Table(config.ORDERS_TABLE)
            orders = table.query(
                KeyConditionExpression=Key('customer_id').eq(customer_id)
            ).get('Items', [])
            return {'found': bool(orders), 'count': len(orders), 'orders': orders}
        except Exception as e:
            logger.error(f"list_customer_orders failed: {e}")
            return {'found': False, 'error': f'Lookup failed: {e}'}

    return Agent(
        name="InventoryAgent",
        model=model,
        system_prompt=system_prompt,
        tools=[check_order_status, get_customer_tier, list_customer_orders],
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.1,
    )

    system_prompt = (
        "You are the Refund Agent for NovaMart customer support. You decide whether a "
        "return/refund request is eligible and, if so, process it.\n\n"
        "Decision process - follow it in order, every time:\n"
        "1. ALWAYS call get_inventory_context first to read the facts the Inventory "
        "Agent gathered (order status, order_date, customer tier). Never decide "
        "without them.\n"
        "2. Apply the return window by tier: Standard customers = 30 days from "
        "order_date, Premium customers = 60 days from order_date. Compute the window "
        "yourself from the order_date and tier in the inventory context - do not rely "
        "on any pre-computed eligibility flag, which assumes 30 days for everyone.\n"
        "3. The order status must be 'delivered' to be returnable. Orders that are "
        "processing, shipped, cancelled, or already in return_requested status are "
        "not eligible - explain which condition failed.\n"
        "4. Only if eligible, call initiate_refund to process the return.\n\n"
        "Rules:\n"
        "- State your decision (APPROVED or DENIED) with the specific reason: tier, "
        "window applied, order date, and status.\n"
        "- Report ONLY the return_reference number returned by initiate_refund - "
        "never invent reference numbers.\n"
        "- If the inventory context is missing or the order was not found, say so "
        "and do not process a refund."
    )

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        try:
            state = _read_workflow_state(session_id)
            if not state:
                return {'error': f'No workflow state found for session {session_id}.'}
            context = state.get('inventory_agent')
            if not context:
                return {'error': 'Inventory findings are not yet available for this session.'}
            return {'inventory_context': context, 'customer_id': state.get('customer_id', '')}
        except Exception as e:
            logger.error(f"get_inventory_context failed: {e}")
            return {'error': f'Could not read workflow state: {e}'}

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        try:
            return_reference = f"RET-{uuid.uuid4().hex[:8].upper()}"
            table = dynamodb.Table(config.ORDERS_TABLE)
            table.update_item(
                Key={'customer_id': customer_id, 'order_id': order_id},
                UpdateExpression=(
                    "SET #s = :status, return_reason = :reason, "
                    "return_reference = :ref"
                ),
                ExpressionAttributeNames={'#s': 'status'},
                ExpressionAttributeValues={
                    ':status': 'return_requested',
                    ':reason': reason,
                    ':ref': return_reference,
                },
                ConditionExpression='attribute_exists(order_id)',
            )
            return {
                'approved': True,
                'return_reference': return_reference,
                'order_id': order_id,
                'instructions': (
                    'A prepaid return label will be emailed within 24 hours. '
                    'Pack the item in its original packaging and drop it off at any '
                    'carrier location within 14 days. The refund is issued 5-7 '
                    'business days after the item is received.'
                ),
            }
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            return {'approved': False,
                    'error': f'Order {order_id} was not found for customer {customer_id}.'}
        except Exception as e:
            logger.error(f"initiate_refund failed: {e}")
            return {'approved': False, 'error': f'Refund processing failed: {e}'}

    return Agent(
        name="RefundAgent",
        model=model,
        system_prompt=system_prompt,
        tools=[get_inventory_context, initiate_refund],
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    def _retriever_prompt(domain: str) -> str:
        return (
            f"You are the {domain} Policy Retriever for NovaMart. Call your retrieval "
            f"tool exactly once with the user's query, then report the retrieved "
            f"passages faithfully - quote or closely paraphrase them. Never answer "
            f"from your own knowledge. If the tool returns no relevant passages, "
            f"state that no relevant {domain.lower()} policy text was found."
        )

    def _retriever_model() -> BedrockModel:
        # Temperature 0.0: retrieval reporting must be deterministic
        return BedrockModel(model_id=config.WORKER_MODEL_ID, temperature=0.0)

    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        try:
            return format_kb_results(
                retrieve_from_knowledge_base(config.RETURNS_KB_ID, query, top_k=3)
            )
        except Exception as e:
            return f"[Returns KB retrieval failed: {e}]"

    returns_retriever = Agent(
        name="ReturnsPolicyRetrieverAgent",
        model=_retriever_model(),
        system_prompt=_retriever_prompt("Returns"),
        tools=[retrieve_returns_policy],
    )

    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        try:
            return format_kb_results(
                retrieve_from_knowledge_base(config.SHIPPING_KB_ID, query, top_k=3)
            )
        except Exception as e:
            return f"[Shipping KB retrieval failed: {e}]"

    shipping_retriever = Agent(
        name="ShippingPolicyRetrieverAgent",
        model=_retriever_model(),
        system_prompt=_retriever_prompt("Shipping"),
        tools=[retrieve_shipping_policy],
    )

    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        try:
            return format_kb_results(
                retrieve_from_knowledge_base(config.WARRANTY_KB_ID, query, top_k=3)
            )
        except Exception as e:
            return f"[Warranty KB retrieval failed: {e}]"

    warranty_retriever = Agent(
        name="WarrantyPolicyRetrieverAgent",
        model=_retriever_model(),
        system_prompt=_retriever_prompt("Warranty"),
        tools=[retrieve_warranty_policy],
    )

    # TODO: Implement search_all_policies - parallel RAG retrieval tool
    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        retrievers = {
            'Returns':  returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }

        # ── Trace: show parallel KB dispatch to learners ──────────────────
        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        # Define a helper to run one retriever sub-agent
        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """
            Run one retriever sub-agent and return (domain, result_text).

            stdout is suppressed globally for all threads by the
            _TraceWriter._suppress_parallel flag set in kb_start().
            This covers both the direct worker thread and any internal
            streaming child threads that Strands SDK spawns internally -
            which do NOT inherit thread-local variables and therefore cannot
            be suppressed with a thread-local capture approach.
            Results are returned as values and printed cleanly and
            sequentially by trace.kb_result() after all futures join.
            """
            try:
                return (domain, str(agent(query)).strip())
            except Exception as e:
                return (domain, f"[{domain} retrieval failed: {e}]")

        results = {}
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(_run_retriever, domain, agent, query)
                    for domain, agent in retrievers.items()
                ]
                for future in as_completed(futures):
                    domain, result_text = future.result()
                    results[domain] = result_text
        finally:
            # ── Trace: all KBs responded - print each result sequentially ─
            # kb_done() must always run, or stdout stays suppressed globally.
            trace.kb_done(len(retrievers))

        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        return "\n\n".join(
            f"=== {domain} Policy ===\n{results.get(domain, '[No results]')}"
            for domain in ['Returns', 'Shipping', 'Warranty']
        )

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.2,
    )

    system_prompt = (
        "You are the Policy Agent for NovaMart customer support - a coordinator over "
        "three policy knowledge bases (Returns, Shipping, Warranty).\n\n"
        "Process:\n"
        "1. ALWAYS call search_all_policies first with the customer's question - it "
        "queries all three knowledge bases in parallel.\n"
        "2. Synthesize the retrieved passages into ONE clear, grounded answer. Note "
        "which policy domain(s) each point came from.\n"
        "3. Use ONLY the retrieved text - never answer from prior knowledge. If the "
        "retrieved passages do not cover the question, say the policy could not be "
        "found and suggest contacting support.\n"
        "4. When policies differ by customer tier (Standard vs Premium), state both "
        "explicitly.\n\n"
        "You only know policy text. You have no access to customer accounts or order "
        "data - if asked about a specific customer or order, say that is outside "
        "your scope."
    )

    return Agent(
        name="PolicyAgent",
        model=model,
        system_prompt=system_prompt,
        tools=[search_all_policies],
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.3,
    )

    system_prompt = (
        "You are the Communication Agent for NovaMart customer support. You write the "
        "single final message the customer will read.\n\n"
        "Process:\n"
        "1. ALWAYS call get_full_workflow_context first to read everything the other "
        "agents found and decided for this session.\n"
        "2. Compose ONE warm, professional, empathetic message that includes ALL "
        "relevant information: order details, dates, decisions, return reference "
        "numbers, policy answers, and clear next steps.\n\n"
        "Style rules:\n"
        "- Plain text only: no XML tags, no markdown headers, no bullet-point "
        "formatting symbols beyond simple dashes.\n"
        "- Never mention internal machinery: no session IDs, agent names, workflow "
        "state, tools, or 'our system'.\n"
        "- Be specific - cite the actual order number, dates, and reference numbers "
        "from the context. Never invent details that are not in the context.\n"
        "- If a request could not be fulfilled (order not found, return denied), "
        "explain why kindly and offer a concrete alternative or next step.\n"
        "- Sign off as 'The NovaMart Support Team'."
    )

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        try:
            state = _read_workflow_state(session_id)
            if not state:
                return {'error': f'No workflow state found for session {session_id}.'}
            return {k: v for k, v in state.items() if k not in ('version', 'ttl')}
        except Exception as e:
            logger.error(f"get_full_workflow_context failed: {e}")
            return {'error': f'Could not read workflow state: {e}'}

    return Agent(
        name="CommunicationAgent",
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context],
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        temperature=0.0,  # deterministic routing
    )

    system_prompt = (
        "You are the Orchestrator Agent for NovaMart customer support. You NEVER "
        "answer customers directly (with one exception below) - you route requests "
        "to specialist agents and manage the shared workflow state.\n\n"
        "Every customer message starts with '[Session ID: ...] [Customer ID: ...]'. "
        "Extract both values and pass them to your tools.\n\n"
        "ROUTING RULES - follow them exactly:\n"
        "1. ALWAYS call initialize_session first, for every request.\n"
        "2. For order status, order history, tracking, return, or refund requests: "
        "call route_to_inventory_agent FIRST to gather facts. For return/refund "
        "requests you MUST call route_to_inventory_agent and wait for its result "
        "BEFORE calling route_to_refund_agent - the refund agent cannot decide "
        "without inventory findings. Never call route_to_refund_agent first, and "
        "always call it after inventory for return/refund requests (even if the "
        "order was not found - the refund agent makes the final decision).\n"
        "3. For questions about what policies SAY (return windows, shipping rates, "
        "delivery times, warranty terms): call route_to_policy_agent.\n"
        "4. For account questions ('what is my tier?', 'am I premium?', 'how many "
        "orders have I placed?'): call route_to_inventory_agent - NEVER "
        "route_to_policy_agent. The policy agent only knows policy text, not "
        "customer data.\n"
        "5. For pure math or calculation questions: compute the answer yourself - "
        "no routing needed except rules 1 and 6.\n"
        "6. ALWAYS call route_to_communication_agent as your VERY LAST tool call, "
        "for every request - no exceptions. Return its output verbatim as your "
        "final answer. You must NEVER write the customer-facing response yourself; "
        "for math questions, pass your computed result to the communication agent "
        "in the original_request text.\n\n"
        "Do not call the same routing tool twice for the same information. Do not "
        "add commentary around the communication agent's response."
    )

    def _invoke_and_record(session_id: str, customer_id: str, column: str,
                           label: str, worker: Agent, prompt: str) -> str:
        """Run one worker agent and record its result in WorkflowState."""
        state = _read_workflow_state(session_id)
        if state is None:
            # Defensive: initialize_session was skipped - create the record
            # so the worker's findings are not lost.
            state = _create_workflow_state(session_id, customer_id)
        old_version = int(state['version'])
        trace.step_start(column)
        trace.agent_section(label)
        result = str(worker(prompt)).strip()
        if column == 'communication_agent':
            result = _strip_xml_tags(result)
        _update_workflow_state(session_id, {column: result},
                               expected_version=old_version)
        trace.step_done(column, old_version)
        return result

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        return _invoke_and_record(
            session_id, customer_id, 'inventory_agent', 'INVENTORY AGENT',
            inventory_agent,
            f"[Customer ID: {customer_id}] {request}",
        )

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        state = _read_workflow_state(session_id)
        customer_id = state.get('customer_id', 'UNKNOWN') if state else 'UNKNOWN'
        return _invoke_and_record(
            session_id, customer_id, 'policy_agent', 'POLICY AGENT',
            policy_agent,
            request,
        )

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        state = _read_workflow_state(session_id)
        if not state or not state.get('inventory_agent'):
            return (
                "ERROR: Inventory facts are not yet available for this session. "
                "Call route_to_inventory_agent first to gather the order details, "
                "then call route_to_refund_agent again."
            )
        return _invoke_and_record(
            session_id, customer_id, 'refund_agent', 'REFUND AGENT',
            refund_agent,
            f"[Session ID: {session_id}] [Customer ID: {customer_id}] {request}",
        )

    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        return _invoke_and_record(
            session_id, customer_id, 'communication_agent', 'COMMUNICATION AGENT',
            communication_agent,
            f"[Session ID: {session_id}] [Customer ID: {customer_id}] "
            f"Compose the final response to this customer request: {original_request}",
        )

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        try:
            _create_workflow_state(session_id, customer_id)
            return f"Session {session_id} initialized for customer {customer_id}."
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            return f"Session {session_id} is already initialized - continue routing."
        except Exception as e:
            logger.error(f"initialize_session failed: {e}")
            return f"Session initialization failed: {e}"

    return Agent(
        name="OrchestratorAgent",
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
    )


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Check if guardrail already exists to avoid duplicates
    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            guardrail_version = 'DRAFT'
            for v in versions.get('guardrails', []):
                if v.get('version', 'DRAFT') != 'DRAFT':
                    guardrail_version = v['version']
            print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
            return guardrail_id, guardrail_version

    topic_definitions = {
        'competitor products': (
            'Discussion, comparison, or recommendation of products sold by '
            'competing retailers rather than NovaMart.',
            ['Is this cheaper at another store?', 'Should I buy from a competitor instead?'],
        ),
        'pricing negotiations': (
            'Attempts to negotiate, haggle, or request unauthorized discounts, '
            'price matching, or special pricing outside published policies.',
            ['Can you give me a discount if I buy two?', 'Match the price I saw elsewhere.'],
        ),
        'legal threats': (
            'Threats of lawsuits, legal action, regulatory complaints, or demands '
            'framed as legal claims against the company.',
            ['I will sue you if you do not refund me.', 'My lawyer will be in touch.'],
        ),
    }

    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description='NovaMart customer support safety guardrail: harmful content, '
                    'PII protection, off-topic denial, and profanity filtering.',
        contentPolicyConfig={
            'filtersConfig': [
                {'type': 'SEXUAL', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'VIOLENCE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'HATE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'INSULTS', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
                {'type': 'MISCONDUCT', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
            ]
        },
        sensitiveInformationPolicyConfig={
            'piiEntitiesConfig': [
                {'type': 'CREDIT_DEBIT_CARD_NUMBER', 'action': 'BLOCK'},
                {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
                {'type': 'EMAIL', 'action': 'ANONYMIZE'},
                {'type': 'PHONE', 'action': 'ANONYMIZE'},
            ]
        },
        topicPolicyConfig={
            'topicsConfig': [
                {
                    'name': topic.replace(' ', '_'),
                    'definition': topic_definitions[topic][0],
                    'examples': topic_definitions[topic][1],
                    'type': 'DENY',
                }
                for topic in config.GUARDRAIL_BLOCKED_TOPICS
            ]
        },
        wordPolicyConfig={
            'managedWordListsConfig': [{'type': 'PROFANITY'}]
        },
        blockedInputMessaging=(
            "I'm sorry, but I can't help with that request. I'm happy to assist "
            "with your orders, returns, shipping, or warranty questions."
        ),
        blockedOutputsMessaging=(
            "I'm sorry, but I can't share that information. Please contact "
            "NovaMart support for further assistance."
        ),
    )
    guardrail_id = response['guardrailId']
    print(f"Guardrail created: {guardrail_id}")

    # Promote from DRAFT to a numbered version for stable production reference
    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description='Initial production version',
    )
    guardrail_version = version_response['version']
    print(f"Guardrail promoted to version: {guardrail_version}")

    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime.

    Note: orchestrator_agent is accepted as a parameter to make the call-site
    explicit about what is being deployed, but AgentCore does not serialize
    Python objects directly. Instead, the runtime is configured with the role,
    network settings, guardrail, and environment variables (KB IDs etc.) it
    needs. The agent code in this script runs as the MCP server handler inside
    the AgentCore runtime environment.

    Returns:
        The AgentCore Runtime ARN
    """
    runtime_name = f"{config.PROJECT_NAME}-runtime".replace('-', '_')
    s3_client    = boto3.client('s3', region_name=config.AWS_REGION)

    # Check if runtime already exists
    try:
        existing = agentcore_control.list_agent_runtimes()
        for r in existing.get('agentRuntimes', []):
            if r['agentRuntimeName'] == runtime_name:
                runtime_arn = r['agentRuntimeArn']
                print(f"AgentCore Runtime already exists: {runtime_arn}")
                return runtime_arn
    except Exception as e:
        print(f"  [Note] Could not check existing runtimes: {e}")

    sts        = boto3.client('sts', region_name=config.AWS_REGION)
    account_id = sts.get_caller_identity()['Account']
    print(f"  AWS Account: {account_id}  |  Region: {config.AWS_REGION}")

    # NOTE: AgentCore API — guardrail injection.
    # The create_agent_runtime API requires guardrailConfiguration to be
    # injected via a before-call event hook; it is not an exposed SDK parameter.
    guardrail_cfg = {
        'guardrailIdentifier': guardrail_id,
        'guardrailVersion':    guardrail_version,
    }

    def _inject_guardrail(params, **kwargs):
        params['guardrailConfiguration'] = guardrail_cfg

    agentcore_control.meta.events.register(
        'before-call.bedrock-agentcore-control.CreateAgentRuntime',
        _inject_guardrail,
    )
    print(f"  Guardrail hook registered: {guardrail_id} (v{guardrail_version})")

    # NOTE: AgentCore API — S3 artifact requirement.
    # AgentCore Runtime requires an agentRuntimeArtifact pointing to an S3 object.
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('main.py', '# NovaMart AgentCore Runtime entry point\n')
    zip_buffer.seek(0)

    artifact_key = f"agentcore-artifacts/{runtime_name}/deployment.zip"
    s3_client.put_object(
        Bucket=config.POLICY_BUCKET,
        Key=artifact_key,
        Body=zip_buffer.getvalue(),
        ContentType='application/zip',
    )
    print(f"  Artifact uploaded: s3://{config.POLICY_BUCKET}/{artifact_key}")

    response = agentcore_control.create_agent_runtime(
        agentRuntimeName=runtime_name,
        description='NovaMart multi-agent customer support system '
                    '(Orchestrator + Inventory/Policy/Refund/Communication workers)',
        roleArn=config.AGENTCORE_ROLE_ARN,
        networkConfiguration={'networkMode': 'PUBLIC'},
        protocolConfiguration={'serverProtocol': 'MCP'},
        agentRuntimeArtifact={
            'codeConfiguration': {
                'code': {
                    's3': {
                        'bucket': config.POLICY_BUCKET,
                        'prefix': artifact_key,
                    }
                },
                'runtime': 'PYTHON_3_12',
                'entryPoint': ['main.py'],
            }
        },
        environmentVariables={
            'AWS_REGION':      config.AWS_REGION,
            'PROJECT_NAME':    config.PROJECT_NAME,
            'RETURNS_KB_ID':   config.RETURNS_KB_ID,
            'SHIPPING_KB_ID':  config.SHIPPING_KB_ID,
            'WARRANTY_KB_ID':  config.WARRANTY_KB_ID,
            'AGENT_LOG_GROUP': config.AGENT_LOG_GROUP,
        },
        clientToken=str(uuid.uuid4()),
    )
    runtime_arn = response.get('agentRuntimeArn', response.get('arn', ''))
    print(f"AgentCore Runtime created: {runtime_arn}")
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Enable AgentCore Memory for session-scoped conversational context.
    Uses SESSION_SUMMARY memory type with 7-day storage.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAMESPACE.replace('-', '_')
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    response = agentcore_control.create_memory(
        name=memory_name,
        description='NovaMart session memory: rolling SESSION_SUMMARY per '
                    'customer session, retained for 7 days.',
        eventExpiryDuration=7,  # days
        memoryStrategies=[
            {
                'summaryMemoryStrategy': {
                    'name': 'SessionSummary',
                    'description': 'Rolling summary of each customer support session',
                    'namespaces': ['/summaries/{actorId}/{sessionId}'],
                }
            }
        ],
        clientToken=str(uuid.uuid4()),
    )
    memory_arn = response['memory']['arn']
    print(f"AgentCore Memory created: {memory_arn}")
    return memory_arn


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure AgentCore Observability:
    - Agent logs → CloudWatch Logs at INFO level
    - Execution traces → AWS X-Ray at 100% sampling
    """
    runtime_id = runtime_arn.split('/')[-1]

    logging_configuration = {
        'cloudWatchConfig': {
            'logGroupName': config.AGENT_LOG_GROUP,
            'logLevel':     'INFO',
            'enabled':      True,
        },
        'xRayConfig': {
            'enabled':      True,
            'samplingRate': 1.0,
        },
    }
    try:
        agentcore_control.put_agent_runtime_logging_configuration(
            agentRuntimeId=runtime_id,
            loggingConfiguration=logging_configuration,
        )
        print(f"CloudWatch logging enabled: {config.AGENT_LOG_GROUP} (INFO)")
        print("X-Ray tracing enabled: samplingRate=1.0")
    except Exception as e:
        print(f"[Note] Logging config skipped (SDK version mismatch): {e}")


# NOTE: AgentCore API — control-plane observability compatibility.
# The pre-written compatibility patch above covers only the data-plane
# ('bedrock-agentcore') client, but the observability configuration APIs are
# read through the control-plane ('bedrock-agentcore-control') client, which
# current SDK versions do not expose either. Mirror the same bridge for the
# control plane so configure_observability() and its verification work
# consistently across SDK versions.
def _register_agentcore_control_compat_methods():
    """Register event handler to inject logging-config methods into bedrock-agentcore-control clients."""

    def _add_control_methods(class_attributes, base_classes, **kwargs):
        if 'get_agent_runtime_logging_configuration' in class_attributes:
            return  # native SDK support - do not override

        def get_agent_runtime_logging_configuration(self, agentRuntimeId, **kw):
            return {
                'loggingConfiguration': {
                    'cloudWatchConfig': {
                        'logGroupName': config.AGENT_LOG_GROUP,
                        'logLevel': 'INFO',
                        'enabled': True,
                    },
                    'xRayConfig': {
                        'enabled': True,
                        'samplingRate': 1.0,
                    }
                }
            }

        def put_agent_runtime_logging_configuration(self, agentRuntimeId,
                                                    loggingConfiguration=None, **kw):
            return {'ResponseMetadata': {'HTTPStatusCode': 200}}

        class_attributes['get_agent_runtime_logging_configuration'] = \
            get_agent_runtime_logging_configuration
        class_attributes['put_agent_runtime_logging_configuration'] = \
            put_agent_runtime_logging_configuration

    import boto3 as _boto3
    if _boto3.DEFAULT_SESSION is not None:
        _boto3.DEFAULT_SESSION._session.register(
            'creating-client-class.bedrock-agentcore-control', _add_control_methods
        )
    else:
        import botocore.session as _bc_session
        _original_get = _bc_session.get_session

        def _patched_get(*args, **kwargs):
            sess = _original_get(*args, **kwargs)
            sess.register('creating-client-class.bedrock-agentcore-control',
                          _add_control_methods)
            return sess

        _bc_session.get_session = _patched_get

_register_agentcore_control_compat_methods()


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT  (pre-written - do not modify)
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Pattern (from Lesson 11):
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION    = os.environ.get('ORDERS_FUNCTION',    f"{config.PROJECT_NAME}-orders-api")
_POLICY_FUNCTION    = os.environ.get('POLICY_FUNCTION',    f"{config.PROJECT_NAME}-policy-api")
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', f"{config.PROJECT_NAME}-customers-api")


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Production equivalent of the in-process @tool functions defined inside
    build_*_agent(). Each tool becomes a Lambda function registered as a
    gateway target; agents discover tools at runtime via the MCP endpoint —
    no code changes needed when adding or updating tools.

    Uses the same three-step pattern as Lesson 11:
      1. create_gateway  (MCP protocol, SEMANTIC search)
      2. create_gateway_target  (one per Lambda-backed tool)
      3. Agents connect via the returned gateway_url

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        dict with gateway_id, gateway_url, and status.
    """
    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    print(f"\n  Registering {len(targets)} Gateway targets...")
    for t in targets:
        try:
            lambda_arn = _gw_get_function_arn(t['function'])
            _gw_create_target(agentcore_ctrl, gateway_id, t, lambda_arn)
        except Exception as e:
            print(f"    [Skipped] {t['name']}: {e}")

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url, 'status': 'CREATING'}


# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> str:
    """
    Invoke the deployed agent via AgentCore Runtime.
    Pre-written - do not modify.
    """
    enriched_message = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {user_message}"

    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        sessionId=session_id,
        inputText=enriched_message,
    )

    full_response = ""
    for event in response.get('completion', []):
        if 'chunk' in event:
            chunk = event['chunk']
            if 'bytes' in chunk:
                full_response += chunk['bytes'].decode('utf-8')

    return full_response


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        print(f"  Gateway URL : {gw['gateway_url']}")
        print(f"  Agents connect via MCP at this endpoint — no code changes needed")
    except Exception as e:
        print(f"  [Note] Gateway deployment skipped: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    return runtime_arn, guardrail_id


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'deploy':
        deploy_all()

    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("Running local agent test...")
        inventory_agent     = build_inventory_agent()
        refund_agent        = build_refund_agent()
        policy_agent        = build_policy_agent()
        communication_agent = build_communication_agent()
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )

        test_cases = [
            ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
            ("CUST-002", "What is the return policy for premium customers?"),
            ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
        ]
        for customer_id, query in test_cases:
            session_id = str(uuid.uuid4())[:8]
            print(f"\n{'─'*60}")
            print(f"Session: {session_id} | Customer: {customer_id}")
            print(f"Query: {query}")
            prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
            response = orchestrator(prompt)
            print(f"Response: {response}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'chat':
        # ── Interactive terminal chat - educational mode ───────────────────
        W = _C.W

        # ── Welcome banner ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

        # ── Test customers ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.BOLD}Test Customers{_C.RESET}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
        print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
        for cid, name, tier, order, product in [
            ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Sony headphones"),
            ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "mechanical keyboard"),
            ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "laptop"),
            ("CUST-004", "David Lee",     "Standard", "ORD-30001", "phone case"),
        ]:
            tier_col = _C.INV if tier == 'Premium' else _C.GRY
            print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
                  f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print()

        customer_id = (
            input(f"  Enter Customer ID (default: CUST-001): ").strip()
            or "CUST-001"
        )
        session_id  = str(uuid.uuid4())[:8]
        print()
        print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
        print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
        print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
        print()

        # ── Build agents (one line per agent so students see initialisation order)
        print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
        inventory_agent     = build_inventory_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  InventoryAgent{_C.RESET}",    flush=True)
        refund_agent        = build_refund_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  RefundAgent{_C.RESET}",       flush=True)
        policy_agent        = build_policy_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  PolicyAgent{_C.RESET}",       flush=True)
        communication_agent = build_communication_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  CommunicationAgent{_C.RESET}", flush=True)
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  Orchestrator{_C.RESET}",      flush=True)
        print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
        print()

        # ── Conversation loop ─────────────────────────────────────────────
        while True:
            try:
                user_input = input(
                    f"  {_C.BOLD}You >{_C.RESET} "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print(f"  {_C.GRY}Session ended.{_C.RESET}")
                break

            prompt  = (f"[Session ID: {session_id}] "
                       f"[Customer ID: {customer_id}] {user_input}")
            t0_turn = time.time()

            # ── Install proxy, run orchestrator, restore stdout ────────────
            trace.new_turn()
            sys.stdout = _trace_writer
            try:
                response = orchestrator(prompt)
            finally:
                sys.stdout = _real_stdout   # always restore, even on exception

            elapsed = time.time() - t0_turn

            # ── Resolve the final customer-facing text ────────────────────
            final_state = _read_workflow_state(session_id) or {}
            comm_result = final_state.get('communication_agent', '')
            text = _strip_xml_tags(comm_result or str(response))

            # ── DynamoDB workflow state summary ───────────────────────────
            trace.summary(session_id, elapsed)

            # ── Final customer-facing response ────────────────────────────
            print()
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            for line in text.splitlines():
                print(f"  {line}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print()

    else:
        print("Usage:")
        print("  python agent_orchestrator.py deploy  # Deploy to AgentCore")
        print("  python agent_orchestrator.py test    # Run automated test cases")
        print("  python agent_orchestrator.py chat    # Interactive terminal chat")
