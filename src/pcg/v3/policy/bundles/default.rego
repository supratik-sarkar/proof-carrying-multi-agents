package pcg

default allow = false

allowed_tools := {"search", "retrieve", "calculator", "sql_read", "http_get_allowlisted"}
allowed_delegations := {"prover", "verifier", "retriever"}

allow {
    input.actor
    input.action
    not tool_denied
    not delegation_denied
    not verifier_context_shared
}

tool_denied { input.tool; not allowed_tools[input.tool] }
delegation_denied { input.delegate_to; not allowed_delegations[input.delegate_to] }
verifier_context_shared { input.verifier_context_shared == true }
