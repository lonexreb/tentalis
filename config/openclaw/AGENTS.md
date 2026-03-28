# Agent Registry

Two agents operate in the Tentalis system:

## manager-01

- **Role**: Task decomposition, assignment, evaluation, and feedback
- **Config**: `config/openclaw/manager/`
- **Skills**: `assign-task`, `submit-feedback`

## worker-01

- **Role**: Step-by-step task solving using LLM inference
- **Config**: `config/openclaw/worker/`
- **Skills**: `submit-result`
