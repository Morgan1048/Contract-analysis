import yaml
from openapi_spec_validator import validate

from core.openapi_schema import analysis_result_schema, compliance_question_schema


def _openapi_document() -> dict:
    with open("openapi.yaml", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    assert isinstance(document, dict)
    return document


def _resolve_local_refs(value, document):
    if isinstance(value, dict):
        if set(value) == {"$ref"} and value["$ref"].startswith("#/"):
            target = document
            for part in value["$ref"][2:].split("/"):
                target = target[part]
            return _resolve_local_refs(target, document)
        return {key: _resolve_local_refs(item, document) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_local_refs(item, document) for item in value]
    return value


def test_openapi_source_declares_openapi_31_and_assignment_fields():
    document = _openapi_document()
    validate(document)
    assert document["openapi"] == "3.1.0"
    source = open("openapi.yaml", encoding="utf-8").read()
    for field in ("Compliance Question", "Compliance State", "Confidence", "Relevant Quotes", "Rationale"):
        assert field in source


def test_generated_schemas_use_pipeline_dataclasses():
    question = compliance_question_schema()
    result = analysis_result_schema()
    assert set(question["required"]) == {
        "Compliance Question", "Compliance State", "Confidence", "Relevant Quotes", "Rationale"
    }
    assert result["properties"]["questions"]["items"] == question


def test_checked_in_openapi_schemas_match_generated_dataclass_schemas():
    document = _openapi_document()
    schemas = document["components"]["schemas"]
    assert _resolve_local_refs(schemas["ComplianceQuestion"], document) == compliance_question_schema()
    assert _resolve_local_refs(schemas["AnalysisResult"], document) == analysis_result_schema()


def test_openapi_declares_tenant_api_key_and_operation_scopes():
    document = _openapi_document()
    scheme = document["components"]["securitySchemes"]["TenantApiKey"]
    assert scheme["type"] == "apiKey"
    assert scheme["in"] == "header"
    assert scheme["name"] == "X-API-Key"
    assert document["security"] == [{"TenantApiKey": []}]
    for path, method, scope in [
        ("/analyze", "post", "runs:write"),
        ("/runs/{run_id}", "get", "runs:read"),
        ("/runs", "get", "runs:read"),
        ("/search", "post", "contracts:search"),
        ("/chat", "post", "chat:write"),
    ]:
        operation = document["paths"][path][method]
        assert operation.get("security", document["security"]) == [{"TenantApiKey": []}]
        assert operation["x-required-scopes"] == [scope]
    validate(document)
