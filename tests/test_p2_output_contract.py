import pytest

from migrationproof.p2_output_contract import OutputContractError, parse_model_output


VALID = '{"decision":"fail","issue_type":"accuracy_gate_failed","evidence":["accuracy=0.44"],"explanation":"Accuracy is below the gate."}'


def test_parse_exact_four_field_json():
    parsed = parse_model_output(VALID)
    assert parsed.decision == "fail"
    assert parsed.issue_type == "accuracy_gate_failed"
    assert parsed.evidence == ("accuracy=0.44",)


@pytest.mark.parametrize("text", [
    "prefix " + VALID,
    VALID + " suffix",
    '{"decision":"fail","issue_type":"accuracy_gate_failed","evidence":"accuracy=0.44","explanation":"bad"}',
    '{"decision":"unknown","issue_type":"accuracy_gate_failed","evidence":["accuracy=0.44"],"explanation":"bad"}',
    '{"decision":"fail","issue_type":"accuracy_gate_failed","evidence":["accuracy=0.44"]}',
])
def test_rejects_nonconforming_output(text):
    with pytest.raises(OutputContractError):
        parse_model_output(text)
