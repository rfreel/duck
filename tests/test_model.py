from duck_harness.model import CheckResult, Status, aggregate_status


def check(status: Status, required: bool = True) -> CheckResult:
    return CheckResult("x", status, required, status.value)


def test_required_passes_aggregate_to_pass():
    assert aggregate_status((check(Status.PASS), check(Status.PASS))) is Status.PASS


def test_required_fail_dominates_unknown():
    assert aggregate_status((check(Status.UNKNOWN), check(Status.FAIL))) is Status.FAIL


def test_required_unknown_is_preserved():
    assert aggregate_status((check(Status.PASS), check(Status.UNKNOWN))) is Status.UNKNOWN


def test_optional_skip_does_not_block_pass():
    assert aggregate_status((check(Status.PASS), check(Status.SKIP, required=False))) is Status.PASS


def test_required_skip_is_not_silently_passed():
    assert aggregate_status((check(Status.SKIP),)) is Status.UNKNOWN
