from tests.helpers import make_settings


def test_model_for_uses_override_when_set():
    settings = make_settings(how_to_model="small-model", classifier_model="tiny-model")

    assert settings.model_for("how_to") == "small-model"
    assert settings.model_for("classifier") == "tiny-model"


def test_model_for_falls_back_to_default():
    settings = make_settings()

    assert settings.model_for("troubleshooting") == "default-model"
    assert settings.model_for("account_access") == "default-model"


def test_blank_override_falls_back_to_default():
    settings = make_settings(how_to_model="   ")

    assert settings.how_to_model is None
    assert settings.model_for("how_to") == "default-model"


def test_unknown_step_falls_back_to_default():
    assert make_settings().model_for("human_escalation") == "default-model"
