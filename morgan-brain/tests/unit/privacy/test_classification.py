"""Unit tests for morgan_brain.privacy.classification.

Coverage:
- Explicit override always wins.
- Secret-tier patterns: API keys, tokens, passwords, PEM headers, AWS keys.
- Sensitive-tier: SSN, credit-card-shaped numbers, health/financial/credential keywords.
- Personal-tier: email addresses, phone numbers.
- Public: plain prose with no identifiable signals.
- Ordering: DataClass comparison operators.
"""

from __future__ import annotations

from morgan_brain.privacy.classification import DataClass, classify


# ---------------------------------------------------------------------------
# Explicit override
# ---------------------------------------------------------------------------


class TestExplicitOverride:
    def test_explicit_public_overrides_secret_text(self) -> None:
        secret_text = "my api_key = sk-abcdefghijklmnopqrstuvwxyz1234567890"
        assert classify(secret_text, explicit=DataClass.PUBLIC) is DataClass.PUBLIC

    def test_explicit_secret_overrides_plain_text(self) -> None:
        assert classify("the weather is fine", explicit=DataClass.SECRET) is DataClass.SECRET

    def test_explicit_personal(self) -> None:
        assert classify("hello world", explicit=DataClass.PERSONAL) is DataClass.PERSONAL

    def test_explicit_sensitive(self) -> None:
        assert classify("hello world", explicit=DataClass.SENSITIVE) is DataClass.SENSITIVE

    def test_explicit_none_falls_through_to_heuristic(self) -> None:
        # plain text → PUBLIC when explicit=None
        result = classify("the weather is nice today", explicit=None)
        assert result is DataClass.PUBLIC


# ---------------------------------------------------------------------------
# SECRET tier
# ---------------------------------------------------------------------------


class TestSecretTier:
    def test_sk_prefixed_api_key(self) -> None:
        assert classify("Authorization: sk-abcdefghijklmnopqrstuvwxyzABCDEF") is DataClass.SECRET

    def test_ghp_token(self) -> None:
        assert classify("token: ghp_abcdefghijklmnopqrstuvwxyz1234") is DataClass.SECRET

    def test_aws_akia_key(self) -> None:
        assert classify("AKIAIOSFODNN7EXAMPLE is the aws key") is DataClass.SECRET

    def test_pem_private_key_header(self) -> None:
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA..."
        assert classify(text) is DataClass.SECRET

    def test_password_label_with_long_value(self) -> None:
        text = "password = HunterTwoLongEnoughToTrigger1234567890"
        assert classify(text) is DataClass.SECRET

    def test_bearer_token(self) -> None:
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"
        assert classify(text) is DataClass.SECRET

    def test_openssh_private_key(self) -> None:
        text = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC..."
        assert classify(text) is DataClass.SECRET

    def test_api_key_equals_long_value(self) -> None:
        text = "api_key=abcdefghijklmnopqrstuvwxyz1234567890"
        assert classify(text) is DataClass.SECRET


# ---------------------------------------------------------------------------
# SENSITIVE tier (must be >= SENSITIVE, but text must NOT match SECRET)
# ---------------------------------------------------------------------------


class TestSensitiveTier:
    def test_ssn_pattern(self) -> None:
        result = classify("SSN: 123-45-6789")
        assert result >= DataClass.SENSITIVE

    def test_financial_keyword(self) -> None:
        result = classify("my salary is competitive")
        assert result >= DataClass.SENSITIVE

    def test_health_keyword(self) -> None:
        result = classify("my diagnosis was anxiety disorder")
        assert result >= DataClass.SENSITIVE

    def test_credential_keyword_without_value(self) -> None:
        # "credential" keyword → SENSITIVE (no long value → not SECRET)
        result = classify("please provide your credential")
        assert result >= DataClass.SENSITIVE

    def test_bank_account_keyword(self) -> None:
        result = classify("bank account details follow")
        assert result >= DataClass.SENSITIVE


# ---------------------------------------------------------------------------
# PERSONAL tier (must be >= PERSONAL, but not SENSITIVE or SECRET)
# ---------------------------------------------------------------------------


class TestPersonalTier:
    def test_email_address(self) -> None:
        result = classify("contact me at user@example.com for details")
        assert result >= DataClass.PERSONAL

    def test_phone_number_us(self) -> None:
        result = classify("call me at 555-123-4567 anytime")
        assert result >= DataClass.PERSONAL

    def test_international_phone(self) -> None:
        result = classify("reach me on +1 415 555 0199")
        assert result >= DataClass.PERSONAL

    def test_email_only_is_personal_not_sensitive(self) -> None:
        result = classify("send to alice@example.org")
        # Should be PERSONAL; no financial/health keyword, so must NOT be SENSITIVE.
        assert result is DataClass.PERSONAL

    def test_phone_only_is_personal_not_sensitive(self) -> None:
        result = classify("dial 415-555-0100")
        assert result is DataClass.PERSONAL


# ---------------------------------------------------------------------------
# PUBLIC tier
# ---------------------------------------------------------------------------


class TestPublicTier:
    def test_plain_prose(self) -> None:
        assert classify("the quick brown fox jumps over the lazy dog") is DataClass.PUBLIC

    def test_empty_string(self) -> None:
        assert classify("") is DataClass.PUBLIC

    def test_numbers_without_pii_shape(self) -> None:
        # Random numbers not shaped like SSN/card/phone → PUBLIC
        assert classify("the answer is 42 and Pi is 3.14159") is DataClass.PUBLIC

    def test_technical_prose(self) -> None:
        assert classify("we deploy microservices on Kubernetes") is DataClass.PUBLIC


# ---------------------------------------------------------------------------
# Ordering / comparison operators
# ---------------------------------------------------------------------------


class TestOrdering:
    def test_public_lt_personal(self) -> None:
        assert DataClass.PUBLIC < DataClass.PERSONAL

    def test_personal_lt_sensitive(self) -> None:
        assert DataClass.PERSONAL < DataClass.SENSITIVE

    def test_sensitive_lt_secret(self) -> None:
        assert DataClass.SENSITIVE < DataClass.SECRET

    def test_secret_ge_all(self) -> None:
        for tier in DataClass:
            assert DataClass.SECRET >= tier

    def test_public_le_all(self) -> None:
        for tier in DataClass:
            assert DataClass.PUBLIC <= tier

    def test_equal(self) -> None:
        assert DataClass.PERSONAL <= DataClass.PERSONAL
        assert DataClass.PERSONAL >= DataClass.PERSONAL

    def test_not_lt_wrong_type(self) -> None:
        result = DataClass.PUBLIC.__lt__("PUBLIC")
        assert result is NotImplemented
