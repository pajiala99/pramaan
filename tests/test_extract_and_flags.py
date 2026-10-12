"""extract_entities and detect_red_flags - pure, deterministic text analysis."""
import pytest

from pramaan.case import JobOfferCase
from pramaan.tools.detect_red_flags import scan
from pramaan.tools.extract_entities import extract


def signals(text_):
    return {h["signal"]: h["severity"] for h in scan(text_)}


def test_email_and_upi_are_told_apart():
    e = extract(JobOfferCase(message="Mail hr@acme-analytics.in or pay to payhr@ybl. Also hr.team@gmail.com"))
    assert [x["address"] for x in e["emails"]] == ["hr@acme-analytics.in", "hr.team@gmail.com"]
    assert e["upi_ids"] == ["payhr@ybl"]
    assert e["emails"][1]["free_provider"] is True


@pytest.mark.parametrize("raw", ["+91 98765 43210", "09876543210", "98765-43210", "+91-9876543210"])
def test_indian_phone_numbers_are_normalized(raw):
    assert extract(JobOfferCase(message=f"Call {raw} today"))["phones"] == ["+919876543210"]


def test_upi_handle_is_not_mistaken_for_a_domain():
    e = extract(JobOfferCase(message="Pay Rs. 499 to amazonhr.jobs@ybl"))
    assert "amazonhr.jobs" not in e["domains"]
    assert e["payment_asks"] == ["Pay Rs. 499 to amazonhr.jobs@ybl"]


def test_job_boards_forms_and_free_mail_are_not_employer_domains():
    e = extract(JobOfferCase(
        message="Apply https://forms.gle/abc or https://www.linkedin.com/jobs/1 or visit https://careers.acme-corp.in/apply",
        recruiter_email="x@gmail.com",
    ))
    assert e["domains"] == ["acme-corp.in"]


def test_scam_message_flags():
    s = signals(
        "No interview required. Pay the refundable registration fee of Rs. 499 to UPI fees@ybl within 24 hours. "
        "Contact on WhatsApp only. Earn Rs 3000 per day."
    )
    assert s["PAYMENT_REQUEST"] == "high"
    assert s["UPI_ID_IN_OFFER"] == "high"
    assert s["NO_INTERVIEW"] == "medium"
    assert s["CHAT_APP_ONLY"] == "medium"
    assert s["UNREALISTIC_PAY"] == "medium"
    assert s["URGENCY_PRESSURE"] == "low"


def test_negated_fee_disclaimer_is_not_a_red_flag():
    s = signals("Please note: Infosys does not charge any registration fee at any stage of recruitment.")
    assert "PAYMENT_REQUEST" not in s


def test_dont_worry_is_not_a_negation():
    assert "PAYMENT_REQUEST" in signals("Don't worry, the registration fee is fully refundable.")


def test_paid_training_is_not_a_fee():
    s = signals("You will join a 3-month paid training program at our Pune campus. Salary INR 30,000 per month.")
    assert not any(sev in ("medium", "high") for sev in s.values())


def test_task_scam_pattern():
    assert signals("Earn by liking YouTube videos, Rs 50 per like")["TASK_BASED_EARNING"] == "medium"
