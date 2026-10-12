"""Evidence groups for recruitment-scam investigations.

Each group answers a different question from a different source of truth.
Facts in different groups are treated as *independent*; facts in the same
group corroborate nothing (they come from one observation).

    message_content   What does the offer itself ask for?     (the submitted text)
    sender_identity   Is the sender who they claim to be?     (email/domain vs. the company)
    domain_registry   Is the website infrastructure real?    (RDAP/WHOIS registry, TLS)
    official_listing  Does the company advertise this job?   (the company's own careers page)
    public_reports    Have other people reported it?          (third-party complaints on the web)
"""

MESSAGE = "message_content"
IDENTITY = "sender_identity"
REGISTRY = "domain_registry"
LISTING = "official_listing"
REPORTS = "public_reports"

ALL_GROUPS = (MESSAGE, IDENTITY, REGISTRY, LISTING, REPORTS)

GROUP_LABELS = {
    MESSAGE: "Message content - what the offer asks for",
    IDENTITY: "Sender identity - email and domain vs. the company",
    REGISTRY: "Domain registry - age, registration, TLS",
    LISTING: "Official listing - the company's own careers page",
    REPORTS: "Public reports - complaints found on the web",
}
