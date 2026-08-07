"""Build the development and test question sets.

The questions are declared here rather than hand-edited in JSON, for three
reasons. Source is reviewable in a way a 2,000-line JSON blob is not; the
grouping and split rules are applied by construction instead of by hand; and
regenerating after a corpus change is one command rather than a careful edit.

Run:

    python scripts/build_question_set.py            # writes both splits
    python scripts/build_question_set.py --check    # validates without writing

Split rules, enforced in ``sme_assistant.evaluation.question_set``:

* every paraphrase of a conflict family stays with its family
* reported families (CONF-*) appear only in the test split
* tuning families (TUNE-*) appear only in the development split
* no test question may expect a chunk that carries a tuning family's disputed
  fact, so that tuning cannot leak an answer into a reported result
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.evaluation.config import load_evaluation_config  # noqa: E402
from sme_assistant.evaluation.conflicts import load_conflicts  # noqa: E402
from sme_assistant.evaluation.question_set import (  # noqa: E402
    Question,
    QuestionSet,
    QuestionSetError,
    validate_question_set,
    write_question_set,
)
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402

# --- conflict questions ------------------------------------------------------
# Three paraphrases per family. The third is deliberately the most natural way
# a person would actually ask it, because a question set written entirely in
# policy vocabulary would flatter retrieval.

CONFLICTS: dict[str, dict] = {
    "CONF-01": {
        "risk": "medium",
        "cite": ["HR-13#001", "HR-13#004"],
        "not": ["HR-03"],
        "gold": (
            "55 pence per mile for the first 10,000 business miles in a tax year and "
            "25 pence per mile thereafter, under HR-13. HR-03, which stated a flat 40 "
            "pence, was withdrawn and superseded by HR-13."
        ),
        "facts": ["55 pence", "25 pence", "10,000"],
        "questions": [
            "What is the mileage rate for business travel?",
            "How much can I claim per mile for using my own car on company business?",
            "I drove 200 miles to see a customer last week. How much do I get back?",
        ],
    },
    "CONF-02": {
        "risk": "high",
        "cite": ["HR-12#002"],
        "not": ["HR-02"],
        "gold": (
            "Statutory Sick Pay is payable from the first qualifying day with no waiting "
            "days, and all employees qualify regardless of earnings, under HR-12. HR-02, "
            "which required three unpaid waiting days and earnings at or above the Lower "
            "Earnings Limit, was withdrawn."
        ),
        "facts": ["first qualifying day"],
        "questions": [
            "When does Statutory Sick Pay start being paid?",
            "Are there unpaid waiting days before sick pay begins?",
            "I have been off sick since Monday. Which days will I actually be paid for?",
        ],
    },
    "CONF-03": {
        "risk": "medium",
        "cite": ["IT-11#001", "IT-11#002"],
        "not": ["IT-01"],
        "gold": (
            "Passwords must be at least 14 characters, three random words are encouraged, "
            "and scheduled rotation has been discontinued for standard accounts, under "
            "IT-11. IT-01, which required 8 characters and a 90-day change, was withdrawn. "
            "Codes sent by SMS are no longer accepted as a second factor."
        ),
        "facts": ["14 characters"],
        "questions": [
            "What is the minimum password length?",
            "How often do I have to change my password?",
            "Can I use a code texted to my phone as my second factor?",
        ],
    },
    "CONF-04": {
        "risk": "low",
        "cite": ["CS-11#002"],
        "not": ["CS-01"],
        "gold": (
            "Standard delivery is £6.95 excluding VAT, free on orders of £75 and over, "
            "and next working day is £12.95, under CS-11. CS-01, which stated £4.95, a £50 "
            "free-delivery threshold and £9.95 next day, was superseded on 1 February 2026."
        ),
        "facts": ["£6.95", "£75", "£12.95"],
        "questions": [
            "How much is standard delivery?",
            "How much do I need to spend to get free delivery?",
            "What does it cost to get something delivered tomorrow?",
        ],
    },
    "CONF-05": {
        "risk": "high",
        "cite": ["GEN-04#003", "IT-03#002"],
        "not": [],
        "gold": (
            "The documents disagree and both are current. IT-03 requires a lost or stolen "
            "device to be reported within 1 hour of becoming aware; GEN-04 allows 24 hours. "
            "Neither supersedes the other. The safe course is to report within 1 hour, but "
            "the discrepancy should be escalated rather than resolved by the assistant."
        ),
        "facts": ["1 hour", "24 hours"],
        "questions": [
            "How quickly must I report a lost company laptop?",
            "My work phone was stolen last night. What is the deadline for reporting it?",
            "What is the time limit for reporting lost or stolen company equipment?",
        ],
    },
    "CONF-06": {
        "risk": "medium",
        "cite": ["IT-04#002", "REG-02#003"],
        "not": [],
        "gold": (
            "The documents disagree and both are current. IT-04 retains an annual backup of "
            "all business systems for seven years; REG-02 requires records to be deleted at "
            "the end of their retention period, including from backups at the next rollover. "
            "Neither supersedes the other and the conflict has data protection implications, "
            "so it should be escalated."
        ),
        "facts": ["seven years"],
        "questions": [
            "How long is data kept in backups after its retention period ends?",
            "If a record reaches the end of its retention period, is it removed from backups too?",
            "We are asked to delete a customer's data. Does that include the backups?",
        ],
    },
    "CONF-07": {
        "risk": "low",
        "cite": ["GEN-04#002", "FIN-03#002"],
        "not": [],
        "gold": (
            "The documents disagree and both are current. GEN-04 requires Finance Manager "
            "approval because the cost exceeds £750; FIN-03 places £501 to £2,500 with the "
            "department head. Neither supersedes the other, so both approvers should be "
            "named and the discrepancy escalated."
        ),
        "facts": ["£750", "£2,500"],
        "questions": [
            "Who approves an £800 laptop purchase?",
            "I need to buy IT equipment costing £800. Whose sign-off do I need?",
            "What is the approval route for an IT purchase of eight hundred pounds?",
        ],
    },
    "CONF-08": {
        "risk": "low",
        "cite": ["GEN-02#001", "CS-11#003"],
        "not": [],
        "gold": (
            "The documents disagree and both are current. GEN-02 gives the trade counter "
            "hours as Monday to Friday 08:00 to 17:00; CS-11 gives 08:30 to 16:30. Neither "
            "supersedes the other, so both should be reported and the discrepancy flagged."
        ),
        "facts": ["08:00", "17:00", "08:30", "16:30"],
        "questions": [
            "What are the trade counter opening hours?",
            "Can I collect an order at 16:45 on a Wednesday?",
            "What time does the trade counter open in the morning?",
        ],
    },
    "CONF-09": {
        "risk": "medium",
        "cite": ["IT-11#003", "IT-02#003"],
        "not": [],
        "gold": (
            "The documents disagree and both are current. IT-11 requires access rights to be "
            "reviewed every six months by each department head; IT-02 states access to "
            "systems holding personal data is reviewed annually by the Data Protection Lead. "
            "Neither supersedes the other, so both should be surfaced."
        ),
        "facts": ["six months"],
        "questions": [
            "How often are user access rights reviewed?",
            "Who reviews access to systems holding personal data, and how often?",
            "What is the review cycle for system access permissions?",
        ],
    },
    "TUNE-01": {
        "risk": "low",
        "cite": ["OPS-02#001", "CS-03#001"],
        "not": [],
        "gold": (
            "The documents disagree and both are current. OPS-02 allows return within 30 days "
            "of delivery; CS-03 states 28 days. Neither supersedes the other. The statutory "
            "14-day cancellation right under the Consumer Contracts Regulations 2013 is "
            "separate and unaffected."
        ),
        "facts": ["30 days", "28 days"],
        "questions": [
            "How long do I have to return an unwanted item?",
            "What is the returns window for goods I no longer want?",
            "My order arrived three weeks ago and I want to send it back. Am I still in time?",
        ],
    },
    "TUNE-02": {
        "risk": "medium",
        "cite": ["OPS-08#004", "REG-01#005"],
        "not": [],
        "gold": (
            "The documents disagree and both are current. OPS-08 requires the Health and "
            "Safety Coordinator to review every accident book entry within two working days; "
            "REG-01 states five working days. Neither supersedes the other."
        ),
        "facts": ["two working days", "five working days"],
        "questions": [
            "How long does the Health and Safety Coordinator have to review an accident book entry?",
            "After an accident is written in the accident book, when is it looked at?",
            "What is the timescale for reviewing accident records?",
        ],
    },
}

SUPERSESSION = {"CONF-01", "CONF-02", "CONF-03", "CONF-04"}

# --- unanswerable questions --------------------------------------------------
# Grouped by gap topic: two probes of one absence are one observation.

GAPS: dict[str, dict] = {
    "pensions and auto-enrolment": {
        "split": "dev",
        "questions": [
            "What is the company pension scheme?",
            "What percentage does the company contribute to my pension?",
        ],
    },
    "company car and vehicle allowance schemes": {
        "split": "dev",
        "questions": [
            "Am I entitled to a company car?",
            "What is the car allowance for department heads?",
        ],
    },
    "export documentation procedure": {
        "split": "dev",
        "questions": [
            "What export documentation is needed for a shipment to Ireland?",
            "Which commodity codes do we use on a commercial invoice?",
        ],
    },
    "maternity, paternity and shared parental leave": {
        "split": "test",
        "questions": [
            "How much maternity leave am I entitled to?",
            "Can I take shared parental leave?",
        ],
    },
    "redundancy and notice periods": {
        "split": "test",
        "questions": [
            "What is my notice period?",
            "How is redundancy pay calculated?",
        ],
    },
    "share options and bonus schemes": {
        "split": "test",
        "questions": [
            "Is there an annual bonus scheme?",
            "Do employees receive share options?",
        ],
    },
    "flexible working requests": {
        "split": "test",
        "questions": [
            "How do I make a flexible working request?",
            "How long does the company have to respond to a flexible working request?",
        ],
    },
    "staff purchase and employee discount": {
        "split": "test",
        "questions": [
            "What staff discount do employees get?",
            "Can I buy products at cost price as an employee?",
        ],
    },
}

# --- partially answerable ----------------------------------------------------
# The hardest abstention case: relevant-looking evidence that does not answer.

PARTIALS: dict[str, dict] = {
    "probationary period": {
        "split": "dev",
        "documents": ["HR-04"],
        "gold": (
            "The corpus does not state the length of the probationary period. HR-04 refers to "
            "it, saying new employees work on site for its duration, but no document defines "
            "how long it lasts or what happens at the end of it. The reference should be "
            "reported and the missing detail acknowledged."
        ),
        "questions": [
            "How long is the probationary period?",
            "What happens at the end of probation?",
        ],
    },
    "grievance procedure": {
        "split": "test",
        "documents": ["HR-05"],
        "gold": (
            "The corpus does not document a grievance procedure. HR-05 names the ACAS Code of "
            "Practice on disciplinary and grievance procedures, so the word appears and HR-05 "
            "will be retrieved, but it sets out the disciplinary process only. The near miss "
            "should be reported and the gap acknowledged rather than filled."
        ),
        "questions": [
            "How do I raise a grievance?",
            "What are the stages of the grievance procedure?",
        ],
    },
}

# --- ordinary factual questions ----------------------------------------------
# One group each: nothing paraphrases anything, so nothing is correlated.
# Development and test draw on disjoint documents wherever possible.

FACTUAL: list[dict] = [
    # development
    {"split": "dev", "id": "leave-entitlement", "q": "How many days of annual leave do full-time employees get?",
     "chunks": ["HR-01#001"], "facts": ["25 days", "8"],
     "gold": "25 days of paid annual leave per leave year, in addition to the 8 English bank holidays. Part-time employees receive a pro-rata entitlement."},
    {"split": "dev", "id": "fire-assembly", "q": "Where is the fire assembly point?",
     "chunks": ["OPS-07#003"], "facts": [],
     "gold": "The north corner of the staff car park, beside the blue container. The marked pedestrian route around the outside of the building is used, not the goods-in bay."},
    {"split": "dev", "id": "petty-cash-limit", "q": "What is the maximum single petty cash transaction?",
     "chunks": ["FIN-02#002"], "facts": ["£50"],
     "gold": "£50. The float is £250, held in the locked cash box in the Finance office."},
    {"split": "dev", "id": "warehouse-speed", "q": "What is the speed limit inside the warehouse?",
     "chunks": ["OPS-05#004"], "facts": ["4"],
     "gold": "4 km/h inside the building and 8 km/h in the yard."},
    {"split": "dev", "id": "knives-dishwasher", "q": "Can I put my knives in the dishwasher?",
     "chunks": ["PRD-01#001"], "facts": [],
     "gold": "No. Knives are washed by hand in warm soapy water and dried immediately. Dishwasher detergent is abrasive and dulls the edge, heat can loosen or crack handle scales, and blades knock against other items."},
    {"split": "dev", "id": "remote-days", "q": "How many days a week can I work from home?",
     "chunks": ["HR-04#002"], "facts": ["two days"],
     "gold": "Up to two days per week for eligible office-based roles. Monday is a core on-site day for all office staff."},
    # test
    {"split": "test", "id": "damaged-report-window", "q": "How quickly must I report a damaged delivery?",
     "chunks": ["CS-02#001"], "facts": ["48 hours"],
     "gold": "Within 48 hours of delivery, to customer services, with the order number and photographs of the damage and the outer packaging."},
    {"split": "test", "id": "contractor-insurance", "q": "What must a contractor provide before their first visit?",
     "chunks": ["GEN-03#003"], "facts": [],
     "gold": "A valid public liability insurance certificate, a site induction with the Facilities Coordinator, and a risk assessment and method statement for the work."},
    {"split": "test", "id": "forklift-authorisation", "q": "Who is allowed to drive a forklift?",
     "chunks": ["OPS-06#001"], "facts": [],
     "gold": "Only employees holding a current RTITB or equivalent accredited certificate for the relevant truck category who also appear on the site authorised operator list. Authorisation is site-specific."},
    {"split": "test", "id": "payment-terms", "q": "What are the standard payment terms?",
     "chunks": ["FIN-01#001"], "facts": ["30 days"],
     "gold": "30 days from the date of invoice. Invoices are raised automatically on despatch and emailed to the address held on the customer account."},
    {"split": "test", "id": "account-hold", "q": "At how many days overdue is an account placed on hold?",
     "chunks": ["FIN-01#003"], "facts": ["30"],
     "gold": "30 days overdue, by formal letter. A reminder email goes at 7 days and a credit control telephone call at 14 days."},
    {"split": "test", "id": "refund-method", "q": "How is a refund paid back to a customer?",
     "chunks": ["OPS-03#003"], "facts": [],
     "gold": "To the original payment method. Card refunds are processed by Finance through the payment gateway and typically reach the customer three to five working days later, which is outside the company's control."},
    {"split": "test", "id": "development-conversation", "q": "When does the annual development conversation happen?",
     "chunks": ["HR-06#002"], "facts": [],
     "gold": "In April, with the employee's line manager, following the performance review cycle."},
    {"split": "test", "id": "forklift-preuse", "q": "What has to be checked on a forklift before a shift?",
     "chunks": ["OPS-06#002"], "facts": [],
     "gold": "A documented pre-use check covering tyres, forks and mast for damage, hydraulic leaks, brakes including the parking brake, horn, lights and beacon, and the battery."},
]

# --- multi-document synthesis ------------------------------------------------
# Answerable, but only by combining documents that do not contradict.

SYNTHESIS: list[dict] = [
    {"split": "dev", "id": "new-starter-week-one",
     "q": "I start on Monday in an office role. What training and equipment should I expect in my first week?",
     "chunks": ["HR-06#001", "HR-04#004"], "documents": ["HR-06", "HR-04"],
     "gold": "A corporate induction on the first day covering company overview, health and safety, fire procedures and data protection. The company provides a laptop and headset. It does not reimburse home broadband, heating or electricity, and a second monitor for home use is provided to department heads only."},
    {"split": "dev", "id": "visitor-evacuation",
     "q": "A visitor is on site when the fire alarm sounds. What happens to them?",
     "chunks": ["GEN-03#002", "OPS-07#003", "OPS-07#004"], "documents": ["GEN-03", "OPS-07"],
     "gold": "Visitors sign in at reception and are accompanied by a member of staff, so their host takes them to the assembly point at the north corner of the staff car park. Fire wardens take a roll call using the daily attendance list and the visitor book from reception, so the visitor is accounted for from the book."},
    {"split": "test", "id": "damaged-order-end-to-end",
     "q": "A customer's order arrived damaged and they want their money back. What happens from the moment they call?",
     "chunks": ["CS-02#001", "CS-02#003", "OPS-03#001"], "documents": ["CS-02", "OPS-03"],
     "gold": "The customer reports the damage within 48 hours with photographs. Customer services raise a damage report and offer either a replacement despatched within one working day of the report being accepted, or a refund processed under OPS-03. Damaged items are collected by the carrier."},
    {"split": "test", "id": "new-trade-customer",
     "q": "A new trade customer wants a credit account and next working day delivery. What do they need to do?",
     "chunks": ["FIN-01#002", "CS-11#002"], "documents": ["FIN-01", "CS-11"],
     "gold": "They apply for a credit account using the account application form. Finance obtains a credit reference and two trade references before opening the account, with an opening limit typically £2,500. Next working day delivery costs £12.95 excluding VAT and requires the order to be placed and paid before 14:00, Monday to Thursday."},
    {"split": "test", "id": "faulty-warranty-return",
     "q": "An item comes back faulty and still under warranty. Who deals with it and what happens?",
     "chunks": ["OPS-02#002", "CS-03#003"], "documents": ["OPS-02", "CS-03"],
     "gold": "A returns operative inspects the item within two working days of receipt and assigns disposition D, which is photographed and routed to a supplier claim under CS-03. The customer claims with the order number, a description of the fault and photographs; assessment normally takes five working days from receipt."},
]


def build() -> QuestionSet:
    questions: list[Question] = []

    for family, spec in CONFLICTS.items():
        tuning = family.startswith("TUNE-")
        behaviour = "cite_current_only" if family in SUPERSESSION else "surface_both_and_qualify"
        for position, text in enumerate(spec["questions"], start=1):
            first = f"{family}-Q1"
            questions.append(Question(
                question_id=f"{family}-Q{position}",
                text=text,
                category="conflict",
                group_id=family,
                split="dev" if tuning else "test",
                answerability="answerable",
                expected_behaviour=behaviour,
                risk_level=spec["risk"],
                family_id=family,
                paraphrase_of=None if position == 1 else first,
                gold_answer=spec["gold"],
                gold_facts=tuple(spec["facts"]),
                expected_chunks=tuple(spec["cite"]),
                expected_documents=tuple(sorted({c.split("#")[0] for c in spec["cite"]})),
                must_not_cite=tuple(spec["not"]),
            ))

    for topic, spec in GAPS.items():
        slug = topic.split(" ")[0].strip(",").lower()
        for position, text in enumerate(spec["questions"], start=1):
            questions.append(Question(
                question_id=f"GAP-{slug}-Q{position}",
                text=text,
                category="unanswerable",
                group_id=topic,
                split=spec["split"],
                answerability="unanswerable",
                expected_behaviour="abstain",
                risk_level="medium",
                gap_topic=topic,
                paraphrase_of=None if position == 1 else f"GAP-{slug}-Q1",
                gold_answer="",
                notes="The corpus contains nothing on this topic. Any specific figure is invented.",
            ))

    for topic, spec in PARTIALS.items():
        slug = topic.split(" ")[0].lower()
        for position, text in enumerate(spec["questions"], start=1):
            questions.append(Question(
                question_id=f"PART-{slug}-Q{position}",
                text=text,
                category="partial",
                group_id=topic,
                split=spec["split"],
                answerability="partial",
                expected_behaviour="answer_and_flag_gap",
                risk_level="medium",
                gap_topic=topic,
                paraphrase_of=None if position == 1 else f"PART-{slug}-Q1",
                gold_answer=spec["gold"],
                expected_documents=tuple(spec["documents"]),
            ))

    for spec in FACTUAL:
        questions.append(Question(
            question_id=f"FACT-{spec['id']}",
            text=spec["q"],
            category="factual",
            group_id=f"FACT-{spec['id']}",
            split=spec["split"],
            answerability="answerable",
            expected_behaviour="answer_directly",
            risk_level="low",
            gold_answer=spec["gold"],
            gold_facts=tuple(spec["facts"]),
            expected_chunks=tuple(spec["chunks"]),
            expected_documents=tuple(sorted({c.split("#")[0] for c in spec["chunks"]})),
        ))

    for spec in SYNTHESIS:
        questions.append(Question(
            question_id=f"SYN-{spec['id']}",
            text=spec["q"],
            category="synthesis",
            group_id=f"SYN-{spec['id']}",
            split=spec["split"],
            answerability="answerable",
            expected_behaviour="answer_directly",
            risk_level="low",
            gold_answer=spec["gold"],
            expected_chunks=tuple(spec["chunks"]),
            expected_documents=tuple(spec["documents"]),
        ))

    return QuestionSet(tuple(questions))


def check_no_tuning_fact_reaches_the_test_split(question_set: QuestionSet, registry, chunk_texts) -> None:
    """No test question may expect a chunk carrying a tuning family's disputed fact.

    The tuning boundary is drawn at the level of the *fact*, not the document.
    Having tuned against the returns-window disagreement does not mean the
    inspection table in the same document has been seen, and excluding four
    whole documents from the test set would cost more than it protects. What
    must not happen is a test question whose gold answer turns on a figure the
    pipeline was tuned against.
    """
    anchored: set[str] = set()
    for family in registry.tuning_families:
        for fact in family.conflicting_facts:
            for doc_id, anchor in fact.anchors.items():
                for chunk_id, text in chunk_texts.items():
                    if chunk_id.startswith(doc_id + "#") and anchor in text:
                        anchored.add(chunk_id)

    offenders = [
        (q.question_id, c)
        for q in question_set
        if q.split == "test"
        for c in q.expected_chunks
        if c in anchored
    ]
    if offenders:
        raise QuestionSetError(
            "These test questions expect a chunk carrying a tuning family's disputed "
            f"fact: {offenders}. The pipeline was tuned against those figures, so a "
            "reported result involving them would be contaminated."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without writing")
    args = parser.parse_args()

    config = load_config()
    evaluation = load_evaluation_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    registry = load_conflicts(evaluation.path("conflicts"))

    from sme_assistant.ingest.chunker import chunk_corpus

    chunks = chunk_corpus(
        kb,
        config.require("chunking.max_words"),
        config.require("chunking.overlap_sentences"),
        config.require("chunking.min_words"),
    )
    chunk_texts = {c.chunk_id: c.text for c in chunks}

    question_set = build()
    validate_question_set(question_set, registry=registry, kb=kb)
    check_no_tuning_fact_reaches_the_test_split(question_set, registry, chunk_texts)

    missing = [
        (q.question_id, c)
        for q in question_set
        for c in q.expected_chunks
        if c not in chunk_texts
    ]
    if missing:
        raise QuestionSetError(
            f"These expected chunks do not exist in the current chunk set: {missing}. "
            "The corpus or the chunker has changed since the questions were written."
        )

    summary = question_set.summary()
    print(json.dumps(summary, indent=2))
    print()
    for split in ("dev", "test"):
        part = question_set.split(split)
        print(f"{split:5} {len(part):3} questions, {len(part.groups):2} groups")

    if args.check:
        print("\nValidated. Nothing written.")
        return 0

    path = write_question_set(
        question_set,
        evaluation.path("question_set"),
        corpus_sha256=kb.fingerprint(),
        registry_sha256=registry.fingerprint(),
        preregistration="docs/PREREGISTRATION.md",
        frozen=(
            "Frozen on writing. Questions are not added, removed or reworded after "
            "any arm has been run. A defective question is reported and excluded "
            "with its reason, never silently replaced."
        ),
    )
    print(f"\nWritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
