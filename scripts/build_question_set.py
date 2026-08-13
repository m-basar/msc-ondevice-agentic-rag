"""Build the development and test question sets.

The questions are declared here rather than hand-edited in JSON, for three
reasons. Source is reviewable in a way a 2,000-line JSON blob is not; the
grouping and split rules are applied by construction instead of by hand; and
regenerating after a corpus change is one command rather than a careful edit.

Run:

    python scripts/build_question_set.py            # writes both splits
    python scripts/build_question_set.py --check    # validates without writing

One focal claim per family
--------------------------
Version 1.0 of this file gave each family three questions that asked genuinely
different things - password length, rotation period and SMS authentication -
and scored them against one shared fact list. A concise correct answer about
rotation would have been marked wrong for omitting "14 characters", and an
answer that recited the length while ignoring the question would have passed.

Each family now has one focal claim and three genuine paraphrases of it, and
every question carries its own required and forbidden claims. The other
disputed facts remain in the registry; they are simply not what these questions
measure.

Split rules, enforced in ``sme_assistant.evaluation.question_set``:

* every paraphrase of a conflict family stays with its family
* reported families appear only in the test split
* tuning families appear only in the development split
* no test question may expect a chunk that carries a tuning family's disputed
  fact, so that tuning cannot leak an answer into a reported result
* every required claim must appear in a chunk the question expects
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

# A claim every current_current answer must make. Stating both figures without
# saying they disagree is a different and lesser behaviour, so it is scored
# separately rather than folded into the figures.
DISAGREE = "the two documents disagree and neither supersedes the other"

CONFLICTS: dict[str, dict] = {
    # --- reported: supersession -------------------------------------------
    "CONF-02": {
        "risk": "high",
        "focal": "the first day of sickness absence for which Statutory Sick Pay is payable",
        "cite": ["HR-12#002"],
        "not": ["HR-02"],
        "required": ["Statutory Sick Pay is payable from the first qualifying day"],
        "variants": ["from day one", "no waiting days", "there are no unpaid waiting days"],
        "forbidden": [
            "payable from the fourth qualifying day",
            "three unpaid waiting days apply",
            "the employee must earn at least the Lower Earnings Limit",
        ],
        "gold": (
            "Statutory Sick Pay is payable from the first qualifying day, with no waiting "
            "days, under HR-12. HR-02, which required three unpaid waiting days, was "
            "withdrawn and superseded by HR-12."
        ),
        "questions": [
            "When does Statutory Sick Pay start being paid?",
            "From which day of a sickness absence is sick pay payable?",
            "I went off sick on Monday, a normal working day for me. Is Monday itself paid?",
        ],
    },
    "CONF-03": {
        "risk": "medium",
        "focal": "the minimum password length",
        "cite": ["IT-11#001"],
        "not": ["IT-01"],
        "required": ["passwords must be at least 14 characters"],
        "variants": ["fourteen characters", "a minimum of 14 characters"],
        "forbidden": ["at least 8 characters", "eight characters is sufficient"],
        "gold": (
            "At least 14 characters, with three random words encouraged, under IT-11. "
            "IT-01, which required 8 characters with complexity rules, was withdrawn."
        ),
        "questions": [
            "What is the minimum password length?",
            "How many characters does my password have to be?",
            "Is a ten-character password long enough?",
        ],
    },
    "CONF-04": {
        "risk": "low",
        "focal": "the order value at which delivery becomes free",
        "cite": ["CS-11#002"],
        "not": ["CS-01"],
        "required": ["delivery is free on orders of £75 and over, excluding VAT"],
        "variants": ["£75", "seventy-five pounds", "orders of £75 or more"],
        "forbidden": ["free on orders of £50 and over", "the threshold is £50"],
        "gold": (
            "Delivery is free on orders of £75 and over excluding VAT, under CS-11. "
            "CS-01, which set the threshold at £50, was superseded on 1 February 2026."
        ),
        "questions": [
            "How much do I need to spend to get free delivery?",
            "What is the order value for free delivery?",
            "My basket comes to £60. Will delivery be free?",
        ],
    },
    "CONF-10": {
        "risk": "medium",
        "focal": "the deadline for resolving a stage 1 complaint",
        "cite": ["CS-14#002"],
        "not": ["CS-04"],
        "required": ["a stage 1 complaint is resolved within 2 working days"],
        "variants": ["two working days", "within 2 working days of the complaint"],
        "forbidden": ["within 3 working days", "three working days"],
        "gold": (
            "A stage 1 complaint is acknowledged the same working day and resolved within "
            "2 working days, under CS-14. CS-04, which allowed 3 working days, was "
            "superseded on 1 March 2026."
        ),
        "questions": [
            "How quickly must a stage 1 complaint be resolved?",
            "What is the deadline for resolving a complaint at stage 1?",
            "A customer complained on Monday morning. By when must the advisor have resolved it?",
        ],
    },
    # --- reported: two live documents disagree ------------------------------
    "CONF-06": {
        "risk": "medium",
        "focal": "whether data survives in backups once its retention period has ended",
        "cite": ["IT-04#002", "REG-02#003"],
        "not": [],
        "required": [
            "IT-04 retains an annual backup for seven years",
            "REG-02 requires deletion at the end of the retention period, including from backups",
            DISAGREE,
        ],
        "variants": ["7 years", "conflict", "contradiction", "inconsistent", "cannot be resolved"],
        "forbidden": [
            "backups are kept for seven years and that settles the question",
            "data is always removed from backups at the end of its retention period",
        ],
        "gold": (
            "The documents disagree and both are current. IT-04 retains an annual backup of "
            "all business systems for seven years; REG-02 requires records to be deleted at "
            "the end of their retention period, including from backups at the next rollover. "
            "Neither supersedes the other, and the conflict has data protection "
            "implications, so it should be escalated."
        ),
        "questions": [
            "How long is data kept in backups after its retention period ends?",
            "Once a record's retention period expires, is it removed from backups as well?",
            "We have been asked to delete a customer's data. Does that include the backups?",
        ],
    },
    "CONF-07": {
        "risk": "low",
        "focal": "who approves an IT equipment purchase of £800",
        "cite": ["GEN-04#002", "FIN-03#002"],
        "not": [],
        "required": [
            "GEN-04 requires Finance Manager approval above £750",
            "FIN-03 places £501 to £2,500 with the department head",
            DISAGREE,
        ],
        "variants": ["conflict", "contradiction", "both approvers", "unclear which applies"],
        "forbidden": [
            "the Finance Manager alone approves it",
            "the department head alone approves it",
        ],
        "gold": (
            "The documents disagree and both are current. GEN-04 requires Finance Manager "
            "approval because the cost exceeds £750; FIN-03 places £501 to £2,500 with the "
            "department head. Neither supersedes the other, so both should be named and the "
            "discrepancy escalated."
        ),
        "questions": [
            "Who approves an £800 laptop purchase?",
            "I need to buy IT equipment costing £800. Whose sign-off do I need?",
            "What is the approval route for an IT purchase of eight hundred pounds?",
        ],
    },
    "CONF-08": {
        "risk": "low",
        "focal": "the trade counter opening hours",
        "cite": ["GEN-02#001", "CS-11#003"],
        "not": [],
        "required": [
            "GEN-02 gives 08:00 to 17:00",
            "CS-11 gives 08:30 to 16:30",
            DISAGREE,
        ],
        "variants": ["conflict", "contradiction", "two different sets of hours"],
        "forbidden": [
            "the trade counter is open 08:00 to 17:00 and that is the answer",
            "the trade counter is open 08:30 to 16:30 and that is the answer",
        ],
        "gold": (
            "The documents disagree and both are current. GEN-02 gives the trade counter "
            "hours as Monday to Friday 08:00 to 17:00; CS-11 gives 08:30 to 16:30. Neither "
            "supersedes the other, so both should be reported and the discrepancy flagged."
        ),
        "questions": [
            "What are the trade counter opening hours?",
            "When is the trade counter open?",
            "Can I collect an order from the trade counter at 16:45 on a Wednesday?",
        ],
    },
    "CONF-09": {
        "risk": "medium",
        "focal": "how often user access rights are reviewed",
        "cite": ["IT-11#003", "IT-02#003"],
        "not": [],
        "required": [
            "IT-11 requires review every six months by each department head",
            "IT-02 states annually by the Data Protection Lead",
            DISAGREE,
        ],
        "variants": ["conflict", "contradiction", "two different cycles"],
        "forbidden": [
            "access rights are reviewed every six months and that is the answer",
            "access rights are reviewed annually and that is the answer",
        ],
        "gold": (
            "The documents disagree and both are current. IT-11 requires access rights to be "
            "reviewed every six months by each department head; IT-02 states that access to "
            "systems holding personal data is reviewed annually by the Data Protection Lead. "
            "Neither supersedes the other, so both should be surfaced."
        ),
        "questions": [
            "How often are user access rights reviewed?",
            "What is the review cycle for system access permissions?",
            "How frequently is my system access supposed to be checked?",
        ],
    },
    "CONF-11": {
        "risk": "high",
        "focal": "whether a visitor needs safety footwear to enter the warehouse",
        "cite": ["GEN-03#002", "OPS-05#001"],
        "not": [],
        "required": [
            "OPS-05 makes safety footwear mandatory for everyone in the warehouse at all times",
            "GEN-03 exempts escorted visits of under fifteen minutes on marked walkways",
            DISAGREE,
        ],
        "variants": ["conflict", "contradiction", "safety implications", "escalate"],
        "forbidden": [
            "a visitor may enter the warehouse without safety footwear",
            "the fifteen-minute exemption is the applicable rule",
        ],
        "gold": (
            "The documents disagree and both are current, and the disagreement is a safety "
            "one. OPS-05 makes safety footwear mandatory for everyone in the warehouse at "
            "all times, including visitors passing through; GEN-03 exempts escorted visits "
            "of less than fifteen minutes that keep to the marked walkways. Neither "
            "supersedes the other. Both should be surfaced and the conflict escalated rather "
            "than resolved by the assistant."
        ),
        "questions": [
            "Does a visitor need safety footwear to enter the warehouse?",
            "Does a visitor need to wear safety boots for a short escorted visit?",
            "A customer wants a ten-minute look at the racking. What do they need on their feet?",
        ],
    },
    # --- development: pilot-contaminated, moved under amendment 1.1 ---------
    "CONF-01": {
        "risk": "medium",
        "focal": "the mileage rate for the first 10,000 business miles",
        "cite": ["HR-13#001"],
        "not": ["HR-03"],
        "required": ["55 pence per mile for the first 10,000 business miles"],
        "variants": ["55p", "fifty-five pence"],
        "forbidden": ["40 pence per mile", "a flat rate of 40 pence"],
        "gold": (
            "55 pence per mile for the first 10,000 business miles in a tax year and 25 "
            "pence per mile thereafter, under HR-13. HR-03, which stated a flat 40 pence, "
            "was withdrawn."
        ),
        "questions": [
            "What is the mileage rate for business travel?",
            "How much can I claim per mile for using my own car on company business?",
            "What do I get paid per mile when I drive to a customer?",
        ],
    },
    "CONF-05": {
        "risk": "high",
        "focal": "the deadline for reporting a lost or stolen company device",
        "cite": ["GEN-04#003", "IT-03#002"],
        "not": [],
        "required": [
            "IT-03 requires reporting within 1 hour of becoming aware",
            "GEN-04 allows 24 hours",
            DISAGREE,
        ],
        "variants": ["conflict", "contradiction", "one hour", "twenty-four hours"],
        "forbidden": [
            "the deadline is 24 hours and that is the answer",
            "the deadline is 1 hour and that is the answer",
        ],
        "gold": (
            "The documents disagree and both are current. IT-03 requires a lost or stolen "
            "device to be reported within 1 hour of becoming aware; GEN-04 allows 24 hours. "
            "Neither supersedes the other."
        ),
        "questions": [
            "How quickly must I report a lost company laptop?",
            "What is the time limit for reporting lost or stolen company equipment?",
            "My work phone was stolen last night. By when must I report it?",
        ],
    },
    "TUNE-01": {
        "risk": "low",
        "focal": "the returns window for an unwanted item",
        "cite": ["OPS-02#001", "CS-03#001"],
        "not": [],
        "required": [
            "OPS-02 allows return within 30 days of delivery",
            "CS-03 states 28 days",
            DISAGREE,
        ],
        "variants": ["conflict", "contradiction"],
        "forbidden": ["the returns window is 30 days", "the returns window is 28 days"],
        "gold": (
            "The documents disagree and both are current. OPS-02 allows return within 30 "
            "days of delivery; CS-03 states 28 days. Neither supersedes the other. The "
            "statutory 14-day cancellation right is separate and unaffected."
        ),
        "questions": [
            "How long do I have to return an unwanted item?",
            "What is the returns window for goods I no longer want?",
            "My order arrived 29 days ago and I want to send it back. Am I still in time?",
        ],
    },
    "TUNE-02": {
        "risk": "medium",
        "focal": "how long the Health and Safety Coordinator has to review an accident book entry",
        "cite": ["OPS-08#004", "REG-01#005"],
        "not": [],
        "required": [
            "OPS-08 requires review within two working days",
            "REG-01 states five working days",
            DISAGREE,
        ],
        "variants": ["conflict", "contradiction"],
        "forbidden": ["the review happens within two working days", "the review happens within five working days"],
        "gold": (
            "The documents disagree and both are current. OPS-08 requires the Health and "
            "Safety Coordinator to review every accident book entry within two working days; "
            "REG-01 states five working days. Neither supersedes the other."
        ),
        "questions": [
            "How long does the Health and Safety Coordinator have to review an accident book entry?",
            "After an accident is written in the accident book, when is it reviewed?",
            "What is the timescale for reviewing accident records?",
        ],
    },
}

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
            "Which commodity codes do we put on a commercial invoice?",
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

PARTIALS: dict[str, dict] = {
    "probationary period": {
        "split": "dev",
        "documents": ["HR-04"],
        "required": [
            "HR-04 refers to a probationary period",
            "no document states how long the probationary period lasts",
        ],
        "variants": ["the corpus does not say", "not specified", "not documented"],
        "forbidden": ["the probationary period is three months", "the probationary period is six months"],
        "gold": (
            "The corpus does not state the length of the probationary period. HR-04 refers "
            "to it, saying new employees work on site for its duration, but no document "
            "defines how long it lasts or what happens at the end of it."
        ),
        "questions": [
            "How long is the probationary period?",
            "What is the length of probation for a new starter?",
        ],
    },
    "grievance procedure": {
        "split": "test",
        "documents": ["HR-05"],
        "required": [
            "HR-05 names the ACAS Code of Practice on disciplinary and grievance procedures",
            "no grievance procedure is documented in the corpus",
        ],
        "variants": ["the corpus does not contain", "not documented", "only the disciplinary procedure"],
        "forbidden": ["raise the grievance in writing with your line manager within five days"],
        "gold": (
            "The corpus does not document a grievance procedure. HR-05 names the ACAS Code "
            "of Practice on disciplinary and grievance procedures, so the word appears and "
            "HR-05 will be retrieved, but HR-05 sets out the disciplinary process only."
        ),
        "questions": [
            "How do I raise a grievance?",
            "What are the stages of the grievance procedure?",
        ],
    },
}

FACTUAL: list[dict] = [
    {"split": "dev", "id": "leave-entitlement", "q": "How many days of annual leave do full-time employees get?",
     "chunks": ["HR-01#001"],
     "required": ["25 days of paid annual leave", "in addition to the 8 English bank holidays"],
     "variants": ["twenty-five days", "eight bank holidays"], "forbidden": [],
     "gold": "25 days of paid annual leave per leave year, in addition to the 8 English bank holidays. Part-time employees receive a pro-rata entitlement."},
    {"split": "dev", "id": "fire-wardens", "q": "How many trained fire wardens does the site have?",
     "chunks": ["OPS-07#005"],
     "required": ["six trained fire wardens"],
     "variants": ["yellow high-visibility tabards", "refreshed every two years"], "forbidden": [],
     "gold": "Six, identified by yellow high-visibility tabards kept at each fire point. Warden training is refreshed every two years.",
     "note": "Replaced the assembly-point question under amendment 1.2: OPS-07#003 now carries CONF-13's disputed assembly point."},
    {"split": "dev", "id": "petty-cash-limit", "q": "What is the maximum single petty cash transaction?",
     "chunks": ["FIN-02#002"],
     "required": ["£50"], "variants": ["fifty pounds"], "forbidden": [],
     "gold": "£50. The float is £250, held in the locked cash box in the Finance office."},
    {"split": "dev", "id": "goods-in-detailed-check", "q": "How soon are item quantities verified after a delivery arrives?",
     "chunks": ["OPS-04#003"],
     "required": ["within one working day of receipt"],
     "variants": ["one working day"], "forbidden": [],
     "gold": "Cartons are opened and item quantities verified within one working day of receipt. Discrepancies are recorded on the system and reported to Procurement the same day.",
     "note": "Replaced the same-day picking question under amendment 1.2: OPS-01#003 now carries CONF-15's disputed cut-off, so a development question expecting it would tune against a chunk that is later scored."},
    {"split": "dev", "id": "knives-dishwasher", "q": "Can I put my knives in the dishwasher?",
     "chunks": ["PRD-01#001"],
     "required": ["knives are washed by hand, not in a dishwasher"],
     "variants": ["hand wash", "no"], "forbidden": ["dishwasher safe"],
     "gold": "No. Knives are washed by hand in warm soapy water and dried immediately. Dishwasher detergent is abrasive and dulls the edge, heat can loosen handle scales, and blades knock against other items."},
    {"split": "dev", "id": "remote-days", "q": "How many days a week can I work from home?",
     "chunks": ["HR-04#002"],
     "required": ["up to two days per week"],
     "variants": ["2 days"], "forbidden": [],
     "gold": "Up to two days per week for eligible office-based roles. Monday is a core on-site day for all office staff."},
    {"split": "test", "id": "damaged-report-window", "q": "How quickly must I report a damaged delivery?",
     "chunks": ["CS-02#001"],
     "required": ["within 48 hours of delivery"],
     "variants": ["48 hours", "two days"], "forbidden": [],
     "gold": "Within 48 hours of delivery, to customer services, with the order number and photographs of the damage and the outer packaging."},
    {"split": "test", "id": "contractor-insurance", "q": "What must a contractor provide before their first visit?",
     "chunks": ["GEN-03#003"],
     "required": ["a valid public liability insurance certificate"],
     "variants": ["public liability insurance"], "forbidden": [],
     "gold": "A valid public liability insurance certificate, a site induction with the Facilities Coordinator, and a risk assessment and method statement for work at height, hot work or electrical isolation."},
    {"split": "test", "id": "forklift-authorisation", "q": "Who is allowed to drive a forklift?",
     "chunks": ["OPS-06#001"],
     "required": ["a current RTITB or equivalent accredited certificate", "the site authorised operator list"],
     "variants": ["RTITB"], "forbidden": [],
     "gold": "Only employees holding a current RTITB or equivalent accredited certificate for the relevant truck category who also appear on the site authorised operator list. Authorisation is site-specific."},
    {"split": "test", "id": "payment-terms", "q": "What are the standard payment terms?",
     "chunks": ["FIN-01#001"],
     "required": ["30 days from the date of invoice"],
     "variants": ["thirty days"], "forbidden": [],
     "gold": "30 days from the date of invoice. Invoices are raised automatically on despatch and emailed to the address held on the customer account."},
    {"split": "test", "id": "account-hold", "q": "At how many days overdue is an account placed on hold?",
     "chunks": ["FIN-01#003"],
     "required": ["30 days overdue"],
     "variants": ["thirty days"], "forbidden": [],
     "gold": "30 days overdue, by formal letter. A reminder email goes at 7 days and a credit control telephone call at 14 days."},
    {"split": "test", "id": "refund-method", "q": "How is a refund paid back to a customer?",
     "chunks": ["OPS-03#003"],
     "required": ["to the original payment method"],
     "variants": ["the card used to pay"], "forbidden": [],
     "gold": "To the original payment method. Card refunds are processed by Finance through the payment gateway and typically reach the customer three to five working days later, which is outside the company's control."},
    {"split": "dev", "id": "development-conversation", "q": "When does the annual development conversation happen?",
     "chunks": ["HR-06#002"],
     "required": ["in April"],
     "variants": ["April, with the line manager"], "forbidden": [],
     "gold": "In April, with the employee's line manager, following the performance review cycle.",
     "note": "Moved from test to development under amendment 1.2: HR-06#002 now also carries TUNE-03's disputed core hours, so a test question expecting that chunk would be scored on evidence the pipeline was tuned against."},
    {"split": "test", "id": "written-warning-duration", "q": "How long does a first written warning stay live?",
     "chunks": ["HR-05#002"],
     "required": ["6 months"],
     "variants": ["six months"], "forbidden": [],
     "gold": "Six months. A final written warning stays live for 12 months."},
    {"split": "test", "id": "forklift-preuse", "q": "What has to be checked on a forklift before a shift?",
     "chunks": ["OPS-06#002"],
     "required": ["a documented pre-use check at the start of every shift"],
     "variants": ["tyres, forks and mast", "brakes", "hydraulic leaks"], "forbidden": [],
     "gold": "A documented pre-use check covering tyres, forks and mast for damage, hydraulic leaks, brakes including the parking brake, horn, lights and beacon, and the battery."},
]

SYNTHESIS: list[dict] = [
    {"split": "dev", "id": "new-starter-week-one",
     "q": "I start on Monday in an office role. What training and equipment should I expect in my first week?",
     "chunks": ["HR-06#001", "HR-04#004"], "documents": ["HR-06", "HR-04"],
     "required": ["a corporate induction on the first day", "the company provides a laptop and headset"],
     "variants": ["induction covers health and safety, fire procedures and data protection"],
     "forbidden": ["the company reimburses home broadband"],
     "gold": "A corporate induction on the first day covering company overview, health and safety, fire procedures and data protection. The company provides a laptop and headset. It does not reimburse home broadband, heating or electricity, and a second monitor for home use goes to department heads only."},
    {"split": "dev", "id": "short-delivery-dispute",
     "q": "A supplier delivery is short and we want to dispute their invoice. What are the steps and the deadline?",
     "chunks": ["OPS-04#003", "FIN-01#004"], "documents": ["OPS-04", "FIN-01"],
     "required": ["discrepancies are recorded and reported to Procurement the same day", "a disputed invoice must be raised within 14 days"],
     "variants": ["one working day", "the undisputed balance remains payable"], "forbidden": [],
     "gold": "Item quantities are verified within one working day of receipt, and any discrepancy is recorded on the system and reported to Procurement the same day so a claim can be raised. A disputed invoice must be raised within 14 days; Finance logs the dispute and holds only the disputed amount, with the undisputed balance payable on the original terms.",
     "note": "Replaced the visitor-evacuation question under amendment 1.2: GEN-03#002 and OPS-07#003 now carry CONF-11's and CONF-13's disputed facts."},
    {"split": "test", "id": "damaged-order-end-to-end",
     "q": "A customer's order arrived damaged and they want their money back. What happens from the moment they call?",
     "chunks": ["CS-02#001", "CS-02#003", "OPS-03#001"], "documents": ["CS-02", "OPS-03"],
     "required": ["report within 48 hours with photographs", "a replacement or a refund at the customer's choice"],
     "variants": ["damage report", "refund processed under OPS-03"], "forbidden": [],
     "gold": "The customer reports the damage within 48 hours with photographs. Customer services raise a damage report and offer, at the customer's choice, a replacement despatched within one working day of the report being accepted, or a refund processed under OPS-03. Damaged items are collected by the carrier."},
    {"split": "test", "id": "new-trade-customer",
     "q": "A new trade customer wants a credit account and next working day delivery. What do they need to do?",
     "chunks": ["FIN-01#002", "CS-11#002"], "documents": ["FIN-01", "CS-11"],
     "required": ["apply using the account application form", "next working day delivery costs £12.95"],
     "variants": ["a credit reference and two trade references", "opening limits are typically £2,500"],
     "forbidden": [],
     "gold": "They apply for a credit account using the account application form. Finance obtains a credit reference and two trade references before the account is opened, with an opening limit typically £2,500. Next working day delivery costs £12.95 excluding VAT."},
    {"split": "test", "id": "faulty-warranty-return",
     "q": "An item comes back faulty and still under warranty. Who deals with it and what happens?",
     "chunks": ["OPS-02#002", "CS-03#003"], "documents": ["OPS-02", "CS-03"],
     "required": ["a returns operative inspects each item within two working days of receipt", "the claim is made with the order number, a description of the fault and photographs"],
     "variants": ["disposition D", "routed to a supplier claim"], "forbidden": [],
     "gold": "A returns operative inspects the item within two working days of receipt and assigns a disposition; a faulty item within warranty is photographed and routed to a supplier claim under CS-03. The customer claims with the order number, a description of the fault and photographs."},
]


def build(registry) -> QuestionSet:
    """Assemble the set, taking split and behaviour from the registry."""
    questions: list[Question] = []
    tuning_ids = {f.family_id for f in registry.tuning_families}

    for family, spec in CONFLICTS.items():
        entry = registry.by_id(family)
        behaviour = entry.expected_behaviour
        split = "dev" if family in tuning_ids else "test"
        for position, text in enumerate(spec["questions"], start=1):
            questions.append(Question(
                question_id=f"{family}-Q{position}",
                text=text,
                category="conflict",
                group_id=family,
                split=split,
                answerability="answerable",
                expected_behaviour=behaviour,
                risk_level=spec["risk"],
                family_id=family,
                paraphrase_of=None if position == 1 else f"{family}-Q1",
                gold_answer=spec["gold"],
                required_claims=tuple(spec["required"]),
                forbidden_claims=tuple(spec["forbidden"]),
                acceptable_variants=tuple(spec["variants"]),
                expected_chunks=tuple(spec["cite"]),
                expected_documents=tuple(sorted({c.split("#")[0] for c in spec["cite"]})),
                must_not_cite=tuple(spec["not"]),
                notes=f"Focal claim: {spec['focal']}.",
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
                required_claims=tuple(spec["required"]),
                forbidden_claims=tuple(spec["forbidden"]),
                acceptable_variants=tuple(spec["variants"]),
                expected_documents=tuple(spec["documents"]),
            ))

    for spec in FACTUAL + SYNTHESIS:
        prefix = "FACT" if spec in FACTUAL else "SYN"
        questions.append(Question(
            question_id=f"{prefix}-{spec['id']}",
            text=spec["q"],
            category="factual" if prefix == "FACT" else "synthesis",
            group_id=f"{prefix}-{spec['id']}",
            split=spec["split"],
            answerability="answerable",
            expected_behaviour="answer_directly",
            risk_level="low",
            gold_answer=spec["gold"],
            required_claims=tuple(spec["required"]),
            forbidden_claims=tuple(spec.get("forbidden", ())),
            acceptable_variants=tuple(spec.get("variants", ())),
            expected_chunks=tuple(spec["chunks"]),
            expected_documents=tuple(
                spec.get("documents") or sorted({c.split("#")[0] for c in spec["chunks"]})
            ),
        ))

    return QuestionSet(tuple(questions))


def _anchored_tuning_chunks(registry, chunk_texts) -> set[str]:
    anchored: set[str] = set()
    for family in registry.tuning_families:
        for fact in family.conflicting_facts:
            for doc_id, anchor in fact.anchors.items():
                for chunk_id, text in chunk_texts.items():
                    if chunk_id.startswith(doc_id + "#") and anchor in text:
                        anchored.add(chunk_id)
    return anchored


def check_no_reported_fact_reaches_the_development_split(question_set, registry, chunk_texts) -> None:
    """The mirror of the check below, and it caught a real case.

    A development question expecting a chunk that carries a *reported* family's
    disputed fact means the pipeline is tuned against the very passage that
    decides a reported result. The original check only guarded tuning leaking
    into test, which is half the boundary.
    """
    anchored: set[str] = set()
    for family in registry.families:
        for fact in family.conflicting_facts:
            for doc_id, anchor in fact.anchors.items():
                for chunk_id, text in chunk_texts.items():
                    if chunk_id.startswith(doc_id + "#") and anchor in text:
                        anchored.add(chunk_id)

    offenders = [
        (q.question_id, c)
        for q in question_set
        if q.split == "dev"
        for c in q.expected_chunks
        if c in anchored
    ]
    if offenders:
        raise QuestionSetError(
            "These development questions expect a chunk carrying a reported family's "
            f"disputed fact: {offenders}. Tuning against the passage that decides a "
            "reported result is contamination in the other direction."
        )


def check_no_tuning_fact_reaches_the_test_split(question_set, registry, chunk_texts) -> None:
    """No test question may expect a chunk carrying a tuning family's disputed fact.

    The tuning boundary is drawn at the level of the fact, not the document.
    Having tuned against the returns-window disagreement does not mean the
    inspection table in the same document has been seen, and excluding whole
    documents would cost more than it protects. What must not happen is a test
    question whose gold answer turns on a figure the pipeline was tuned against.
    """
    anchored = _anchored_tuning_chunks(registry, chunk_texts)
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
            f"fact: {offenders}."
        )


DOC_ID_RE = __import__("re").compile(r"\b[A-Z]{2,4}-\d{2}\b")
FIGURE_RE = __import__("re").compile(r"£?\d[\d,]*(?:[.:]\d+)?%?")


def check_required_claims_are_evidenced(question_set, registry, chunk_texts, kb) -> None:
    """Every required claim must be supported by a chunk the question expects.

    Checking that an expected chunk *exists* proves nothing about whether it
    contains what the question asks for. Two checks run here.

    First, exactly: for a conflict question, each expected chunk must contain
    the literal anchor the registry records for that document. The anchor is
    the text that evidences the disputed value, so this is a direct assertion
    that the question points at the passage the disagreement lives in.

    Second, approximately: any figure appearing in a required claim must appear
    in the expected evidence. Document identifiers are stripped first. ``IT-04``
    otherwise contributes the figure "04", which no chunk contains, so every
    claim naming a document would be reported unsupported - the same failure
    that made citation scoring read zero everywhere until it was found.

    A claim with no figures after stripping is qualitative and is not checkable
    this way. That is a limit of the check, not a pass.
    """
    unevidenced: list[tuple[str, str]] = []
    misanchored: list[tuple[str, str, str]] = []

    for question in question_set:
        pool = " ".join(chunk_texts.get(c, "") for c in question.expected_chunks)

        if question.family_id:
            family = registry.by_id(question.family_id)
            # A family declares several disputed facts and their anchors live in
            # different chunks. A focal question expects only the chunk carrying
            # the fact it asks about, so the requirement is that the chunk
            # carries *one* of that document's anchors, not all of them.
            anchors: dict[str, list[str]] = {}
            for fact in family.conflicting_facts:
                for doc_id, anchor in fact.anchors.items():
                    anchors.setdefault(doc_id, []).append(anchor)
            # A compatible family's reconciliation evidence counts too: showing
            # why two documents do not conflict is exactly what its questions
            # must be answered from.
            for doc_id, anchor in family.reconciliation_anchors.items():
                anchors.setdefault(doc_id, []).append(anchor)
            for chunk_id in question.expected_chunks:
                candidates = anchors.get(chunk_id.split("#")[0], [])
                text = chunk_texts.get(chunk_id, "")
                if candidates and not any(a in text for a in candidates):
                    misanchored.append((question.question_id, chunk_id, candidates[0]))

        for claim in question.required_claims:
            figures = set(FIGURE_RE.findall(DOC_ID_RE.sub(" ", claim)))
            if not figures:
                continue
            if not any(f in pool for f in figures):
                unevidenced.append((question.question_id, claim))

    if misanchored:
        raise QuestionSetError(
            "These questions expect a chunk that does not contain its family's "
            f"anchor text: {misanchored}. The question points at the wrong passage."
        )
    if unevidenced:
        raise QuestionSetError(
            "These required claims are not supported by the evidence the question "
            f"expects: {unevidenced}. Either the claim is wrong or the expected "
            "chunks are."
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
    from sme_assistant.ingest.index import chunk_set_fingerprint

    chunks = chunk_corpus(
        kb,
        config.require("chunking.max_words"),
        config.require("chunking.overlap_sentences"),
        config.require("chunking.min_words"),
    )
    chunk_texts = {c.chunk_id: c.text for c in chunks}

    question_set = build(registry)
    validate_question_set(question_set, registry=registry, kb=kb)
    check_no_tuning_fact_reaches_the_test_split(question_set, registry, chunk_texts)
    check_no_reported_fact_reaches_the_development_split(question_set, registry, chunk_texts)
    check_required_claims_are_evidenced(question_set, registry, chunk_texts, kb)

    missing = [
        (q.question_id, c)
        for q in question_set
        for c in q.expected_chunks
        if c not in chunk_texts
    ]
    if missing:
        raise QuestionSetError(f"Expected chunks not in the current chunk set: {missing}")

    print(json.dumps(question_set.summary(), indent=2))
    print()
    for split in ("dev", "test"):
        part = question_set.split(split)
        families = {q.family_id for q in part if q.family_id}
        print(f"{split:5} {len(part):3} questions, {len(part.groups):2} groups, "
              f"{len(families)} conflict families")

    if args.check:
        print("\nValidated. Nothing written.")
        return 0

    path = write_question_set(
        question_set,
        evaluation.path("question_set"),
        version="1.1",
        corpus_sha256=kb.fingerprint(),
        chunk_set_sha256=chunk_set_fingerprint(chunks),
        registry_sha256=registry.fingerprint(),
        preregistration="docs/PREREGISTRATION.md, amendment 1.1 of 8 August 2026",
        frozen=(
            "Frozen on writing. Questions are not added, removed or reworded after "
            "any arm has been run. A defective question is reported and excluded "
            "with its reason, never silently replaced."
        ),
    )
    print(f"\nWritten to {path}")
    return 0



# --- amendment 1.2 ----------------------------------------------------------
# Eight families added and two rewritten. CONF-07 and CONF-09 were not
# contradictions: GEN-04 requires the department head AND the Finance Manager,
# which FIN-03 does not contradict, and IT-11 and IT-02 describe two review
# layers rather than two answers. They are now negative controls, and a
# verifier that flags them is producing a false positive.
#
# Merged over CONFLICTS rather than edited into it, so the diff shows exactly
# what amendment 1.2 changed.

AMENDMENT_1_2: dict[str, dict] = {
    "CONF-07": {
        "risk": "low",
        "focal": "who approves an IT equipment purchase of £800",
        "cite": ["GEN-04#002", "FIN-03#002"],
        "not": [],
        "required": [
            "the department head approves it",
            "the Finance Manager also approves it because the cost exceeds £750",
        ],
        "variants": ["both approvals are needed", "department head and Finance Manager"],
        "forbidden": [
            "the two documents contradict each other",
            "only the department head is required",
            "only the Finance Manager is required",
        ],
        "gold": (
            "Both approvals. GEN-04 requires written approval from the department head "
            "and, because the cost exceeds £750, from the Finance Manager. FIN-03 places "
            "£501 to £2,500 with the department head. The requirements are cumulative, "
            "not competing: GEN-04 is simply stricter for IT equipment."
        ),
        "questions": [
            "Who approves an £800 laptop purchase?",
            "I need to buy IT equipment costing £800. Whose sign-off do I need?",
            "What is the approval route for an IT purchase of eight hundred pounds?",
        ],
    },
    "CONF-09": {
        "risk": "medium",
        "focal": "how often user access rights are reviewed",
        "cite": ["IT-11#003", "IT-02#003"],
        "not": [],
        "required": [
            "department heads review access rights every six months",
            "the Data Protection Lead reviews systems holding personal data annually",
        ],
        "variants": ["two reviews", "both reviews take place"],
        "forbidden": [
            "the two documents contradict each other",
            "only one of the two reviews takes place",
        ],
        "gold": (
            "Two separate reviews. Each department head reviews access rights every six "
            "months under IT-11. The Data Protection Lead reviews access to systems "
            "holding personal data annually under IT-02. Different reviewer, different "
            "scope, different cadence; both happen."
        ),
        "questions": [
            "How often are user access rights reviewed?",
            "What is the review cycle for system access permissions?",
            "Who checks my system access, and how often?",
        ],
    },
    "CONF-12": {
        "risk": "medium",
        "focal": "how many days overdue an invoice must be before orders are held",
        "cite": ["OPS-01#002", "FIN-01#003"],
        "not": [],
        "required": [
            "OPS-01 holds orders at more than 45 days overdue",
            "FIN-01 places the account on hold at 30 days overdue",
            DISAGREE,
        ],
        "variants": ["conflict", "contradiction", "escalate"],
        "forbidden": ["orders are held at 30 days and that is the answer",
                      "orders are held at 45 days and that is the answer"],
        "gold": (
            "The documents disagree and both are current. OPS-01 holds trade orders when "
            "an invoice is more than 45 days overdue; FIN-01 places the account on hold "
            "at 30 days overdue. Neither supersedes the other, so both should be reported "
            "and the discrepancy escalated."
        ),
        "questions": [
            "How overdue does an invoice have to be before orders are held?",
            "At what point does an unpaid invoice stop a customer ordering?",
            "A customer has an invoice 35 days overdue. Can they still place an order?",
        ],
    },
    "CONF-13": {
        "risk": "high",
        "focal": "where staff assemble on hearing the fire alarm",
        "cite": ["OPS-07#003", "OPS-05#005"],
        "not": [],
        "required": [
            "OPS-07 gives the north corner of the staff car park",
            "OPS-05 sends warehouse staff to the main gate",
            DISAGREE,
        ],
        "variants": ["conflict", "contradiction", "safety", "escalate"],
        "forbidden": ["the assembly point is the staff car park and that is the answer",
                      "the assembly point is the main gate and that is the answer"],
        "gold": (
            "The documents disagree and both are current, and the disagreement is a "
            "safety one. OPS-07 gives the assembly point as the north corner of the staff "
            "car park; OPS-05 sends warehouse staff to the main gate. Two assembly points "
            "means a roll call cannot account for everyone. Neither supersedes the other, "
            "so both should be surfaced and the conflict escalated urgently."
        ),
        "questions": [
            "Where do I go when the fire alarm sounds?",
            "What is the assembly point in an evacuation?",
            "I work in the warehouse. Where do I assemble if the alarm goes off?",
        ],
    },
    "CONF-14": {
        "risk": "medium",
        "focal": "how quickly a complaint or damage report is acknowledged",
        "cite": ["CS-14#002", "CS-02#003"],
        "not": [],
        "required": [
            "CS-14 requires acknowledgement the same working day",
            "CS-02 allows two working days",
            "the same working day is the safe course",
        ],
        "variants": ["stricter", "escalate", "discrepancy"],
        "forbidden": ["two working days is the deadline"],
        "gold": (
            "Acknowledge the same working day. CS-14 requires acknowledgement on the same "
            "working day; CS-02 allows two working days for a damage report. Acknowledging "
            "same day satisfies both, so it is the safe course, but the discrepancy should "
            "be raised for resolution."
        ),
        "questions": [
            "How quickly must a complaint be acknowledged?",
            "What is the deadline for acknowledging a customer complaint?",
            "A damage report came in this morning. By when must we acknowledge it?",
        ],
    },
    "CONF-15": {
        "risk": "low",
        "focal": "the latest time an order can be placed and still ship the same day",
        "cite": ["OPS-01#003", "CS-11#001"],
        "not": [],
        "required": [
            "OPS-01 picks only orders released before 13:00",
            "CS-11 promises next working day on orders placed before 14:00",
            "13:00 is the safe course",
        ],
        "variants": ["stricter", "escalate", "discrepancy"],
        "forbidden": ["an order placed at 13:30 will ship the same day"],
        "gold": (
            "Treat 13:00 as the cut-off. OPS-01 picks only orders released before 13:00; "
            "CS-11 offers next working day delivery on orders placed and paid before "
            "14:00. An order taken at 13:30 under CS-11 will not be picked that day under "
            "OPS-01, so 13:00 is the safe answer and the discrepancy should be raised."
        ),
        "questions": [
            "What is the cut-off for an order to be picked the same day?",
            "How late can an order be placed and still go out today?",
            "A customer wants to order at 13:30 for next-day delivery. Will it make it?",
        ],
    },
    "CONF-16": {
        "risk": "high",
        "focal": "how quickly a suspected personal data breach is reported",
        "cite": ["IT-02#003", "REG-02#003"],
        "not": [],
        "required": [
            "IT-02 requires reporting within 24 hours of discovery",
            "REG-02 allows 72 hours",
            "24 hours is the safe course",
        ],
        "variants": ["stricter", "escalate", "discrepancy"],
        "forbidden": ["the deadline is 72 hours"],
        "gold": (
            "Report within 24 hours. IT-02 requires a suspected personal data breach to "
            "be reported to the Data Protection Lead within 24 hours of discovery; REG-02 "
            "allows 72 hours. Reporting within 24 hours satisfies both, so it is the safe "
            "course, and the discrepancy should be escalated given the regulatory exposure."
        ),
        "questions": [
            "How quickly must a suspected data breach be reported?",
            "What is the deadline for telling the Data Protection Lead about a breach?",
            "I think personal data was sent to the wrong person. When do I have to report it?",
        ],
    },
    "CONF-17": {
        "risk": "low",
        "focal": "who approves a £30 operational purchase",
        "cite": ["FIN-02#001", "FIN-02#004", "FIN-03#002"],
        "not": [],
        "required": [
            "petty cash above £25 is approved by the Finance Manager under FIN-02",
            "FIN-02 states it is not an alternative to the procurement process in FIN-03",
        ],
        "variants": ["separate routes", "different processes"],
        "forbidden": [
            "the two documents contradict each other",
            "petty cash may be used instead of a purchase order",
        ],
        "gold": (
            "It depends which route is used, and the documents say so themselves. Petty "
            "cash above £25 requires Finance Manager approval before purchase under "
            "FIN-02. FIN-03's £500 line-manager band covers procurement. FIN-02 states "
            "explicitly that petty cash is not an alternative to the procurement process "
            "in FIN-03, so these are two routes, not two answers."
        ),
        "questions": [
            "Who approves a £30 purchase?",
            "I need to buy something small for £30. Whose approval do I need?",
            "Does a £30 purchase need a line manager or the Finance Manager?",
        ],
    },
    "TUNE-03": {
        "risk": "medium",
        "focal": "the core hours during which an employee must be contactable",
        "cite": ["HR-04#003", "HR-06#002"],
        "not": [],
        "required": [
            "HR-04 gives core hours of 09:30 to 16:00",
            "HR-06 gives 10:00 to 16:30",
            DISAGREE,
        ],
        "variants": ["conflict", "contradiction", "escalate"],
        "forbidden": ["core hours are 09:30 to 16:00 and that is the answer"],
        "gold": (
            "The documents disagree and both are current. HR-04 gives core hours of 09:30 "
            "to 16:00 for remote working; HR-06 gives 10:00 to 16:30 during training. "
            "Neither supersedes the other."
        ),
        "questions": [
            "What are the core hours I have to be contactable?",
            "Between what times do I need to be available?",
            "Am I expected to be contactable at 09:45?",
        ],
    },
    "TUNE-04": {
        "risk": "low",
        "focal": "whether washing-up liquid may be used when washing by hand",
        "cite": ["PRD-01#001", "PRD-02#002"],
        "not": [],
        "required": [
            "PRD-01 covers knives and PRD-02 covers cast iron",
            "the advice differs because the products differ",
        ],
        "variants": ["different products", "not the same item"],
        "forbidden": ["the two guides contradict each other"],
        "gold": (
            "They are not in conflict: PRD-01 is about knives and PRD-02 about cast iron. "
            "Knives are washed by hand in warm soapy water and dried immediately. Cast "
            "iron is washed by hand with hot water and a stiff brush, and a little "
            "washing-up liquid is fine on a well-seasoned pan."
        ),
        "questions": [
            "Can I use washing-up liquid when washing by hand?",
            "Is detergent safe on my cookware?",
            "Should I use soap on a pan or a knife?",
        ],
    },
}

CONFLICTS.update(AMENDMENT_1_2)



# --- amendment 1.3 ----------------------------------------------------------
# Two representative mutually_exclusive tuning families. Development previously
# had one, TUNE-03, whose conflict-pair recall of 0.00 could not be told apart
# from a property of the type because its disputed fact is buried in a chunk
# about study leave.

AMENDMENT_1_3: dict[str, dict] = {
    "TUNE-05": {
        "risk": "low",
        "focal": "the dates of the company leave year",
        "cite": ["HR-01#002", "GEN-01#004"],
        "not": [],
        "required": [
            "HR-01 gives 1 April to 31 March",
            "GEN-01 gives 1 January to 31 December",
            DISAGREE,
        ],
        "variants": ["conflict", "contradiction", "escalate"],
        "forbidden": ["the leave year runs from 1 April to 31 March and that is the answer"],
        "gold": (
            "The documents disagree and both are current. HR-01 gives the leave year as "
            "1 April to 31 March; GEN-01 gives 1 January to 31 December. Neither "
            "supersedes the other."
        ),
        "questions": [
            "When does the leave year start and end?",
            "What are the dates of the company leave year?",
            "If I have leave left in March, do I lose it?",
        ],
    },
    "TUNE-06": {
        "risk": "low",
        "focal": "how long a Returns Authorisation number stays valid",
        "cite": ["OPS-02#001", "OPS-03#002"],
        "not": [],
        "required": [
            "OPS-02 gives 21 days",
            "OPS-03 gives 14 days",
            DISAGREE,
        ],
        "variants": ["conflict", "contradiction", "escalate"],
        "forbidden": ["an RA number is valid for 21 days and that is the answer"],
        "gold": (
            "The documents disagree and both are current. OPS-02 issues a Returns "
            "Authorisation number valid for 21 days; OPS-03 states 14 days from issue "
            "and will not process a refund against an expired number. Neither "
            "supersedes the other."
        ),
        "questions": [
            "How long is a Returns Authorisation number valid for?",
            "When does an RA number expire?",
            "My RA number was issued 18 days ago. Can I still use it?",
        ],
    },
}

CONFLICTS.update(AMENDMENT_1_3)


if __name__ == "__main__":
    raise SystemExit(main())
