"""Independently authored, non-confidential supply-contract fixtures for live tests.

Neither the assignment questions nor any sample-contract text is sent to a
hosted provider by this fixture. These cases test the generic evidence engine,
not accuracy on the confidential assignment.
"""
from pathlib import Path
from textwrap import wrap

from core.rubric.schema import Question, SubRequirement
from eval.variants import _write_text_pdf


GROUPS = [
    ("Delivery", [
        ("weekly_dispatch", "The supplier shall dispatch confirmed orders every Monday."),
        ("sealed_cartons", "Every dispatched carton must be sealed before carrier collection."),
        ("order_labels", "Each carton must display the customer's order number."),
        ("carrier_receipts", "The supplier shall obtain a collection receipt from the carrier."),
        ("delay_notice", "The supplier must notify the customer of a delivery delay within one business day."),
    ]),
    ("Billing", [
        ("monthly_invoices", "The supplier shall issue invoices monthly."),
        ("electronic_invoices", "Invoices must be sent electronically to the customer's billing mailbox."),
        ("itemized_charges", "Each invoice must separately itemize product and transport charges."),
        ("invoice_currency", "All invoice amounts shall be denominated in US dollars."),
        ("payment_window", "Payment shall be due thirty days after receipt of a valid invoice."),
    ]),
    ("Support", [
        ("weekday_desk", "The supplier must operate a customer support desk each weekday."),
        ("telephone_channel", "The support desk shall accept customer requests by telephone."),
        ("email_channel", "The support desk shall also accept requests by email."),
        ("escalation_owner", "The supplier must name an escalation manager for unresolved requests."),
        ("ticket_numbers", "Every support request shall receive a unique ticket number."),
    ]),
    ("Records", [
        ("order_retention", "The supplier must retain order records for at least one year."),
        ("receipt_archive", "Carrier collection receipts shall be archived by the supplier."),
        ("stock_counts", "The supplier must keep a record of each warehouse stock count."),
        ("return_dates", "The supplier must record the date on which each returned shipment is received."),
        ("invoice_archive", "The supplier shall archive every issued invoice."),
    ]),
    ("Packaging", [
        ("recyclable_cartons", "Outer cartons must be made from recyclable material."),
        ("no_glass", "Glass packing material shall not be used."),
        ("pallet_straps", "Palletized shipments must be secured with straps."),
        ("returnable_containers", "The supplier must accept reusable transport containers returned by the customer."),
        ("moisture_barrier", "Each shipment must include a moisture barrier around its contents."),
    ]),
]


def rubric():
    return {f"Q{i}": Question(f"Q{i}", title, f"Does the supply contract require the listed {title.lower()} obligations?", [
        SubRequirement(check_id, sentence, facets=[sentence])
        for check_id, sentence in checks
    ]) for i, (title, checks) in enumerate(GROUPS, 1)}


def write_contract(path: Path, *, omit: str | None = None, renumber: bool = False) -> Path:
    lines = ["Fictional supply agreement for automated testing", "All following terms are binding on the parties.", ""]
    for number, (title, checks) in enumerate(GROUPS, 1):
        lines.append(f"{number + (70 if renumber else 0)}. {title}")
        for check_id, sentence in checks:
            if check_id != omit:
                lines.extend(wrap(sentence, width=88))
        lines.append("")
    _write_text_pdf(lines, path)
    return path
