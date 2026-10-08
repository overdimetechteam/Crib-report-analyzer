#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
crib_report_generator.py
=========================
Reads a raw CRIB ("Corporate Credit Report Plus" or "Consumer Credit Report
Plus") PDF export, computes the portfolio / arrears / commitment analysis
defined in "Sample format of the report.pdf", runs a rule-based risk-assessment layer
on top of that analysis, and writes a single-page, self-contained HTML
credit-risk dashboard (styled after "crib_dashboard.html").

USAGE
-----
    python crib_report_generator.py "PV1234 crib.pdf" -o "PV1234 dashboard.html"
    python crib_report_generator.py "Mark..pdf" -o "Mark dashboard.html"

    # dump the parsed raw data as JSON instead of (or in addition to) the
    # dashboard, useful for double-checking what the script actually read
    # out of the source:
    python crib_report_generator.py "RPS DE SILVA 01.pdf" --json parsed.json

REQUIREMENTS
------------
    pip install pdfplumber

HOW IT WORKS
------------
1.  PARSE    -- `parse_crib_report()` opens the raw CRIB PDF with pdfplumber
    and reconstructs the "List of All Credit Facilities" table, the monthly
    "Payment" (NDIA) table, subject/company details, score history, the
    dishonoured cheque list and the inquiry list, into a plain Python dict.

2.  ANALYSE  -- `analyse()` turns that raw data into every table defined by
    the report template: portfolio breakdowns by product/repayment type,
    vintage buckets, arrears ageing, total CRIB commitments, guaranteed
    facilities, relations/inquiries/cheque-return summaries and a CRIB
    score trend.

3.  ASSESS   -- `assess_risk()` is a rule-based heuristic layer on top of
    `analyse()`'s output. It has no judgement of its own -- it just checks
    the computed numbers against a handful of underwriting-style rules
    (arrears present, guarantor defaults, score deterioration, exposure
    concentration, recent inquiry activity, dishonoured cheques, name-
    spelling variants, ...) and turns each triggered rule into a red flag /
    positive factor / checklist item, then rolls those up into an overall
    decision badge and risk grade. IT IS NOT A CREDIT DECISION -- it is a
    starting point for a human analyst, who should always review the
    underlying facility data before signing off.

4.  RENDER   -- `build_report_html()` fills the computed analysis + risk
    assessment into the dashboard layout (KPI cards, score-trend chart,
    exposure breakdown, facility table, red flags / positive factors,
    recommendation + checklist, sign-off) and writes it out as one
    self-contained HTML file (no external assets, safe to email or archive).

A KNOWN SOURCE-DATA QUIRK
--------------------------
The raw CRIB PDF export embeds a second, visually-hidden copy of the
Dashboard sidebar text on every page (an artifact of how the source
webpage was printed to PDF). That hidden text can literally
character-interleave with real content inside a single word (e.g. a
calendar row's stray digits bleeding into "Security" -> "Sec2u0r2it4y").
The parser works around this with fuzzy, position-anchored label matching
(see `find_label_row`) rather than trusting raw reading-order text
extraction. This is thoroughly tested against the two sample layouts
included in this folder (Corporate: 100% clean; Consumer: a handful of
low-priority fields such as "Currency" may occasionally be left blank
rather than guessed).
"""
import argparse
import html
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from typing import TypedDict, TYPE_CHECKING

try:
    import pdfplumber
except ImportError:
    sys.exit("Missing dependency: pip install pdfplumber")

try:
    from gemini_analyzer import analyze_with_gemini, is_gemini_enabled, get_gemini_settings
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    def analyze_with_gemini(parsed, analysis, risk, model=None):
        """Fallback when gemini_analyzer is not available."""
        return risk
    def is_gemini_enabled() -> bool:
        return False
    def get_gemini_settings():
        return {'gemini_model': 'gemini-2.0-flash'}


DATE_RE = re.compile(r'^\d{2}-[A-Za-z]{3}-\d{4}$')


# ============================================================================
# 1. PARSE -- raw CRIB PDF -> structured dict
# ============================================================================

def page_rows(page, y_tol=2.3):
    """Group a page's words into visual rows (lines), sorted top-to-bottom,
    each row's words sorted left-to-right."""
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    words = sorted(words, key=lambda w: (w['top'], w['x0']))
    rows, cur, ref_top = [], [], None
    for w in words:
        if cur and abs(w['top'] - ref_top) > y_tol:
            rows.append(cur)
            cur, ref_top = [], None
        cur.append(w)
        ref_top = w['top'] if ref_top is None else ref_top
    if cur:
        rows.append(cur)
    out = []
    for r in rows:
        r_sorted = sorted(r, key=lambda w: w['x0'])
        top = sum(w['top'] for w in r_sorted) / len(r_sorted)
        out.append({'top': top, 'words': r_sorted,
                     'text': ' '.join(w['text'] for w in r_sorted)})
    return out


_PUA_RE = re.compile('[--￿\x00]')
_DIGIT_RE = re.compile(r'\d')


def _clean_word(text, strip_digits=False):
    t = _PUA_RE.sub('', text)
    if strip_digits:
        t = _DIGIT_RE.sub('', t)
    return t


def find_label_span(rows, label, start=0):
    """Like find_label_row, but also returns the label's own starting word
    index within the row (needed to read off the label's x0 position for
    column-anchor tables, as opposed to the row's leftmost word overall)."""
    target = ' '.join(label.split()).lower()
    for strip_digits in (False, True):
        for i in range(start, len(rows)):
            words = rows[i]['words']
            for j0 in range(len(words)):
                acc = ''
                for j in range(j0, len(words)):
                    token = _clean_word(words[j]['text'], strip_digits=strip_digits)
                    if not token:
                        continue
                    acc = (acc + ' ' + token).strip() if acc else token
                    accl = acc.lower()
                    if accl == target:
                        return i, j0, j + 1
                    if len(accl) > len(target) + 8 or not target.startswith(accl[:min(len(accl), len(target))]):
                        break
    return None, None, None


def find_label_row(rows, label, start=0, avoid_preceded_by=None):
    """Find the first row (at/after index `start`) whose leading words match
    `label` (case-insensitive, whitespace-normalised). Returns
    (row_index, matched_word_count) or (None, None).

    The source PDFs (consumer/individual report layout) carry a visually
    hidden duplicate text layer that occasionally character-interleaves
    with real content INSIDE a single word token (e.g. a calendar row's
    digits bleeding into "Security" -> "Sec2u0r2it4y"). We therefore try
    an exact match first, and if that fails, retry after stripping
    private-use-area icon glyphs and then stray digits from each word
    before comparing -- values are still read from the original,
    unmodified word text, only the label comparison is fuzzed. We also
    slide the candidate start position across the row, since a hidden
    section heading sharing this row's y-band can shift the real label
    away from word index 0.

    `avoid_preceded_by`, if given, is a set of (lowercased) words that
    disqualify a match if one of them is the word immediately before the
    matched span -- e.g. the hidden dashboard sidebar bleeds a "Total
    Amount in Arrears" / "Current Amount in Arrears" summary line onto
    the per-facility table's page, which would otherwise satisfy a plain
    "Amount in Arrears" search (the row-sliding above happily skips over
    "Total"/"Current" to find the substring). Rejected matches keep
    searching later start positions/rows rather than returning."""
    target = ' '.join(label.split()).lower()
    avoid = {w.lower() for w in avoid_preceded_by} if avoid_preceded_by else None
    for strip_digits in (False, True):
        for i in range(start, len(rows)):
            words = rows[i]['words']
            for j0 in range(len(words)):
                acc = ''
                for j in range(j0, len(words)):
                    token = _clean_word(words[j]['text'], strip_digits=strip_digits)
                    if not token:
                        continue
                    acc = (acc + ' ' + token).strip() if acc else token
                    accl = acc.lower()
                    if accl == target:
                        if avoid:
                            prev = None
                            for k in range(j0 - 1, -1, -1):
                                t = _clean_word(words[k]['text'], strip_digits=strip_digits)
                                if t:
                                    prev = t.lower()
                                    break
                            if prev in avoid:
                                break
                        return i, j + 1
                    if len(accl) > len(target) + 8 or not target.startswith(accl[:min(len(accl), len(target))]):
                        break
    return None, None


def nearest_anchor_map(value_words, anchors, tol=28):
    """Map a list of value words onto the nearest column anchor (x0
    position). Returns dict anchor_index -> joined text. Best for grid
    tables where each column is a single, well-separated token."""
    buckets = defaultdict(list)
    valid = [(i, a) for i, a in enumerate(anchors) if a is not None]
    for w in value_words:
        if not valid:
            continue
        idx, a = min(valid, key=lambda ia: abs(ia[1] - w['x0']))
        if abs(a - w['x0']) <= tol:
            buckets[idx].append(w)
    result = {}
    for idx, ws in buckets.items():
        ws.sort(key=lambda w: w['x0'])
        result[idx] = ' '.join(w['text'] for w in ws)
    return result


def cluster_by_gap(words, gap=10):
    """Split a row's words into contiguous groups (a 'cell' of text such as
    'Insufficient Funds' or 'SEYLAN BANK PLC'), breaking wherever the
    horizontal gap between consecutive words exceeds `gap`."""
    if not words:
        return []
    ws = sorted(words, key=lambda w: w['x0'])
    groups = [[ws[0]]]
    for a, b in zip(ws, ws[1:]):
        if b['x0'] - a['x1'] > gap:
            groups.append([])
        groups[-1].append(b)
    return groups


def map_row_to_columns(words, anchors, gap=10):
    """Cell-group aware column mapping for row-style (header-across-top)
    tables: groups words into cells by proximity, then assigns each whole
    cell to its nearest column anchor. Needed because a multi-word cell's
    later word can sit numerically closer to the NEXT column's anchor than
    to its own column's anchor, which defeats per-word nearest-anchor
    assignment."""
    result = {}
    valid = [(i, a) for i, a in enumerate(anchors) if a is not None]
    for group in cluster_by_gap(words, gap=gap):
        if not valid:
            continue
        idx, _ = min(valid, key=lambda ia: abs(ia[1] - group[0]['x0']))
        text = ' '.join(w['text'] for w in group)
        result[idx] = (result.get(idx, '') + ' ' + text).strip()
    return result


def clean_num(text):
    """Parse a CRIB numeric cell ('12,345', '0', '-', '-', bullet glyphs
    used for blank/N-A cells) into a float, or None."""
    if text is None:
        return None
    t = text.strip()
    if t in ('', '-', '–', '—') or not re.search(r'\d', t):
        return None
    t = t.replace(',', '')
    m = re.search(r'-?\d+(\.\d+)?', t)
    return float(m.group(0)) if m else None


def parse_date(text):
    if not text:
        return None
    t = text.strip()
    if not DATE_RE.match(t):
        return None
    try:
        return datetime.strptime(t, '%d-%b-%Y').date()
    except ValueError:
        return None


FACILITY_FIELDS = [
    ('institution_category', 'Institution Category', False),
    ('member', 'Member', False),
    ('branch', 'Branch', False),
    ('type', 'Credit Facility Type', False),
    ('status', 'Credit Facility Status', False),
    ('ownership', 'Ownership Indicator', False),
    ('currency', 'Currency', False),
    ('granted', 'Amount Granted/Limit', False),
    ('balance', 'Current Balance', False),
    ('arrears', 'Amount in Arrears', False),
    ('installment', 'Installment Amount', False),
    ('written_off', 'Amount Written Off', False),
    ('last_update', 'Last Update Date', False),
    ('first_disbursement', 'First Disbursement Date', False),
    ('last_payment', 'Last Payment Date', False),
    ('trmn_date', 'TRMN/RSHD/RSTR/WRTN', True),   # wraps to a 2nd label line "Date"
    ('end_date', 'Credit Facility End Date', False),
    ('repayment_type', 'Repayment Type', False),
    ('purpose', 'Purpose of Credit Facility', False),
    ('security', 'Security Coverage', False),
    ('guarantee', 'Guarantee Coverage', False),
    ('entity_code', 'Entity Code', False),
]

# Hidden dashboard-sidebar text (see module docstring) bleeds "Total Amount
# in Arrears" / "Current Amount in Arrears" summary lines onto the
# per-facility table's page; these satisfy a plain "Amount in Arrears"
# substring search and must be rejected in favour of the real per-facility
# values row.
FACILITY_FIELD_AVOID_PRECEDING = {
    'arrears': ('Total', 'Current'),
}


def parse_facility_pages(pdf):
    """Scans every page for a 'Credit Facility Index' transposed table and
    returns {facility_index (int): {field: value}}."""
    facilities = {}
    for page in pdf.pages:
        rows = page_rows(page)
        hdr_i, hdr_n = find_label_row(rows, 'Credit Facility Index')
        if hdr_i is None:
            continue
        header_words = rows[hdr_i]['words'][hdr_n:]
        try:
            indices = [int(w['text']) for w in header_words]
        except ValueError:
            continue
        anchors = [w['x0'] for w in header_words]
        cursor = hdr_i + 1
        for field, label, wraps in FACILITY_FIELDS:
            avoid = FACILITY_FIELD_AVOID_PRECEDING.get(field)
            i, n = find_label_row(rows, label, start=cursor, avoid_preceded_by=avoid)
            if i is None:
                continue
            if wraps:
                value_words = rows[i + 1]['words'] if i + 1 < len(rows) else []
                cursor = i + 2
            else:
                value_words = rows[i]['words'][n:]
                cursor = i + 1
            colmap = nearest_anchor_map(value_words, anchors)
            for col_idx, fac_idx in enumerate(indices):
                facilities.setdefault(fac_idx, {})[field] = colmap.get(col_idx)
    return facilities


def parse_payment_pages(pdf):
    """Scans for 'Payment' transposed monthly tables (rows like '2025 Feb')
    keyed by facility index. Returns {facility_index: {'YYYY Mon': value}}.

    A page can carry more than one 'Credit Facility Index' header (the main
    facility-fields table, then the monthly-payment table further down), so
    every occurrence on the page is examined -- not just the first. We also
    can't reliably spot the payment table by looking for a preceding
    'Payment' heading: the hidden dashboard text layer can character-
    interleave with it (e.g. "Payment" + "Utilities" -> "PaUymtileitniets",
    see module docstring), which defeats a plain substring check. Instead we
    detect it structurally -- a header block is a payment table if any row
    beneath it (before the next header, if any) has the unambiguous
    'YYYY Mon' shape."""
    payments = defaultdict(dict)
    year_re = re.compile(r'^(19|20)\d{2}$')
    mon3_re = re.compile(r'^[A-Za]{3}$', re.IGNORECASE)
    for page in pdf.pages:
        rows = page_rows(page)
        headers = []
        search_start = 0
        while True:
            hdr_i, hdr_n = find_label_row(rows, 'Credit Facility Index', start=search_start)
            if hdr_i is None:
                break
            headers.append((hdr_i, hdr_n))
            search_start = hdr_i + 1
        for k, (hdr_i, hdr_n) in enumerate(headers):
            end_i = headers[k + 1][0] if k + 1 < len(headers) else len(rows)
            header_words = rows[hdr_i]['words'][hdr_n:]
            try:
                indices = [int(w['text']) for w in header_words]
            except ValueError:
                continue
            anchors = [w['x0'] for w in header_words]
            block = defaultdict(dict)
            found_month_row = False
            for r in rows[hdr_i + 1:end_i]:
                words = r['words']
                if len(words) < 2:
                    continue
                w0, w1 = words[0]['text'], words[1]['text']
                if year_re.match(w0) and mon3_re.match(w1):
                    found_month_row = True
                    label = f'{w0} {w1}'
                    colmap = nearest_anchor_map(words[2:], anchors)
                    for col_idx, fac_idx in enumerate(indices):
                        block[fac_idx][label] = colmap.get(col_idx)
            if found_month_row:
                for fac_idx, hist in block.items():
                    payments[fac_idx].update(hist)
    return payments


SUBJECT_LABELS_CONSUMER = [
    ('full_name', 'Full Name'), ('date_of_birth', 'Date of Birth'),
    ('gender', 'Gender'), ('citizenship', 'Citizenship'),
    ('employment', 'Employment'), ('profession', 'Profession'),
    ('employer_name', 'Employer Name'), ('business_name', 'Business Name'),
    ('mail_address', 'Mail Address'), ('permanent_address', 'Permanent Address'),
    ('city', 'City'), ('postal_code', 'Postal Code'), ('district', 'District'),
    ('province', 'Province'), ('country', 'Country'),
    ('nic', 'NIC'), ('passport', 'Passport'),
    ('driving_license', 'Driving License Number'),
    ('business_reg_number', 'Business Registration Number'),
    ('business_reg_date', 'Business Registration Date'),
]

SUBJECT_LABELS_CORPORATE = [
    ('company_name', 'Company Name'), ('trade_name', 'Trade Name'),
    ('business_status', 'Business Status'), ('legal_form', 'Legal Form'),
    ('economic_activity_1', 'Economic Activity 1'), ('economic_activity_2', 'Economic Activity 2'),
    ('mail_address', 'Mail Address'), ('permanent_address', 'Permanent Address'),
    ('city', 'City'), ('postal_code', 'Postal Code'), ('district', 'District'),
    ('province', 'Province'), ('country', 'Country'),
    ('phone_number', 'Phone Number'), ('phone_number2', 'Phone Number 2'),
    ('mobile_phone', 'Mobile Phone'), ('fax', 'Fax'), ('web_page', 'Web Page'),
    ('business_reg_number', 'Business Registration Number'),
    ('business_reg_date', 'Business Registration Date'),
    ('vat_reg_number', 'Vat Registration Number'),
]


def _truncate_before_next_label(words, other_labels):
    """Given a run of value words, cut it short as soon as it starts
    spelling out one of `other_labels` (a same-row/next-row neighbouring
    field's label bleeding in, e.g. "Mail Address" values running into
    "Permanent Address")."""
    targets = [' '.join(l.split()).lower() for l in other_labels]
    for k in range(len(words)):
        acc = ''
        for w in words[k:]:
            token = _clean_word(w['text'])
            if not token:
                continue
            acc = (acc + ' ' + token).strip() if acc else token
            accl = acc.lower()
            if accl in targets:
                return words[:k]
            if len(acc) > 40 or not any(t.startswith(accl[:min(len(accl), len(t))]) for t in targets):
                break
    return words


def _row_value_after_label(rows, label, all_labels=(), start=0, max_x_gap=90):
    i, n = find_label_row(rows, label, start=start)
    if i is None:
        return None
    words = rows[i]['words'][n:]
    if not words:
        return None
    other_labels = [l for l in all_labels if l != label]
    out = [words[0]]
    for a, b in zip(words, words[1:]):
        if b['x0'] - a['x1'] > max_x_gap:
            break
        out.append(b)
    out = _truncate_before_next_label(out, other_labels)
    text = ' '.join(w['text'] for w in out).strip()
    if out and i + 1 < len(rows):
        nxt_all = rows[i + 1]['words']
        start_idx = next((k for k, w in enumerate(nxt_all) if abs(w['x0'] - out[0]['x0']) < 6), None)
        nxt = nxt_all[start_idx:] if start_idx is not None else []
        if nxt:
            cont = [nxt[0]]
            for a, b in zip(nxt, nxt[1:]):
                if b['x0'] - a['x1'] > max_x_gap:
                    break
                cont.append(b)
            cont = _truncate_before_next_label(cont, other_labels)
            if cont:
                text += ' ' + ' '.join(w['text'] for w in cont).strip()
    return text.strip()


def parse_subject_info(pdf, report_type):
    labels = SUBJECT_LABELS_CONSUMER if report_type == 'consumer' else SUBJECT_LABELS_CORPORATE
    page = pdf.pages[1]  # subject info is always on page 2 of the report
    rows = page_rows(page)
    info = {}
    all_labels = [lbl for _, lbl in labels]
    for field, label in labels:
        val = _row_value_after_label(rows, label, all_labels=all_labels, max_x_gap=90)
        if val:
            info[field] = val
    return info


MONTH_YEAR_SUFFIX_RE = re.compile(r'^\d{2}$')


def parse_score_history(pdf):
    """Finds the 'Score History' transposed table (Month / Score / Risk
    Grade / Probability of default)."""
    for page in pdf.pages:
        rows = page_rows(page)
        hdr_i, hdr_n = find_label_row(rows, 'Month')
        if hdr_i is None:
            continue
        header_words = rows[hdr_i]['words'][hdr_n:]
        if not header_words:
            continue
        # Each column is printed as two tokens, e.g. "Mar" "24" -- merge them
        # into one "Mar-24" column anchored on the month token's x0, since
        # that's the position the Score/Risk Grade/PD rows below actually
        # align their single value per column to (the year suffix sits at
        # its own, slightly offset x0 that those rows never populate).
        months, anchors = [], []
        k = 0
        while k < len(header_words):
            w = header_words[k]
            nxt = header_words[k + 1] if k + 1 < len(header_words) else None
            if nxt is not None and MONTH_YEAR_SUFFIX_RE.match(nxt['text']):
                months.append(f"{w['text']}-{nxt['text']}")
                k += 2
            else:
                months.append(w['text'])
                k += 1
            anchors.append(w['x0'])
        result = {'months': months, 'score': [], 'risk_grade': [], 'pd': []}
        si, sn = find_label_row(rows, 'Score', start=hdr_i + 1)
        if si is not None:
            colmap = nearest_anchor_map(rows[si]['words'][sn:], anchors)
            result['score'] = [colmap.get(k) for k in range(len(months))]
        gi, gn = find_label_row(rows, 'Risk Grade', start=hdr_i + 1)
        if gi is not None:
            colmap = nearest_anchor_map(rows[gi]['words'][gn:], anchors)
            result['risk_grade'] = [colmap.get(k) for k in range(len(months))]
        pi, pn = find_label_row(rows, 'Probability of default', start=hdr_i + 1)
        if pi is not None:
            colmap = nearest_anchor_map(rows[pi]['words'][pn:], anchors)
            result['pd'] = [colmap.get(k) for k in range(len(months))]
        if result['score']:
            return result
    return None


def parse_dishonoured_cheques(pdf):
    """Row-per-record table: Entity Code | Dishonour Date | Cheque Number |
    Amount LKR | Dishonour Reason | Member | Branch."""
    cheques = []
    anchors = None
    field_names = ['entity_code', 'date', 'cheque_number', 'amount', 'reason', 'member', 'branch']
    for page in pdf.pages:
        rows = page_rows(page)
        hdr_i, hdr_n = find_label_row(rows, 'Entity Code')
        if hdr_i is not None:
            def label_x0(lbl):
                i, j0, _ = find_label_span(rows, lbl, start=hdr_i) # pyright: ignore[reportArgumentType]
                return rows[i]['words'][j0]['x0'] if i is not None else None
            anchors = [label_x0('Entity Code'), label_x0('Dishonour Date'),
                       label_x0('Cheque Number'), label_x0('Amount LKR'),
                       None,  # Reason column anchor filled from the wrapped header below
                       label_x0('Member'), label_x0('Branch')]
            for r in rows[hdr_i:hdr_i + 3]:
                if r['text'].strip() == 'Reason':
                    anchors[4] = r['words'][0]['x0']
            if anchors[4] is None:
                anchors[4] = (anchors[3] or 0) + 90
        if anchors is None:
            continue
        for r in rows:
            words = r['words']
            if len(words) < 3:
                continue
            date_word = next((w for w in words if DATE_RE.match(w['text'])), None)
            if date_word is None:
                continue
            colmap = map_row_to_columns(words, anchors, gap=12)
            rec = {field_names[k]: colmap.get(k) for k in range(len(field_names))}
            if rec.get('date') and DATE_RE.match(rec['date'].split(' ')[0]): # pyright: ignore[reportOptionalMemberAccess]
                cheques.append(rec)
    seen, unique = set(), []
    for c in cheques:
        key = (c.get('entity_code'), c.get('date'), c.get('cheque_number'))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def parse_inquiries_detail(pdf):
    """'Inquiries During the Last N Months' table: Sector | Reason for
    Inquiry | Date of Inquiry. Reconstructed by anchoring on each date
    token and pulling Sector from the same line and Reason from the
    line(s) immediately around it (the printed column headers for this
    particular table are baked into a background image, not selectable
    text, so we can't anchor on them directly)."""
    records = []
    for page in pdf.pages:
        rows = page_rows(page)
        hdr_i, _ = find_label_row(rows, 'Inquiries During the Last')
        if hdr_i is None:
            hdr_i, _ = find_label_row(rows, 'During the Last')
        if hdr_i is None:
            continue
        end_i = len(rows)
        for stop_label in ('Disputes', 'Dishonoured Cheques', 'Utilities'):
            si, _ = find_label_row(rows, stop_label, start=hdr_i + 1)
            if si is not None:
                end_i = min(end_i, si)
        block = rows[hdr_i + 1:end_i]
        for k, r in enumerate(block):
            date_word = next((w for w in r['words'] if DATE_RE.match(w['text'])), None)
            if date_word is None:
                continue
            sector = ' '.join(w['text'] for w in r['words'] if w is not date_word).strip()
            reason_parts = []
            if k > 0 and not any(DATE_RE.match(w['text']) for w in block[k - 1]['words']):
                reason_parts.append(block[k - 1]['text'])
            if k + 1 < len(block) and not any(DATE_RE.match(w['text']) for w in block[k + 1]['words']):
                nxt_words = block[k + 1]['words']
                if not (nxt_words and re.match(r'^License', nxt_words[0]['text'])):
                    reason_parts.append(block[k + 1]['text'])
            records.append({'sector': sector, 'reason': ' '.join(reason_parts).strip(),
                             'date': date_word['text']})
    return records


def parse_counts_page(pdf):
    """Best-effort parse of the clean 'Relations' summary block (Related /
    Involved Subjects) that appears as its own section later in the
    document. Inquiry-period counts are computed downstream from
    `inquiries_detail` instead, which is more reliable."""
    out = {}
    for page in pdf.pages:
        rows = page_rows(page)
        i, _ = find_label_row(rows, 'Related Subjects')
        if i is not None and i > 0:
            nums = [w['text'] for w in rows[i - 1]['words'] if re.fullmatch(r'\d+', w['text'])]
            if nums:
                out['related_subjects'] = int(nums[0])
        i, _ = find_label_row(rows, 'Involved Subjects')
        if i is not None and i > 0:
            nums = [w['text'] for w in rows[i - 1]['words'] if re.fullmatch(r'\d+', w['text'])]
            if nums:
                out['involved_subjects'] = int(nums[-1])
    return out


def detect_report_type(pdf):
    text = pdf.pages[0].extract_text() or ''
    return 'corporate' if 'Corporate Credit Report' in text else 'consumer'


def parse_meta(pdf):
    """Reference Number / Requested by / Report Created / Version sit as a
    4-column header directly below the report title on page 1; the value
    row is the next row after the labels that actually contains a date
    (skipping over the report title line, which can share the same
    y-band)."""
    rows = page_rows(pdf.pages[0])
    meta = {}
    i, n = find_label_row(rows, 'Reference Number')
    if i is not None:
        for k in range(i + 1, min(i + 4, len(rows))):
            texts = [w['text'] for w in rows[k]['words']]
            date_idx = next((m for m, t in enumerate(texts) if DATE_RE.match(t)), None)
            if date_idx is not None and date_idx >= 1:
                meta['reference_number'] = texts[0]
                meta['requested_by'] = texts[date_idx - 1]
                meta['report_created'] = texts[date_idx]
                meta['report_created_time'] = texts[date_idx + 1] if date_idx + 1 < len(texts) else None
                meta['version'] = texts[-1]
                break
    si, sn = find_label_row(rows, 'Searched Name')
    if si is not None:
        meta['searched_name'] = ' '.join(w['text'] for w in rows[si]['words'][sn:] if w['text'] not in ('-', '–'))
    di, dn = find_label_row(rows, 'Searched ID No.')
    if di is not None:
        meta['searched_id'] = ' '.join(w['text'] for w in rows[di]['words'][dn:] if w['text'] not in ('-', '–'))
    return meta


def parse_crib_report(path):
    with pdfplumber.open(path) as pdf:
        report_type = detect_report_type(pdf)
        data = {
            'report_type': report_type,
            'meta': parse_meta(pdf),
            'subject': parse_subject_info(pdf, report_type),
            'score_history': parse_score_history(pdf),
            'facilities': parse_facility_pages(pdf),
            'payments': dict(parse_payment_pages(pdf)),
            'dishonoured_cheques': parse_dishonoured_cheques(pdf),
            'inquiries_detail': parse_inquiries_detail(pdf),
            'counts': parse_counts_page(pdf),
        }
    return data


# ============================================================================
# 2. ANALYSE -- structured dict -> every template table
# ============================================================================

def g(fac, field):
    return clean_num(fac.get(field))


def d(fac, field):
    return parse_date(fac.get(field))


def as_of_date(parsed):
    dt = parse_date(parsed['meta'].get('report_created'))
    return dt or date.today()


def own_active(facilities):
    """Facilities the subject owes on and that are still open (Section [A]:
    'Own Active Facilities')."""
    return {i: f for i, f in facilities.items()
            if (f.get('ownership') or '').strip() in ('OWN', 'JNT')
            and (f.get('status') or '').strip() == 'ACTV'}


def own_settled(facilities):
    return {i: f for i, f in facilities.items()
            if (f.get('ownership') or '').strip() in ('OWN', 'JNT')
            and (f.get('status') or '').strip() == 'SETL'}


def guaranteed(facilities):
    """Section [B]: facilities where the subject is a guarantor."""
    return {i: f for i, f in facilities.items()
            if (f.get('ownership') or '').strip() == 'GRT'}


def pct(part, whole):
    if not whole:
        return None
    return round(100.0 * part / whole, 1)


def vintage_bucket(as_of, disb_date):
    if disb_date is None:
        return 'Unknown'
    days = (as_of - disb_date).days
    if days < 365:
        return 'Within last 01 Year'
    if days < 730:
        return 'Within last 1-2 Years'
    if days < 1095:
        return 'Within the last 2-3 Years'
    return 'Beyond last 3 Years'


def settled_bucket(as_of, ref_date):
    if ref_date is None:
        return 'Unknown'
    days = (as_of - ref_date).days
    if days < 365:
        return 'Last 1 Year'
    if days < 730:
        return 'Within last 1-2 Years'
    if days < 1095:
        return 'Within last 2-3 Years'
    return 'Beyond 3 Years'


def latest_ndia(payments, idx):
    """Most recent non-blank monthly NDIA reading for a facility, used as
    its current 'age' of arrears in the ageing table."""
    hist = payments.get(idx) or payments.get(str(idx))
    if not hist:
        return None
    for label, val in hist.items():  # dict preserves insertion order == chronological, newest first
        n = clean_num(val)
        if n is not None:
            return n
    return None


#newly added for get the exact NDIA
def get_exact_ndia(raw_text):
    if not raw_text:
        return 0
    numbers = re.findall(r'\d+', str(raw_text))
    if numbers:
        return int(numbers[-1]) 
    return 0

def ndia_bucket(n):
    if n is None:
        return None
    if n == 0:
        return '0'
    if n <= 30:
        return '1-30'
    if n <= 60:
        return '31-60'
    if n <= 90:
        return '61-90'
    return '90 and above'


def build_portfolio_analysis(parsed, as_of):
    facilities = parsed['facilities']
    payments = parsed['payments']
    active = own_active(facilities)
    settled = own_settled(facilities)

    # 1.1 Facility wise
    facility_wise = []
    for idx in sorted(active):
        f = active[idx]
        granted, bal = g(f, 'granted'), g(f, 'balance')
        facility_wise.append({
            'facility_no': idx, 'granted': granted, 'os': bal,
            'pct_os_granted': pct(bal, granted) if bal is not None else None,
        })
    total_granted = sum(r['granted'] or 0 for r in facility_wise)
    total_os = sum(r['os'] or 0 for r in facility_wise)

    # 1.2 Product wise
    by_product = defaultdict(lambda: {'count': 0, 'granted': 0.0, 'os': 0.0})
    for f in active.values():
        p = by_product[f.get('type') or 'Unknown']
        p['count'] += 1
        p['granted'] += g(f, 'granted') or 0
        p['os'] += g(f, 'balance') or 0
    product_wise = []
    for ptype, v in sorted(by_product.items()):
        product_wise.append({
            'product': ptype, 'count': v['count'], 'granted': v['granted'],
            'avg': v['granted'] / v['count'] if v['count'] else 0,
            'os': v['os'], 'pct_os': pct(v['os'], total_os),
        })

    # 1.3 Repayment type
    by_repay = defaultdict(lambda: {'count': 0, 'granted': 0.0, 'os': 0.0})
    for f in active.values():
        r = by_repay[f.get('repayment_type') or 'Unknown']
        r['count'] += 1
        r['granted'] += g(f, 'granted') or 0
        r['os'] += g(f, 'balance') or 0
    total_repay_granted = sum(v['granted'] for v in by_repay.values())
    repayment_type = []
    for rtype, v in sorted(by_repay.items()):
        repayment_type.append({
            'repayment_type': rtype, 'count': v['count'], 'granted': v['granted'],
            'avg': v['granted'] / v['count'] if v['count'] else 0,
            'os': v['os'], 'pct_of_granted': pct(v['granted'], total_repay_granted),
        })

    # 1.3(b) Vintage of facilities
    by_vintage = defaultdict(lambda: {'count': 0, 'granted': 0.0, 'os': 0.0})
    for f in active.values():
        b = vintage_bucket(as_of, d(f, 'first_disbursement'))
        v = by_vintage[b]
        v['count'] += 1
        v['granted'] += g(f, 'granted') or 0
        v['os'] += g(f, 'balance') or 0
    vintage_order = ['Within last 01 Year', 'Within last 1-2 Years',
                      'Within the last 2-3 Years', 'Beyond last 3 Years', 'Unknown']
    vintage = []
    for b in vintage_order:
        if b in by_vintage:
            v = by_vintage[b]
            vintage.append({'bucket': b, 'count': v['count'], 'granted': v['granted'],
                             'os': v['os'], 'pct': pct(v['granted'], total_granted)})

    # 1.4 Most recent borrowings (by facility type, non-exclusive 3/6/12 month windows)
    types = sorted({f.get('type') or 'Unknown' for f in active.values()})
    recent = []
    win_totals = {3: [0, 0.0], 6: [0, 0.0], 12: [0, 0.0]}
    for t in types:
        row = {'type': t}
        for months in (3, 6, 12):
            cnt, amt = 0, 0.0
            for f in active.values():
                if (f.get('type') or 'Unknown') != t:
                    continue
                dd = d(f, 'first_disbursement')
                if dd and (as_of - dd).days <= months * 30.44:
                    cnt += 1
                    amt += g(f, 'granted') or 0
            row[f'{months}m_count'], row[f'{months}m_amount'] = cnt, amt
            win_totals[months][0] += cnt
            win_totals[months][1] += amt
        recent.append(row)

    # 2.0 Table B: facilities settled within the last 5 years
    by_settled = defaultdict(lambda: {'count': 0, 'amount': 0.0})
    for f in settled.values():
        ref = d(f, 'end_date') or d(f, 'last_update')
        b = settled_bucket(as_of, ref)
        v = by_settled[b]
        v['count'] += 1
        v['amount'] += g(f, 'granted') or 0
    settled_order = ['Last 1 Year', 'Within last 1-2 Years', 'Within last 2-3 Years', 'Beyond 3 Years', 'Unknown']
    table_b = []
    for b in settled_order:
        if b in by_settled:
            v = by_settled[b]
            table_b.append({'bucket': b, 'count': v['count'], 'amount': v['amount'],
                             'avg': v['amount'] / v['count'] if v['count'] else 0})

    return {
        'facility_wise': facility_wise, 'total_granted': total_granted, 'total_os': total_os,
        'product_wise': product_wise, 'repayment_type': repayment_type, 'vintage': vintage,
        'recent_borrowings': recent, 'recent_totals': win_totals, 'table_b': table_b,
        'active_count': len(active), 'settled_count': len(settled),
    }


def build_arrears_analysis(parsed, as_of):
    facilities = parsed['facilities']
    payments = parsed['payments']
    active = own_active(facilities)

    in_arrears = {i: f for i, f in active.items() if (g(f, 'arrears') or 0) > 0}
    facility_wise = []
    for idx in sorted(in_arrears):
        f = in_arrears[idx]
        os_amt, arrears = g(f, 'balance'), g(f, 'arrears')
        facility_wise.append({
            'facility_no': idx, 'granted': g(f, 'granted'), 'os': os_amt,
            'arrears': arrears, 'age': latest_ndia(payments, idx),
            'pct': pct(arrears, os_amt) if os_amt else None,
        })

    by_product = defaultdict(lambda: {'granted': 0.0, 'os': 0.0, 'arrears': 0.0})
    for f in in_arrears.values():
        p = by_product[f.get('type') or 'Unknown']
        p['granted'] += g(f, 'granted') or 0
        p['os'] += g(f, 'balance') or 0
        p['arrears'] += g(f, 'arrears') or 0
    total_os_arrears = sum(v['os'] for v in by_product.values())
    product_wise = []
    for ptype, v in sorted(by_product.items()):
        product_wise.append({'product': ptype, 'granted': v['granted'], 'os': v['os'],
                              'arrears': v['arrears'], 'pct': pct(v['arrears'], v['os'])})

    # 3.3 Age analysis -- covers ALL own facilities (active + settled), classifying
    # each into a delinquency-age bucket or a status bucket (legal/written-off/
    # terminated/reschedule) based on status flags + amount written off.
    all_own = {i: f for i, f in facilities.items() if (f.get('ownership') or '').strip() in ('OWN', 'JNT')}
    buckets = ['0', '1-30', '31-60', '61-90', '90 and above', 'legal', 'Written-off', 'Terminated', 'Reschedule']
    age = {b: {'count': 0, 'granted': 0.0, 'os': 0.0} for b in buckets}
    for idx, f in all_own.items():
        status = (f.get('status') or '').strip()
        written_off = g(f, 'written_off') or 0
        bucket = None
        if written_off > 0:
            bucket = 'Written-off'
        elif status == 'RSTR':
            bucket = 'Reschedule'
        elif status == 'SETL':
            bucket = None  # closed with nothing outstanding -- excluded from ageing
        else:
            n = latest_ndia(payments, idx)
            if n is None:
                n = 0 if not (g(f, 'arrears') or 0) else None
            bucket = ndia_bucket(n)
        if bucket and bucket in age:
            a = age[bucket]
            a['count'] += 1
            a['granted'] += g(f, 'granted') or 0
            a['os'] += g(f, 'balance') or 0
    total_os_all = sum(a['os'] for a in age.values()) or 1
    for a in age.values():
        a['pct'] = pct(a['os'], total_os_all)

    return {
        'facility_wise': facility_wise, 'product_wise': product_wise,
        'age_analysis': age, 'buckets_order': buckets,
        'in_arrears_count': len(in_arrears),
    }


def build_commitments(parsed):
    facilities = parsed['facilities']
    active = own_active(facilities)
    rows = []
    total_granted = total_os = 0.0
    for idx in sorted(active):
        f = active[idx]
        granted, os_amt = g(f, 'granted') or 0, g(f, 'balance') or 0
        rows.append({'facility_no': idx, 'type': f.get('type'), 'granted': granted,
                     'os': os_amt, 'current_commitment': os_amt})
        total_granted += granted
        total_os += os_amt
    return {'rows': rows, 'total_granted': total_granted, 'total_os': total_os}


def build_guaranteed_analysis(parsed, as_of):
    facilities = parsed['facilities']
    grt = guaranteed(facilities)
    status = []
    for idx in sorted(grt):
        f = grt[idx]
        status.append({'facility_no': idx, 'granted': g(f, 'granted'), 'os': g(f, 'balance'),
                        'arrears': g(f, 'arrears'), 'age': latest_ndia(parsed['payments'], idx)})
    buckets = ['0', '1-30', '31-60', '61-90', '90 and above', 'legal', 'Written-off', 'Terminated', 'Reschedule']
    age = {b: {'count': 0, 'granted': 0.0, 'os': 0.0} for b in buckets}
    for idx, f in grt.items():
        status_flag = (f.get('status') or '').strip()
        written_off = g(f, 'written_off') or 0
        bucket = None
        if written_off > 0:
            bucket = 'Written-off'
        elif status_flag == 'RSTR':
            bucket = 'Reschedule'
        elif status_flag == 'SETL':
            bucket = None
        else:
            bucket = ndia_bucket(latest_ndia(parsed['payments'], idx) or 0)
        if bucket and bucket in age:
            a = age[bucket]
            a['count'] += 1
            a['granted'] += g(f, 'granted') or 0
            a['os'] += g(f, 'balance') or 0
    total_os_all = sum(a['os'] for a in age.values()) or 1
    for a in age.values():
        a['pct'] = pct(a['os'], total_os_all)
    return {'status': status, 'age_analysis': age, 'buckets_order': buckets, 'count': len(grt)}


def build_other_factors(parsed, as_of):
    counts = parsed.get('counts', {})
    inquiries = parsed.get('inquiries_detail', [])
    cheques = parsed.get('dishonoured_cheques', [])

    def within(months, dt):
        return dt is not None and (as_of - dt).days <= months * 30.44

    ref_counts = {}
    for months in (3, 6, 12):
        ref_counts[months] = sum(1 for r in inquiries if within(months, parse_date(r.get('date'))))

    cheque_dates = [parse_date(c.get('date')) for c in cheques]
    cheque_dates = [dtv for dtv in cheque_dates if dtv]
    cheque_amounts = [clean_num(c.get('amount')) for c in cheques]
    cheque_amounts = [a for a in cheque_amounts if a is not None]

    by_reason = defaultdict(int)
    for c in cheques:
        by_reason[(c.get('reason') or 'Unknown').strip()] += 1

    months_span = 1
    if cheque_dates:
        span_days = (max(cheque_dates) - min(cheque_dates)).days
        months_span = max(1, round(span_days / 30.44))
    avg_per_month = len(cheques) / months_span if cheques else 0

    recent6 = sum(1 for dtv in cheque_dates if within(6, dtv))
    prior6 = sum(1 for dtv in cheque_dates if dtv and 6 * 30.44 < (as_of - dtv).days <= 12 * 30.44)
    if recent6 > prior6:
        cheque_trend = 'worsening (more dishonoured cheques in the last 6 months than the preceding 6 months)'
    elif recent6 < prior6:
        cheque_trend = 'improving (fewer dishonoured cheques in the last 6 months than the preceding 6 months)'
    else:
        cheque_trend = 'stable'

    score_hist = parsed.get('score_history')
    score_trend = 'No score history available.'
    if score_hist and score_hist.get('score'):
        vals = [clean_num(v) for v in score_hist['score'] if clean_num(v) not in (None, 0)]
        # filter out any None values before computing min/max to satisfy type checkers
        numeric_vals = [v for v in vals if v is not None]
        if len(numeric_vals) >= 2:
            first, last = numeric_vals[0], numeric_vals[-1]
            if first is not None and last is not None:
                direction = 'improved' if last > first else ('deteriorated' if last < first else 'remained stable')
                score_trend = (f'CRIB score {direction} from {int(first)} to {int(last)} over the reported history '
                                f'(range {int(min(numeric_vals))}-{int(max(numeric_vals))}).')

    return {
        'related_subjects': counts.get('related_subjects', 0),
        'involved_subjects': counts.get('involved_subjects', 0),
        'inquiries_3m': ref_counts[3], 'inquiries_6m': ref_counts[6], 'inquiries_12m': ref_counts[12],
        'cheque_count': len(cheques),
        'cheque_total': sum(cheque_amounts),
        'cheque_most_recent': max(cheque_dates) if cheque_dates else None,
        'cheque_highest': max(cheque_amounts) if cheque_amounts else None,
        'cheque_lowest': min(cheque_amounts) if cheque_amounts else None,
        'cheque_by_reason': dict(by_reason),
        'cheque_avg_per_month': round(avg_per_month, 2),
        'cheque_trend': cheque_trend,
        'score_trend': score_trend,
    }


def analyse(parsed):
    as_of = as_of_date(parsed)
    return {
        'as_of': as_of,
        'portfolio': build_portfolio_analysis(parsed, as_of),
        'arrears': build_arrears_analysis(parsed, as_of),
        'commitments': build_commitments(parsed),
        'guaranteed': build_guaranteed_analysis(parsed, as_of),
        'other': build_other_factors(parsed, as_of),
    }


# ============================================================================
# 3. RENDER -- computed analysis -> filled-in PDF (mirrors the template)
# ============================================================================

def money(v):
    if v is None:
        return '-'
    return f'{v:,.0f}'


def pctstr(v):
    return '-' if v is None else f'{v:.1f}%'


# ============================================================================
# 3. ASSESS -- analysis dict -> rule-based risk assessment
#
# This is a heuristic layer, not a credit decision. Every item below is
# produced by a simple, inspectable rule against numbers already computed
# by `analyse()`. It exists to give a human analyst a fast starting point
# -- flagging what's worth a second look -- not to replace their judgement.
# ============================================================================

def score_series(parsed):
    """Chronological (oldest-first) list of {'month','score','grade','pd'}
    entries, skipping any column with no score reading."""
    sh = parsed.get('score_history')
    if not sh or not sh.get('score'):
        return []
    months = sh.get('months') or []
    scores = sh.get('score') or []
    grades = sh.get('risk_grade') or []
    pds = sh.get('pd') or []
    out = []
    for i, m in enumerate(months):
        sc = clean_num(scores[i]) if i < len(scores) else None
        if sc is None:
            continue
        gr = (grades[i] or '').strip() if i < len(grades) and grades[i] else None
        pdv = clean_num(pds[i]) if i < len(pds) else None
        out.append({'month': m, 'score': sc, 'grade': gr, 'pd': pdv})
    return out


def grade_letter(grade):
    return (grade or '').strip()[:1].upper()


def grade_color_class(grade):
    """CSS value-color class for a KPI card, based on the leading grade letter."""
    l = grade_letter(grade)
    if l in ('A', 'B'):
        return 'v-green'
    if l == 'C':
        return 'v-amber'
    if l:
        return 'v-red'
    return 'v-ink'


def bar_class_for_grade(grade):
    """CSS bar-fill class for the score-trend chart, based on grade letter."""
    l = grade_letter(grade)
    if l == 'A':
        return 'bar-b1'
    if l == 'B':
        return 'bar-b2'
    if l == 'C':
        return 'bar-c1'
    if l:
        return 'bar-d1'
    return 'bar-b2'


def assess_risk(parsed, analysis):
    facilities = parsed['facilities']
    payments = parsed['payments']
    port, arr, grt, oth = (analysis['portfolio'], analysis['arrears'],
                            analysis['guaranteed'], analysis['other'])
    as_of = analysis['as_of']

    scores = score_series(parsed)
    latest = scores[-1] if scores else None

    red_flags, positives, checklist = [], [], []

    # ---- own-facility arrears ----
    if arr['in_arrears_count'] > 0:
        total_arrears_amt = sum((r['arrears'] or 0) for r in arr['facility_wise'])
        red_flags.append(
            f"{arr['in_arrears_count']} of {port['active_count']} active facility(ies) currently in "
            f"arrears (total {money(total_arrears_amt)}).")
        checklist.append('Obtain a written repayment / regularisation plan for the facility(ies) in arrears.')
    elif port['active_count']:
        positives.append(f"Zero arrears across all {port['active_count']} active direct-debtor facility(ies).")

    # ---- guarantor exposure ----
    grt_map = guaranteed(facilities)
    grt_written_off, grt_in_arrears = [], []
    for idx, f in grt_map.items():
        wo = g(f, 'written_off') or 0
        arrears_amt = g(f, 'arrears') or 0
        if wo > 0:
            grt_written_off.append((idx, f, wo))
        elif arrears_amt > 0:
            grt_in_arrears.append((idx, f, arrears_amt))
    if grt_written_off:
        total_wo = sum(wo for _, _, wo in grt_written_off)
        red_flags.append(
            f"Guarantor default: {money(total_wo)} written off on {len(grt_written_off)} facility(ies) "
            f"where the subject is a guarantor.")
        checklist.append('Written confirmation on the guarantor write-off (settled / recovery plan / disputed).')
    if grt_in_arrears:
        red_flags.append(
            f"{len(grt_in_arrears)} guaranteed facility(ies) currently in arrears "
            f"(total {money(sum(a for _, _, a in grt_in_arrears))}) -- contingent exposure.")
        checklist.append('Confirm current status of the guaranteed facility(ies) in arrears.')
    grt_flagged = grt_written_off + grt_in_arrears

    # ---- score / grade trend ----
    if len(scores) >= 2:
        first, last = scores[0], scores[-1]
        if last['score'] < first['score']:
            red_flags.append(
                f"CRIB score deteriorated from {first['score']:.0f} to {last['score']:.0f} over the "
                f"reported history ({first['month']} \u2192 {last['month']}).")
        elif last['score'] > first['score']:
            positives.append(
                f"CRIB score improved from {first['score']:.0f} to {last['score']:.0f} over the reported history.")
        dips = [s['month'] for s in scores if grade_letter(s['grade']) > 'B' and grade_letter(s['grade']) != '']
        if dips and grade_letter(last['grade']) <= 'B' and grade_letter(last['grade']) != '':
            red_flags.append(f"Risk grade dipped below 'B' during {', '.join(dips)}.")
        recent = scores[-3:]
        if len(recent) == 3 and len({round(s['score']) for s in recent}) == 1:
            positives.append(
                f"Score stable at {recent[-1]['score']:.0f} (grade {recent[-1]['grade'] or '-'}) for the "
                f"last 3 reported months.")

    # ---- concentration risk ----
    if port['facility_wise'] and port['total_granted']:
        biggest = max(port['facility_wise'], key=lambda r: r['granted'] or 0)
        share = pct(biggest['granted'] or 0, port['total_granted'])
        if share and share >= 40:
            red_flags.append(
                f"Facility #{biggest['facility_no']} ({money(biggest['granted'])}) makes up {share:.0f}% "
                f"of total granted exposure -- concentration risk.")

    # ---- recent inquiry activity ----
    if oth['inquiries_3m'] >= 3:
        red_flags.append(
            f"{oth['inquiries_3m']} credit inquiries in the last 3 months -- possible new borrowing in progress.")
        checklist.append('Confirm whether any new facility/guarantee has been taken out since the most recent inquiry.')

    # ---- dishonoured cheques ----
    if oth['cheque_count'] > 0:
        red_flags.append(
            f"{oth['cheque_count']} dishonoured cheque(s) on file, totalling {money(oth['cheque_total'])} "
            f"({oth['cheque_trend']}).")
    else:
        positives.append('No dishonoured cheques, disputes, or returned-cheque history on file.')

    # ---- own written-off facilities ----
    wo_bucket = arr['age_analysis'].get('Written-off', {})
    if wo_bucket.get('count'):
        red_flags.append(
            f"{int(wo_bucket['count'])} own facility(ies) written off (O/S {money(wo_bucket['os'])}).")

    # ---- credit file age ----
    disb_dates = [dd for dd in (d(f, 'first_disbursement') for f in facilities.values()) if dd]
    if disb_dates:
        earliest = min(disb_dates)
        years = (as_of - earliest).days / 365.25
        positives.append(f"{years:.0f}-year credit file (earliest facility {earliest.strftime('%b-%Y')}).")

    # ---- revolving-credit utilisation ----
    card_rows = [r for r in port['product_wise'] if 'CARD' in (r['product'] or '').upper()]
    if card_rows:
        granted = sum(r['granted'] for r in card_rows)
        os_amt = sum(r['os'] for r in card_rows)
        util = pct(os_amt, granted) if granted else None
        if util is not None and util < 30:
            positives.append(f"Low utilisation on revolving credit (cards): {util:.0f}% drawn of {money(granted)} limit.")

    # ---- overall decision ----
    severe_arrears = any((r['pct'] or 0) > 50 for r in arr['facility_wise'])
    if grt_written_off or wo_bucket.get('count') or severe_arrears:
        decision, risk_level, risk_class = 'DECLINE / REFER FOR REVIEW', 'HIGH RISK', 'red'
    elif not red_flags:
        decision, risk_level, risk_class = 'APPROVE', 'LOW RISK', 'green'
    else:
        decision, risk_level, risk_class = 'APPROVE WITH CONDITIONS', 'MEDIUM RISK', 'amber'

    if risk_class == 'green':
        recommendation = (
            'Clean repayment conduct across the file with no red flags raised by the automated checks below. '
            'Standard underwriting terms should apply, subject to normal verification.')
    elif risk_class == 'amber':
        recommendation = (
            f"Overall conduct is acceptable, but the automated checks below raised {len(red_flags)} item(s) "
            f"worth a closer look before approval, notably: {red_flags[0]} Recommend approving only once "
            f"these are clarified and any resulting conditions are applied.")
    else:
        recommendation = (
            'The automated checks below raised one or more high-severity items (write-off, guarantor default, '
            'or severe arrears). Recommend declining, or referring to a senior credit officer for manual review, '
            'before proceeding.')

    if latest and latest['pd'] is not None:
        checklist.append(
            f"Price for the assessed grade ({latest['grade'] or '-'}, PD {latest['pd']:.1f}%) rather than best-tier pricing.")
    checklist.append('Decision deadline / requested facility amount: confirm with the applicant (not present in the CRIB report).')

    # ---- KPI roll-up ----
    active_map = own_active(facilities)
    creditors = {(f.get('member') or f.get('institution_category') or '').strip() for f in active_map.values()}
    creditors.discard('')

    ndia_all = []
    for idx, f in facilities.items():
        n = latest_ndia(payments, idx)
        if n:
            ndia_all.append((idx, f, n))
    worst = max(ndia_all, key=lambda t: t[2]) if ndia_all else None
    worst_note = None
    if worst:
        _, wf, _ = worst
        worst_note = ('guarantor a/c' if (wf.get('ownership') or '').strip() == 'GRT' else 'direct debtor a/c')
        if (g(wf, 'written_off') or 0) > 0:
            worst_note += ' (WRTN)'

    kpis = {
        'score': latest['score'] if latest else None,
        'grade': latest['grade'] if latest else None,
        'pd': latest['pd'] if latest else None,
        'total_exposure': port['total_os'],
        'total_granted': port['total_granted'],
        'active_count': port['active_count'],
        'creditor_count': len(creditors),
        'overdue_count': arr['in_arrears_count'],
        'worst_delinquency': worst[2] if worst else 0,
        'worst_delinquency_note': worst_note,
        'recent_inquiries': oth['inquiries_6m'],
    }

    return {
        'decision': decision, 'risk_level': risk_level, 'risk_class': risk_class,
        'red_flags': red_flags, 'positives': positives, 'recommendation': recommendation,
        'checklist': checklist, 'kpis': kpis, 'scores': scores, 'grt_flagged': grt_flagged,
        'ai_summary': None,
    }


# ============================================================================
# 4. RENDER -- analysis + risk assessment -> self-contained HTML dashboard
# ============================================================================

DASHBOARD_CSS = """
  :root{
    --ink:#12181f; --paper:#eef0ee; --card:#ffffff; --line:#dfe3e0; --muted:#5b6570;
    --amber:#b7791f; --amber-bg:#fdf3df; --amber-line:#e8c988;
    --green:#1f7a4d; --green-bg:#e9f6ef; --green-line:#a9d9bf;
    --red:#a3291f; --red-bg:#fbe9e7; --red-line:#eeb7b0;
    --navy:#1c2b3a;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--paper);color:var(--ink);
    font-family:'SF Mono','Courier New',monospace;
    -webkit-font-smoothing:antialiased;font-size:13px;}
  .wrap{max-width:1180px;margin:16px auto;padding:0 12px;}
  .serif{font-family:'Iowan Old Style','Georgia',serif;}

  .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:10px;}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:12px 14px;}
  .panel-title{font-size:9.5px;text-transform:uppercase;letter-spacing:0.6px;
    color:var(--muted);margin:0 0 8px;font-weight:700;}

  .header{grid-column:span 12;border:1px solid var(--line);border-left:6px solid var(--ink);
    padding:16px 22px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;}
  .header .id-block .name{font-family:'Iowan Old Style','Georgia',serif;font-size:19px;font-weight:700;color:var(--ink);}
  .header .id-block .meta{font-size:11px;color:var(--muted);margin-top:2px;}
  .header .decision-block{display:flex;align-items:center;gap:14px;}
  .decision-badge{font-size:24px;font-weight:700;letter-spacing:0.4px;}
  .risk-pill{font-size:11.5px;font-weight:700;padding:5px 12px;border-radius:3px;color:#fff;white-space:nowrap;}

  .kpi{grid-column:span 2;text-align:center;}
  .kpi .label{font-size:9px;text-transform:uppercase;color:var(--muted);letter-spacing:0.4px;margin-bottom:5px;}
  .kpi .value{font-size:19px;font-weight:700;}
  .kpi .sub{font-size:9.5px;color:var(--muted);margin-top:2px;}
  .v-green{color:var(--green);} .v-amber{color:var(--amber);} .v-red{color:var(--red);} .v-ink{color:var(--ink);}

  .chart-panel{grid-column:span 7;}
  .bars{display:flex;align-items:flex-end;gap:5px;height:110px;margin-top:6px;}
  .bar-col{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;}
  .bar{width:100%;border-radius:2px 2px 0 0;}
  .bar-b1{background:#2e9e63;}
  .bar-b2{background:var(--green);}
  .bar-c1{background:var(--amber);}
  .bar-d1{background:var(--red);}
  .bar-label{font-size:8px;color:var(--muted);margin-top:4px;}
  .bar-score{font-size:8.5px;color:var(--ink);font-weight:700;margin-bottom:2px;}
  .legend{display:flex;gap:14px;margin-top:8px;font-size:10px;color:var(--muted);flex-wrap:wrap;}
  .legend span.dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px;vertical-align:middle;}

  .guarantor-track{margin-top:10px;padding-top:8px;border-top:1px dashed var(--line);}
  .guarantor-track .track-label{font-size:9.5px;color:var(--red);font-weight:700;margin-bottom:4px;}
  .guarantor-bars{display:flex;align-items:flex-end;gap:2px;height:40px;}
  .g-bar{flex:1;background:var(--red);border-radius:1px 1px 0 0;opacity:0.85;}

  .exposure-panel{grid-column:span 5;}
  .exp-row{margin-bottom:9px;}
  .exp-row .exp-label{display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px;}
  .exp-row .exp-label b{font-weight:700;}
  .exp-bar-bg{background:#eef0ee;border-radius:3px;height:14px;overflow:hidden;}
  .exp-bar-fill{height:100%;border-radius:3px;}
  .exp-by-sector{margin-top:10px;padding-top:8px;border-top:1px dashed var(--line);font-size:10.5px;color:var(--muted);}
  .exp-by-sector .sector-row{display:flex;justify-content:space-between;padding:2px 0;}

  .table-panel{grid-column:span 12;}
  table{width:100%;border-collapse:collapse;font-size:11.5px;}
  th{text-align:left;font-size:9.5px;text-transform:uppercase;color:var(--muted);
    border-bottom:1px solid var(--ink);padding:4px 8px;letter-spacing:0.3px;}
  td{padding:4px 8px;border-bottom:1px solid var(--line);}
  tr.flag-row td{background:var(--red-bg);color:var(--red);font-weight:700;}
  .pill{font-size:9.5px;padding:2px 7px;border-radius:3px;font-weight:700;display:inline-block;}
  .pill-active{background:var(--green-bg);color:var(--green);}
  .pill-written{background:var(--red-bg);color:var(--red);}
  .pill-settled{background:#eef0ee;color:var(--muted);}

  .flags-panel{grid-column:span 6;}
  ul.flags{list-style:none;margin:0;padding:0;}
  ul.flags li{font-size:11.5px;line-height:1.5;padding:4px 0 4px 20px;position:relative;}
  ul.flags.red li::before{content:"\\26A0";position:absolute;left:0;color:var(--red);}
  ul.flags.good li::before{content:"\\2713";position:absolute;left:0;color:var(--green);font-weight:700;}

  .reco-panel{grid-column:span 12;background:#fafbf9;}
  .reco-panel p{font-size:12.5px;line-height:1.5;margin:0 0 8px;font-family:'Iowan Old Style','Georgia',serif;}
  .checklist{list-style:none;margin:0;padding:0;columns:2;column-gap:24px;}
  .checklist li{font-size:11.5px;line-height:1.6;padding-left:18px;position:relative;break-inside:avoid;}
  .checklist li::before{content:"\\2610";position:absolute;left:0;color:var(--amber);}

  .signoff-panel{grid-column:span 12;display:grid;grid-template-columns:1fr 1fr;gap:20px;}
  .signoff-panel .field{font-size:10.5px;color:var(--muted);}
  .signoff-panel .line{border-top:1px solid var(--ink);margin-top:24px;padding-top:3px;}

  .footnote{grid-column:span 12;font-size:9.5px;color:var(--muted);padding:2px 4px;}

  @media print{ body{background:#fff;} .panel{box-shadow:none;} }
  @media (max-width:800px){
    .kpi{grid-column:span 4;}
    .chart-panel,.exposure-panel,.flags-panel{grid-column:span 12;}
  }
"""

PALETTE = ['var(--navy)', 'var(--amber)', 'var(--green)', '#7a5c9e', '#2e9e63', 'var(--muted)']


def esc(x):
    return html.escape(str(x)) if x is not None else '-'


def build_kpi_html(kpis):
    def card(label, value, sub, val_cls='v-ink'):
        return (f'<div class="panel kpi"><div class="label">{esc(label)}</div>'
                f'<div class="value {val_cls}">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>')

    score_cls = grade_color_class(kpis['grade'])
    overdue_cls = 'v-green' if not kpis['overdue_count'] else 'v-red'
    worst_cls = 'v-green' if not kpis['worst_delinquency'] else 'v-red'
    score_sub = (f"Grade {kpis['grade'] or '-'} \u00b7 PD {kpis['pd']:.1f}%" if kpis['pd'] is not None
                 else f"Grade {kpis['grade'] or '-'}")
    cards = [
        card('Credit Score', f"{kpis['score']:.0f}" if kpis['score'] is not None else '-', score_sub, score_cls),
        card('Total Exposure', f"{kpis['total_exposure']:,.0f}",
             f"LKR \u00b7 of {kpis['total_granted']:,.0f} granted"),
        card('Active Facilities', kpis['active_count'], f"{kpis['creditor_count']} creditor(s)"),
        card('Overdue Accounts', kpis['overdue_count'], 'direct debtor', overdue_cls),
        card('Worst Delinquency', f"{kpis['worst_delinquency']:.0f}d",
             kpis['worst_delinquency_note'] or '-', worst_cls),
        card('Recent Inquiries', kpis['recent_inquiries'], 'last 6 months',
             'v-amber' if kpis['recent_inquiries'] >= 3 else 'v-ink'),
    ]
    return ''.join(cards)


def build_score_chart_html(scores):
    if not scores:
        return '<p class="sub">No score history available in this report.</p>'
    vals = [s['score'] for s in scores]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1)
    bars = []
    for s in scores:
        heightpct = 20 + 75 * ((s['score'] - lo) / span)
        cls = bar_class_for_grade(s['grade'])
        month_label = esc(s['month']).replace("'", '').replace('-', '')
        bars.append(
            f'<div class="bar-col"><div class="bar-score">{s["score"]:.0f}</div>'
            f'<div class="bar {cls}" style="height:{heightpct:.0f}%"></div>'
            f'<div class="bar-label">{month_label}</div></div>')
    return '<div class="bars">' + ''.join(bars) + '</div>'


def build_legend_html(scores):
    color_map = {'bar-b1': '#2e9e63', 'bar-b2': 'var(--green)', 'bar-c1': 'var(--amber)', 'bar-d1': 'var(--red)'}
    label_map = {'bar-b1': 'Grade A \u2014 Low Risk', 'bar-b2': 'Grade B \u2014 Low Risk',
                 'bar-c1': 'Grade C \u2014 Average Risk', 'bar-d1': 'Grade D/E \u2014 Elevated Risk'}
    seen, items = set(), []
    for s in scores:
        cls = bar_class_for_grade(s['grade'])
        if cls in seen:
            continue
        seen.add(cls)
        items.append(f'<div><span class="dot" style="background:{color_map[cls]}"></span>{label_map[cls]}</div>')
    return '<div class="legend">' + ''.join(items) + '</div>' if items else ''


def build_guarantor_track_html(parsed, grt_flagged):
    if not grt_flagged:
        return ''
    idx, f, _ = max(grt_flagged, key=lambda t: t[2])
    hist = parsed['payments'].get(idx) or parsed['payments'].get(str(idx)) or {}
    chrono = list(hist.items())[::-1]  # source dict is newest-first; flip to chronological
    vals = [(m, get_exact_ndia(v)) for m, v in chrono if v is not None]
    vals = [v for v in vals if v[1]]
    if not vals:
        return ''
    max_n = max(n for _, n in vals) or 1
    bars = ''.join(
        f'<div class="g-bar" style="height:{max(4, 100 * n / max_n):.0f}%" title="{esc(m)}: {n}d"></div>'
        for m, n in vals)
    status = 'written off' if (g(f, 'written_off') or 0) > 0 else 'in arrears'
    label = f"Guarantor Facility #{idx} \u2014 Days in Arrears ({vals[0][0]} \u2192 {vals[-1][0]}, {status})"
    return (f'<div class="guarantor-track"><div class="track-label">\u26a0 {esc(label)}</div>'
            f'<div class="guarantor-bars">{bars}</div></div>')


def build_exposure_html(port):
    rows = sorted(port['product_wise'], key=lambda r: r['os'] or 0, reverse=True)
    total_os = port['total_os'] or 1
    out = []
    for i, r in enumerate(rows):
        share = pct(r['os'], total_os) or 0
        color = PALETTE[i % len(PALETTE)]
        label = f"{esc(r['product'])} (\u00d7{r['count']})" if r['count'] and r['count'] > 1 else esc(r['product'])
        out.append(
            '<div class="exp-row">'
            f'<div class="exp-label"><span>{label}</span><b>{money(r["os"])} ({share:.1f}%)</b></div>'
            f'<div class="exp-bar-bg"><div class="exp-bar-fill" style="width:{share:.1f}%;background:{color};"></div></div>'
            '</div>')
    return ''.join(out) if out else '<p class="sub">No active facilities found.</p>'


def build_sector_html(facilities):
    active = own_active(facilities)
    by_cat, total = defaultdict(float), 0.0
    for f in active.values():
        cat = (f.get('institution_category') or 'Unknown').strip() or 'Unknown'
        os_amt = g(f, 'balance') or 0
        by_cat[cat] += os_amt
        total += os_amt
    if not total:
        return ''
    rows = ''.join(
        f'<div class="sector-row"><span>By Institution Type \u2014 {esc(cat)}</span>'
        f'<b>{money(amt)} ({pct(amt, total):.1f}%)</b></div>'
        for cat, amt in sorted(by_cat.items(), key=lambda kv: -kv[1]))
    return f'<div class="exp-by-sector">{rows}</div>'


def build_facility_table_rows(parsed):
    rows = []
    for idx in sorted(parsed['facilities']):
        f = parsed['facilities'][idx]
        ownership = (f.get('ownership') or '').strip()
        if ownership not in ('OWN', 'JNT', 'GRT'):
            continue
        status = (f.get('status') or '').strip()
        wo = g(f, 'written_off') or 0
        if wo > 0:
            pill, pill_cls = 'WRITTEN OFF', 'pill-written'
        elif status == 'ACTV':
            pill, pill_cls = 'ACTIVE', 'pill-active'
        elif status == 'SETL':
            pill, pill_cls = 'SETTLED', 'pill-settled'
        else:
            continue  # skip states we can't confidently classify (e.g. legal/terminated edge cases)
        rows.append({
            'institution': f.get('member') or f.get('institution_category') or '-',
            'type': (f.get('type') or '-') + (' (Guarantor)' if ownership == 'GRT' else ''),
            'granted': g(f, 'granted'), 'os': g(f, 'balance'),
            'pill': pill, 'pill_class': pill_cls, 'flag': wo > 0,
        })
    return rows


def build_facility_table_html(rows):
    if not rows:
        return '<tr><td colspan="5">No active, settled, or written-off facilities found.</td></tr>'
    out = []
    for r in rows:
        tr_cls = ' class="flag-row"' if r['flag'] else ''
        out.append(
            f'<tr{tr_cls}><td>{esc(r["institution"])}</td><td>{esc(r["type"])}</td>'
            f'<td>{money(r["granted"])}</td><td>{money(r["os"])}</td>'
            f'<td><span class="pill {r["pill_class"]}">{esc(r["pill"])}</span></td></tr>')
    return ''.join(out)


def build_list_html(items, cls):
    items = items or ['None identified by the automated checks.']
    return '<ul class="flags ' + cls + '">' + ''.join(f'<li>{esc(i)}</li>' for i in items) + '</ul>'


def build_checklist_html(items):
    items = items or ['No outstanding conditions identified.']
    return '<ul class="checklist">' + ''.join(f'<li>{esc(i)}</li>' for i in items) + '</ul>'


def build_report_html(parsed, analysis, risk, out_path):
    subject, meta = parsed.get('subject', {}), parsed.get('meta', {})
    name = subject.get('full_name') or subject.get('company_name') or meta.get('searched_name') or 'Unknown Subject'
    id_no = subject.get('nic') or subject.get('business_reg_number') or meta.get('searched_id') or '-'
    id_label = 'Reg. No' if parsed.get('report_type') == 'corporate' else 'NIC'
    report_date = meta.get('report_created') or '-'
    ref = meta.get('reference_number') or '-'

    risk_class = risk['risk_class']
    accent = {'green': 'var(--green)', 'amber': 'var(--amber)', 'red': 'var(--red)'}[risk_class]
    bg = {'green': 'var(--green-bg)', 'amber': 'var(--amber-bg)', 'red': 'var(--red-bg)'}[risk_class]
    line = {'green': 'var(--green-line)', 'amber': 'var(--amber-line)', 'red': 'var(--red-line)'}[risk_class]
    emoji = {'green': '\U0001f7e2', 'amber': '\U0001f7e1', 'red': '\U0001f534'}[risk_class]

    facility_rows = build_facility_table_rows(parsed)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Credit Risk Dashboard \u2014 {esc(name)}</title>
<style>
{DASHBOARD_CSS}
</style>
</head>
<body>
<div class="wrap">
<div class="grid">

  <div class="header" style="background:{bg};border-color:{line};border-left-color:{accent};">
    <div class="id-block">
      <div class="name">{esc(name)}</div>
      <div class="meta">{esc(id_label)} {esc(id_no)} &nbsp;\u00b7&nbsp; Report date {esc(report_date)} &nbsp;\u00b7&nbsp; Ref {esc(ref)}<br>
      Purpose: Not specified in report &nbsp;\u00b7&nbsp; Facility requested: Not available in report</div>
    </div>
    <div class="decision-block">
      <div class="decision-badge" style="color:{accent};">{esc(risk['decision'])}</div>
      <div class="risk-pill" style="background:{accent};">{emoji} {esc(risk['risk_level'])}</div>
    </div>
  </div>

  {build_kpi_html(risk['kpis'])}

  <div class="panel chart-panel">
    <div class="panel-title">Payment Behavior \u2014 Score Trend</div>
    {build_score_chart_html(risk['scores'])}
    {build_legend_html(risk['scores'])}
    {build_guarantor_track_html(parsed, risk['grt_flagged'])}
  </div>

  <div class="panel exposure-panel">
    <div class="panel-title">Exposure Breakdown \u2014 By Facility Type (Active, LKR)</div>
    {build_exposure_html(analysis['portfolio'])}
    {build_sector_html(parsed['facilities'])}
  </div>

  <div class="panel table-panel">
    <div class="panel-title">Facility Table \u2014 Active, Settled &amp; Written-Off</div>
    <table>
      <tr><th>Institution</th><th>Type</th><th>Granted (LKR)</th><th>Outstanding (LKR)</th><th>Status</th></tr>
      {build_facility_table_html(facility_rows)}
    </table>
  </div>

  <div class="panel flags-panel">
    <div class="panel-title">Red Flags</div>
    {build_list_html(risk['red_flags'], 'red')}
  </div>
  <div class="panel flags-panel">
    <div class="panel-title">Positive Factors</div>
    {build_list_html(risk['positives'], 'good')}
  </div>

  <div class="panel reco-panel">
    <div class="panel-title">Recommendation</div>
    <p>{esc(risk['recommendation'])}</p>
    {build_checklist_html(risk['checklist'])}
  </div>

  <div class="panel signoff-panel">
    <div class="field">Analyst<div class="line">Name / Date</div></div>
    <div class="field">Approver<div class="line">Name / Date</div></div>
  </div>

  <div class="footnote">Source: CRIB {esc((parsed.get('report_type') or '').title())} Credit Report Plus. Decision badge and
  flags below are generated by automated, rule-based checks -- not a credit decision. A human analyst should review the
  underlying facility data (or the --json export of this tool) before approving.</div>

</div>
</div>
</body>
</html>
"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(doc)


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input_pdf', help='Path to the raw CRIB PDF export')
    ap.add_argument('-o', '--output', help='Output dashboard HTML path (default: <input>_dashboard.html)')
    ap.add_argument('--json', help='Also/instead write the raw parsed data as JSON to this path')
    ap.add_argument('--no-html', action='store_true', help='Skip HTML dashboard generation (only useful with --json)')
    ap.add_argument('--gemini', action='store_true', help='Generate an AI summary with Gemini via LangChain and LangGraph')
    ap.add_argument('--gemini-model', default=get_gemini_settings().get('gemini_model', 'gemini-2.0-flash'), help='Gemini model name to use')
    args = ap.parse_args()

    print(f'Parsing {args.input_pdf} ...')
    parsed = parse_crib_report(args.input_pdf)
    print(f'  report type: {parsed["report_type"]}')
    print(f'  facilities found: {len(parsed["facilities"])}')
    print(f'  dishonoured cheques found: {len(parsed["dishonoured_cheques"])}')
    print(f'  inquiries found: {len(parsed["inquiries_detail"])}')

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(parsed, f, indent=2, default=str, ensure_ascii=False)
        print(f'Wrote parsed data to {args.json}')

    if not args.no_html:
        analysis = analyse(parsed)
        risk = assess_risk(parsed, analysis)
        if args.gemini:
            if not GEMINI_AVAILABLE:
                print('  Gemini analysis skipped: install langchain, langchain-google-genai, and langgraph')
            elif not is_gemini_enabled():
                print('  Gemini analysis skipped: no API key configured -- set google_api_key in .env')
            else:
                risk = analyze_with_gemini(parsed, analysis, risk, model=args.gemini_model)
                print('  Gemini analysis applied to red flags, positives, and recommendation')
        out_path = args.output or re.sub(r'\.pdf$', '', args.input_pdf, flags=re.I) + '_dashboard.html'
        build_report_html(parsed, analysis, risk, out_path)
        print(f'  decision: {risk["decision"]} ({risk["risk_level"]})')
        print(f'  red flags: {len(risk["red_flags"])}, positive factors: {len(risk["positives"])}')
        print(f'Wrote dashboard to {out_path}')


if __name__ == '__main__':
    main()
