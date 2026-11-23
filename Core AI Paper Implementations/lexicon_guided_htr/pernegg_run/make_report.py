"""Generate the methodology + Pernegg comparison PDF report."""
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

OUT = Path(__file__).parent / "Lexicon_Guided_HTR_Report.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1x", parent=styles["Heading1"], spaceBefore=14)
H2 = ParagraphStyle("H2x", parent=styles["Heading2"], spaceBefore=10)
BODY = ParagraphStyle("Bodyx", parent=styles["Normal"], fontSize=10,
                      leading=14, spaceAfter=6)
BULLET = ParagraphStyle("Bull", parent=BODY, leftIndent=14, bulletIndent=4)
CAPTION = ParagraphStyle("Cap", parent=styles["Normal"], fontSize=8.5,
                         textColor=colors.grey, spaceAfter=10)

doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=20 * mm, rightMargin=20 * mm,
                        topMargin=18 * mm, bottomMargin=18 * mm,
                        title="Lexicon-Guided HTR — Methodology and Pernegg Results")
S = []

S.append(Paragraph("Lexicon-Guided HTR for Austrian Parish Registers", styles["Title"]))
S.append(Paragraph(
    f"Pre-OCR context injection from the Reichstelefonbuch 1942 · "
    f"SoDa Labs Austrian Register Project · {date.today().strftime('%d %B %Y')}",
    CAPTION))

S.append(Paragraph("1&nbsp;&nbsp;How the auxiliary vocabulary feeds the per-mask candidates", H1))
S.append(Paragraph(
    "The pass-2 prompt guides the vision-language model with two different "
    "mechanisms:", BODY))
S.append(Paragraph(
    "• <b>Bulk lists</b> — the parish-scoped vocabulary printed by category: "
    "surnames attested in Pernegg itself (1-hop), surnames from Steiermark "
    "(2-hop), given names, place names, occupations, and the static calendar/"
    "formulae terms (Jänner, Feber, gestorben, begraben …). The model sees "
    "these as general context.", BULLET))
S.append(Paragraph(
    "• <b>Per-mask candidate lines</b> — for each word the model itself marked "
    "uncertain in pass 1 (wrapped in [[…]] masks), the pipeline fuzzy-matches "
    "that specific garbled token against the lexicon pools and prints a "
    "targeted suggestion, e.g. <i>[[Hu6er]] → Huber, Heuer</i>. This is a much "
    "stronger signal than a word sitting somewhere in a 250-word list: the "
    "model only has to <i>verify</i> a named candidate against the strokes, "
    "not search for it.", BULLET))
S.append(Paragraph(
    "The change in question wired the calendar/formulae vocabulary into the "
    "second mechanism, not just the first. Previously the per-mask matcher "
    "searched only RTB-derived pools, which contain no month names, so a "
    "masked month could never receive a targeted suggestion. Caveat: for the "
    "actual error <i>Feinnet → Jänner</i> the edit distance is so large that "
    "the fuzzy matcher will not propose it even now — that case is carried by "
    "the bulk calendar list plus the model re-reading the pixels. The "
    "per-mask mechanism fires for nearer misses such as <i>Jämmer</i> or "
    "<i>Jänuer</i>.", BODY))

S.append(Paragraph("2&nbsp;&nbsp;The comparison harness (pernegg_compare.py)", H1))
S.append(Paragraph(
    "The systems under comparison produce differently shaped outputs — the "
    "previous pipeline wrote xlsx tables, the new one writes Markdown. A "
    "naive whole-page character error rate would mostly measure formatting "
    "differences, not reading accuracy. The harness therefore evaluates at "
    "the <b>entity level</b>:", BODY))
S.append(Paragraph(
    "1. The ground truth is 55 entity fields (name, occupation, place, cause "
    "of death for each of the 14 records on Seite 11-12 of Sterberegister "
    "8758, year 1895), transcribed manually from 1.4× upscaled image crops.",
    BULLET))
S.append(Paragraph(
    "2. For each ground-truth string (e.g. “Leopold Grabmayer”) rapidfuzz "
    "alignment locates the best-matching substring anywhere in a system's "
    "full output text, wherever and however it is laid out.", BULLET))
S.append(Paragraph(
    "3. CER = character edit distance between the aligned span and the ground "
    "truth, normalised by ground-truth length; WER likewise over words; "
    "micro-averaged across all 55 fields. The metric is layout-independent "
    "and focused exactly on what the lexicon method claims to improve. "
    "<font face='Courier'>--details</font> prints every aligned span for "
    "audit.", BULLET))

S.append(Paragraph("3&nbsp;&nbsp;Post-OCR correction of the existing Pernegg output", H1))
S.append(Paragraph(
    "“Fix the already-digitised Pernegg surnames using the RTB surnames of "
    "that region” is the classic post-OCR correction approach. It is now "
    "implemented (<font face='Courier'>postocr_correct.py</font>) as a "
    "comparison arm: it takes the existing Seite 11-12 xlsx (read-only), "
    "finds capitalised words that are not valid lexicon entries, and "
    "fuzzy-replaces them with the closest Pernegg/Steiermark-scoped RTB "
    "entry when the match is confident (ratio ≥ 86) and unambiguous "
    "(margin ≥ 3 over the runner-up). Measured result:", BODY))

tbl = Table([["system", "CER", "WER"],
             ["prev (old single-pass Chandra)", "0.250", "0.642"],
             ["prev + post-OCR lexicon fix", "0.249", "0.653"],
             ["baseline (new pipeline, pass 1 only)", "pending server run", ""],
             ["guided (lexicon-guided final)", "pending server run", ""]],
            colWidths=[95 * mm, 35 * mm, 30 * mm])
tbl.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.92)),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.97)]),
]))
S.append(tbl)
S.append(Spacer(1, 6))
S.append(Paragraph(
    "Entity-level scores against 55 manually transcribed ground-truth fields, "
    "Pernegg 8758 Sterberegister Seite 11-12 (1895).", CAPTION))
S.append(Paragraph(
    "The post-OCR arm made only two replacements on the whole page "
    "(<i>Krebs- → Krebs</i>; <i>Tuberculose → Tuberkulose</i>, which was in "
    "fact wrong — the register uses the period spelling). The large errors "
    "were untouchable for two distinct reasons:", BODY))
S.append(Paragraph(
    "• <b>Cascade failure</b> — <i>Feinnet</i> (Jänner), <i>Güldingler</i> "
    "(Auszügler), <i>Stumpfmagd</i> (Dienstmagd), <i>Kropobau</i> "
    "(Kropsbauer) land too far in edit space from any lexicon entry; no safe "
    "threshold reaches them without producing false replacements elsewhere.",
    BULLET))
S.append(Paragraph(
    "• <b>Valid-but-wrong words</b> — the old output reads <i>Pommer</i> "
    "where the page (probably) says <i>Sommer</i>. Both are real RTB "
    "surnames, so no text-only method can fix it; only re-examining the "
    "pixels can decide. This is precisely why pass 2 of the new pipeline "
    "sends the image again instead of patching text.", BULLET))

S.append(Paragraph("4&nbsp;&nbsp;What Stage A (build_lexicon.py) built", H1))
S.append(Paragraph(
    "Stage A is the one-off compilation of the RTB digitisation into the "
    "reference resource the pipeline retrieves from. It reads the five "
    "<font face='Courier'>master_RTB_*_clean.xlsx</font> files (1.88 M rows "
    "of region heading + transcribed text) and, per row, parses the region "
    "heading (“Pernegg (Steiermark)” → town <i>Pernegg</i>, state "
    "<i>Steiermark</i>) and the entry (“Jäger, Richard, Viehandlung.” → "
    "surname <i>Jäger</i>, given name <i>Richard</i>, occupation "
    "<i>Viehandlung</i>) using shape heuristics plus filters that reject "
    "institutional entries (Gemeindeamt, Reichsbahn) and directory ditto "
    "marks (“Dieselbe”). Frequency filtering drops surnames seen only once "
    "as OCR noise. Two gzipped JSON artifacts result (4.3 MB total, ~1 s to "
    "load):", BODY))
S.append(Paragraph(
    "• <b>lexicon.json.gz</b> — flat frequency dictionaries: 83,450 surnames, "
    "4,256 given names, 18,869 occupations, 17,748 towns, plus the "
    "town→state map.", BULLET))
S.append(Paragraph(
    "• <b>kg_town_surnames.json.gz</b> — the knowledge graph: for each of "
    "12,299 towns, the surnames attested there with counts. This is what "
    "makes retrieval geographic (1-hop = Pernegg's own surnames, 2-hop = all "
    "of Steiermark) instead of one giant national list.", BULLET))

S.append(Paragraph("Appendix&nbsp;&nbsp;Where parish Pernegg sits in the RTB 1942", H1))
S.append(Paragraph(
    "The parish (Pernegg an der Mur, diocese Graz-Seckau) appears in the RTB "
    "as the region heading <b>“Pernegg (Steiermark)”</b>, in volume 4 "
    "(<font face='Courier'>RTB_4_Karbow-Quittainen.pdf</font>), page 1275, "
    "column 2, sections 08–09. In the digitised corpus it is in "
    "<font face='Courier'>master_RTB_4_Karbow-Quittainen_clean.xlsx</font> "
    "(batches page_1275_b16/b17). A different Pernegg — “Pernegg b. "
    "Hötzelndorf (Niederdonau)”, in the Waldviertel — also exists and must "
    "not be confused with the parish. Identification is confirmed by entry "
    "addresses such as “Kirchdorf bei Pernegg a. Mur” and the hamlet names "
    "Kirchdorf and Traföß, which also occur in the 1895 Sterberegister.",
    BODY))

doc.build(S)
print(f"wrote {OUT}")
