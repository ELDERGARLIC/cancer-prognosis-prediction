"""Task 2: Parse citations across thesis chapters; dedupe; build Vancouver-style
bibliography.md + references.bib.

Inputs:
  - results/thesis/{abstract,introduction,methods,results,discussion,conclusion}.md
  - results/methodological_notes.md (supplementary; cited works appear here too)

Outputs:
  - results/thesis/bibliography.md   (numbered Vancouver-style human-readable)
  - results/thesis/references.bib    (BibTeX for pandoc integration)
  - results/thesis/citations_audit.md  (where each citation appears + completeness)

Citation regex covers the four patterns used in the thesis:
  (Author, YYYY)
  (Author et al., YYYY)
  (Author and Author, YYYY)
  Author et al. (YYYY)
  Author (YYYY)
  Author and Author (YYYY)
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT / "results" / "thesis"

CHAPTER_FILES = [
    ("Abstract", THESIS / "abstract.md"),
    ("Introduction", THESIS / "introduction.md"),
    ("Methods", THESIS / "methods.md"),
    ("Results", THESIS / "results.md"),
    ("Discussion", THESIS / "discussion.md"),
    ("Conclusion", THESIS / "conclusion.md"),
]

OUT_BIB_MD = THESIS / "bibliography.md"
OUT_BIB_BIB = THESIS / "references.bib"
OUT_AUDIT = THESIS / "citations_audit.md"

# ---- Best-effort bibliography metadata ----
# Confident: well-known foundational papers + identified within-thesis citations.
# Flagged with `_incomplete=True` where DOI/year/journal needs supervisor verification.
REFERENCES: list[dict] = [
    # Foundational survival analysis
    {
        "key": "cox1972",
        "authors": "Cox DR",
        "year": 1972,
        "title": "Regression models and life-tables",
        "journal": "Journal of the Royal Statistical Society Series B",
        "volume": "34", "issue": "2", "pages": "187-220",
        "doi": "10.1111/j.2517-6161.1972.tb00899.x",
        "type": "article",
        "_short": "Cox 1972",
    },
    {
        "key": "harrell1982",
        "authors": "Harrell FE Jr, Califf RM, Pryor DB, Lee KL, Rosati RA",
        "year": 1982,
        "title": "Evaluating the yield of medical tests",
        "journal": "JAMA",
        "volume": "247", "issue": "18", "pages": "2543-2546",
        "doi": "10.1001/jama.1982.03320430047030",
        "type": "article",
        "_short": "Harrell et al. 1982",
    },
    # Cancer / BRCA biology
    {
        "key": "sorlie2001",
        "authors": "Sørlie T, Perou CM, Tibshirani R, Aas T, Geisler S, Johnsen H, et al.",
        "year": 2001,
        "title": ("Gene expression patterns of breast carcinomas distinguish "
                  "tumor subclasses with clinical implications"),
        "journal": "Proceedings of the National Academy of Sciences",
        "volume": "98", "issue": "19", "pages": "10869-10874",
        "doi": "10.1073/pnas.191367098",
        "type": "article",
        "_short": "Sørlie et al. 2001",
    },
    {
        "key": "howlader2014",
        "authors": ("Howlader N, Altekruse SF, Li CI, Chen VW, Clarke CA, "
                    "Ries LAG, Cronin KA"),
        "year": 2014,
        "title": ("US incidence of breast cancer subtypes defined by joint "
                  "hormone receptor and HER2 status"),
        "journal": "Journal of the National Cancer Institute",
        "volume": "106", "issue": "5", "pages": "dju055",
        "doi": "10.1093/jnci/dju055",
        "type": "article",
        "_short": "Howlader et al. 2014",
    },
    # Statistical learning
    {
        "key": "tibshirani1996",
        "authors": "Tibshirani R",
        "year": 1996,
        "title": "Regression shrinkage and selection via the lasso",
        "journal": "Journal of the Royal Statistical Society Series B",
        "volume": "58", "issue": "1", "pages": "267-288",
        "doi": "10.1111/j.2517-6161.1996.tb02080.x",
        "type": "article",
        "_short": "Tibshirani 1996",
    },
    {
        "key": "kingma2015",
        "authors": "Kingma DP, Ba J",
        "year": 2015,
        "title": "Adam: A method for stochastic optimization",
        "booktitle": ("Proceedings of the 3rd International Conference on "
                      "Learning Representations (ICLR)"),
        "pages": "",
        "arxiv": "1412.6980",
        "type": "inproceedings",
        "_short": "Kingma and Ba 2015",
    },
    # Survival deep learning
    {
        "key": "katzman2018",
        "authors": ("Katzman JL, Shaham U, Cloninger A, Bates J, Jiang T, "
                    "Kluger Y"),
        "year": 2018,
        "title": ("DeepSurv: personalized treatment recommender system using "
                  "a Cox proportional hazards deep neural network"),
        "journal": "BMC Medical Research Methodology",
        "volume": "18", "issue": "1", "pages": "24",
        "doi": "10.1186/s12874-018-0482-1",
        "type": "article",
        "_short": "Katzman et al. 2018",
    },
    # Graph neural networks
    {
        "key": "hamilton2017",
        "authors": "Hamilton WL, Ying R, Leskovec J",
        "year": 2017,
        "title": "Inductive representation learning on large graphs",
        "booktitle": ("Advances in Neural Information Processing Systems "
                      "(NeurIPS)"),
        "pages": "1024-1034",
        "arxiv": "1706.02216",
        "type": "inproceedings",
        "_short": "Hamilton et al. 2017",
    },
    {
        "key": "ling2022",
        "authors": "Ling Y, [METADATA INCOMPLETE — verify full author list]",
        "year": 2022,
        "title": "[METADATA INCOMPLETE — verify exact title; cited for oversmoothing in deeper GNNs on small graphs]",
        "journal": "[METADATA INCOMPLETE — verify journal/venue]",
        "volume": "", "issue": "", "pages": "", "doi": "",
        "type": "article",
        "_short": "Ling et al. 2022",
        "_incomplete": True,
    },
    # Patient-as-graph and inductive paradigm
    {
        "key": "vaida2025",
        "authors": "Vaida M, [METADATA INCOMPLETE — verify full author list]",
        "year": 2025,
        "title": "[METADATA INCOMPLETE — verify exact title; cited for patient-as-graph paradigm in cancer prognosis]",
        "journal": "[METADATA INCOMPLETE — verify journal/venue]",
        "volume": "", "issue": "", "pages": "", "doi": "",
        "type": "article",
        "_short": "Vaida et al. 2025",
        "_incomplete": True,
    },
    {
        "key": "madanipour2024",
        "authors": "Madanipour H, [METADATA INCOMPLETE — verify full author list]",
        "year": 2024,
        "title": "[METADATA INCOMPLETE — verify exact title; cited for inductive GraphSAGE on cancer prognosis]",
        "journal": "[METADATA INCOMPLETE — verify journal/venue]",
        "volume": "", "issue": "", "pages": "", "doi": "",
        "type": "article",
        "_short": "Madanipour et al. 2024",
        "_incomplete": True,
    },
    # Reviews on external validation
    {
        "key": "liang2025",
        "authors": "Liang [METADATA INCOMPLETE — verify full author list]",
        "year": 2025,
        "title": "[METADATA INCOMPLETE — verify exact title; cited for review calling for external validation in cancer AI]",
        "journal": "[METADATA INCOMPLETE — verify journal/venue]",
        "volume": "", "issue": "", "pages": "", "doi": "",
        "type": "article",
        "_short": "Liang 2025",
        "_incomplete": True,
    },
    {
        "key": "vavekanand2026",
        "authors": "Vavekanand R, Liang [METADATA INCOMPLETE — verify full author list]",
        "year": 2026,
        "title": "[METADATA INCOMPLETE — verify exact title; cited for review on cancer-AI evaluation practice]",
        "journal": "[METADATA INCOMPLETE — verify journal/venue]",
        "volume": "", "issue": "", "pages": "", "doi": "",
        "type": "article",
        "_short": "Vavekanand and Liang 2026",
        "_incomplete": True,
    },
    # Cancer GNN / clinical fusion
    {
        "key": "gao2021",
        "authors": "Gao J, [METADATA INCOMPLETE — verify full author list]",
        "year": 2021,
        "title": "[METADATA INCOMPLETE — verify exact title; cited for clinical-fusion ablation showing gene-only vs gene+clinical lift on TCGA prognosis]",
        "journal": "[METADATA INCOMPLETE — verify journal/venue]",
        "volume": "", "issue": "", "pages": "", "doi": "",
        "type": "article",
        "_short": "Gao et al. 2021",
        "_incomplete": True,
    },
    {
        "key": "choudhry2025",
        "authors": "Choudhry [METADATA INCOMPLETE — verify full author list]",
        "year": 2025,
        "title": "[METADATA INCOMPLETE — verify exact title; cited for pathway-attention pooling design precedent in oncology GNNs]",
        "journal": "[METADATA INCOMPLETE — verify journal/venue]",
        "volume": "", "issue": "", "pages": "", "doi": "",
        "type": "article",
        "_short": "Choudhry et al. 2025",
        "_incomplete": True,
    },
    # METABRIC original
    {
        "key": "curtis2012",
        "authors": ("Curtis C, Shah SP, Chin SF, Turashvili G, Rueda OM, "
                    "Dunning MJ, et al."),
        "year": 2012,
        "title": ("The genomic and transcriptomic architecture of 2,000 "
                  "breast tumours reveals novel subgroups"),
        "journal": "Nature",
        "volume": "486", "issue": "7403", "pages": "346-352",
        "doi": "10.1038/nature10983",
        "type": "article",
        "_short": "Curtis et al. 2012",
    },
    # LLM gene priors
    {
        "key": "lee2020biobert",
        "authors": ("Lee J, Yoon W, Kim S, Kim D, Kim S, So CH, Kang J"),
        "year": 2020,
        "title": ("BioBERT: a pre-trained biomedical language representation "
                  "model for biomedical text mining"),
        "journal": "Bioinformatics",
        "volume": "36", "issue": "4", "pages": "1234-1240",
        "doi": "10.1093/bioinformatics/btz682",
        "type": "article",
        "_short": "Lee et al. 2020",
    },
    {
        "key": "chen2023genept",
        "authors": "Chen Y, Zou J",
        "year": 2023,
        "title": ("GenePT: a simple but effective foundation model for genes "
                  "and cells built from ChatGPT"),
        "journal": "bioRxiv",
        "volume": "", "issue": "", "pages": "",
        "doi": "10.1101/2023.10.16.562533",
        "type": "article",
        "_short": "Chen and Zou 2023",
    },
    # Software citations from Methods §5.2 and §3
    {
        "key": "davidsonpilon2019",
        "authors": "Davidson-Pilon C",
        "year": 2019,
        "title": ("lifelines: survival analysis in Python"),
        "journal": "Journal of Open Source Software",
        "volume": "4", "issue": "40", "pages": "1317",
        "doi": "10.21105/joss.01317",
        "type": "article",
        "_short": "Davidson-Pilon 2019",
    },
    {
        "key": "polsterl2020",
        "authors": "Pölsterl S",
        "year": 2020,
        "title": ("scikit-survival: a library for time-to-event analysis "
                  "built on top of scikit-learn"),
        "journal": "Journal of Machine Learning Research",
        "volume": "21", "issue": "212", "pages": "1-6",
        "type": "article",
        "_short": "Pölsterl 2020",
    },
    # STRING (Methods §1.2)
    {
        "key": "szklarczyk2023",
        "authors": ("Szklarczyk D, Kirsch R, Koutrouli M, Nastou K, Mehryary F, "
                    "Hachilif R, et al."),
        "year": 2023,
        "title": ("The STRING database in 2023: protein-protein association "
                  "networks and functional enrichment analyses for any "
                  "sequenced genome of interest"),
        "journal": "Nucleic Acids Research",
        "volume": "51", "issue": "D1", "pages": "D638-D646",
        "doi": "10.1093/nar/gkac1000",
        "type": "article",
        "_short": "Szklarczyk et al. 2023",
    },
]


# ---- Helpers ----

def parse_citations(text: str) -> list[str]:
    """Find all citation short-forms in a chapter text. Returns Author Year list."""
    out = []
    # (Author et al., YYYY) and (Author et al. YYYY)
    out += re.findall(r"\(([A-Z][a-zA-ZÀ-ſ-]+ et al\.?,?\s+\d{4})\)", text)
    # (Author and Author, YYYY) and (Author and Author YYYY)
    out += re.findall(r"\(([A-Z][a-zA-ZÀ-ſ-]+ and [A-Z][a-zA-ZÀ-ſ-]+,?\s+\d{4})\)", text)
    # (Author, YYYY) -- single author, careful not to overmatch
    out += re.findall(r"\(([A-Z][a-zA-ZÀ-ſ-]+,\s+\d{4})\)", text)
    # Inline forms: Author et al. (YYYY); Author (YYYY)
    out += re.findall(r"([A-Z][a-zA-ZÀ-ſ-]+ et al\.?\s+\((\d{4})\))", text)
    out = [c[0] if isinstance(c, tuple) else c for c in out]
    out += re.findall(r"([A-Z][a-zA-ZÀ-ſ-]+ and [A-Z][a-zA-ZÀ-ſ-]+\s+\((\d{4})\))", text)
    out = [c[0] if isinstance(c, tuple) else c for c in out]
    # Multi-cite: (Author1, YYYY; Author2, YYYY)
    multi = re.findall(r"\(([^)]+;[^)]+)\)", text)
    for m in multi:
        for piece in m.split(";"):
            piece = piece.strip()
            single = re.match(r"([A-Z][a-zA-ZÀ-ſ-]+(?: et al\.?)?(?:\s+and\s+[A-Z][a-zA-ZÀ-ſ-]+)?,?\s+\d{4})", piece)
            if single:
                out.append(single.group(1))
    return out


def normalize_citation(raw: str) -> str:
    """Reduce 'Author et al. (2020)' / 'Author et al., 2020' to 'Author et al. 2020'."""
    s = raw.strip().rstrip(")").lstrip("(")
    s = re.sub(r"\(\s*", " ", s)
    s = re.sub(r"\s*\)\s*$", "", s)
    s = re.sub(r",\s*", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def find_match(short: str, refs: list[dict]) -> dict | None:
    norm = normalize_citation(short)
    # Match by author surname + year
    m = re.match(r"([A-Za-zÀ-ſ-]+)(?:\s+et al\.?)?(?:\s+and\s+[A-Za-zÀ-ſ-]+)?\s+(\d{4})", norm)
    if not m:
        return None
    surname, year = m.group(1), int(m.group(2))
    surname_low = surname.lower()
    for r in refs:
        ra = r["authors"].split(",")[0].split()[0].lower()
        if ra == surname_low and r["year"] == year:
            return r
        # Also handle "Vavekanand and Liang 2026" — match Vavekanand
        if r["_short"].split()[0].lower() == surname_low and r["year"] == year:
            return r
    return None


def vancouver_format(ref: dict, idx: int) -> str:
    """Format one bibliography entry in Vancouver style."""
    incomplete = ref.get("_incomplete", False)
    parts = [f"**{idx}.** {ref['authors']}. {ref['title']}."]
    if ref["type"] == "article":
        if ref.get("journal"):
            parts.append(f"*{ref['journal']}*. {ref['year']}")
        else:
            parts.append(f"{ref['year']}")
        if ref.get("volume"):
            vol_str = f";{ref['volume']}"
            if ref.get("issue"):
                vol_str += f"({ref['issue']})"
            if ref.get("pages"):
                vol_str += f":{ref['pages']}"
            parts[-1] += vol_str
        parts[-1] += "."
        if ref.get("doi"):
            parts.append(f"doi:{ref['doi']}.")
    elif ref["type"] == "inproceedings":
        parts.append(f"In: *{ref['booktitle']}*. {ref['year']}")
        if ref.get("pages"):
            parts[-1] += f":{ref['pages']}"
        parts[-1] += "."
        if ref.get("arxiv"):
            parts.append(f"arXiv:{ref['arxiv']}.")
    out = " ".join(parts)
    if incomplete:
        out = f"⚠️ **METADATA INCOMPLETE — VERIFY:** {out}"
    return out


def bibtex_format(ref: dict) -> str:
    """Format one BibTeX entry."""
    incomplete = ref.get("_incomplete", False)
    type_map = {"article": "article", "inproceedings": "inproceedings"}
    btype = type_map.get(ref["type"], "misc")
    lines = [f"@{btype}{{{ref['key']},"]
    fields = [
        ("author", ref["authors"]),
        ("title", ref["title"]),
        ("year", ref["year"]),
    ]
    if ref["type"] == "article":
        fields += [
            ("journal", ref.get("journal", "")),
            ("volume", ref.get("volume", "")),
            ("number", ref.get("issue", "")),
            ("pages", ref.get("pages", "")),
            ("doi", ref.get("doi", "")),
        ]
    else:
        fields += [
            ("booktitle", ref.get("booktitle", "")),
            ("pages", ref.get("pages", "")),
            ("eprint", ref.get("arxiv", "")),
        ]
    for k, v in fields:
        if v == "" or v is None:
            continue
        v_str = str(v)
        # Clean placeholders
        if "[METADATA INCOMPLETE" in v_str:
            v_str = v_str  # keep as-is for the user to see + fix
        lines.append(f"  {k:9s} = {{{v_str}}},")
    lines.append("}")
    if incomplete:
        lines.insert(1, "  % ⚠️ METADATA INCOMPLETE — VERIFY before submission")
    return "\n".join(lines)


# ---- Main ----

def main():
    print("==> Parsing citations across chapters")
    cite_locations: dict[str, list[str]] = defaultdict(list)
    all_chapters_text = ""
    for label, path in CHAPTER_FILES:
        text = path.read_text()
        all_chapters_text += text
        cites = parse_citations(text)
        for c in cites:
            cite_locations[normalize_citation(c)].append(label)

    # Dedupe
    unique = sorted(cite_locations.keys())
    print(f"    found {len(unique)} unique citation tokens across chapters")

    # Match to bibliography
    matched: list[tuple[str, dict]] = []
    unmatched: list[str] = []
    for tok in unique:
        ref = find_match(tok, REFERENCES)
        if ref is None:
            unmatched.append(tok)
        else:
            matched.append((tok, ref))

    # Build deduped reference list ordered by first appearance
    used_keys = set()
    ordered_refs: list[dict] = []
    for _, r in matched:
        if r["key"] not in used_keys:
            used_keys.add(r["key"])
            ordered_refs.append(r)

    # Also include REFERENCES entries that are in our metadata table but didn't
    # get parsed (e.g., because the regex missed an inline form). Append at end.
    for r in REFERENCES:
        if r["key"] not in used_keys:
            ordered_refs.append(r)
            used_keys.add(r["key"])

    print(f"    matched {len(matched)} citations to {len(ordered_refs)} unique sources")
    if unmatched:
        print(f"    UNMATCHED tokens (not in REFERENCES table): {unmatched}")

    # Write bibliography.md (Vancouver)
    md_lines = ["# Bibliography\n"]
    md_lines.append(
        "Vancouver style. Numbered in order of citation grouping. Entries "
        "marked ⚠️ have incomplete metadata that requires supervisor "
        "verification before submission.\n"
    )
    for i, r in enumerate(ordered_refs, start=1):
        md_lines.append(vancouver_format(r, i))
        md_lines.append("")
    OUT_BIB_MD.write_text("\n".join(md_lines))
    print(f"==> wrote {OUT_BIB_MD}")

    # Write references.bib
    bib_lines = ["% References for thesis. Auto-generated; verify [METADATA INCOMPLETE] entries.\n"]
    for r in ordered_refs:
        bib_lines.append(bibtex_format(r))
        bib_lines.append("")
    OUT_BIB_BIB.write_text("\n".join(bib_lines))
    print(f"==> wrote {OUT_BIB_BIB}")

    # Citations audit: where each cite appears + completeness
    audit_lines = [
        "# Citations Audit\n",
        f"Total in-text citation occurrences (token-level dedup): {len(unique)}",
        f"Unique sources after metadata match: {len(ordered_refs)}",
        f"Sources flagged METADATA INCOMPLETE: "
        f"{sum(1 for r in ordered_refs if r.get('_incomplete'))}",
        "",
        "## Citations by source (with chapter locations)",
        "",
    ]
    # For each ref, list which chapters cite it
    audit_lines.append("| # | Short form | Chapters cited in | Metadata status |")
    audit_lines.append("|---:|---|---|---|")
    for i, r in enumerate(ordered_refs, start=1):
        # Find any matching token
        chapters_for_this = []
        for tok, locs in cite_locations.items():
            if find_match(tok, [r]) is not None:
                chapters_for_this.extend(locs)
        chapters_uniq = sorted(set(chapters_for_this))
        chapters_str = ", ".join(chapters_uniq) if chapters_uniq else "(not parsed — check inline forms)"
        status = "⚠️ INCOMPLETE" if r.get("_incomplete") else "✓ complete"
        audit_lines.append(f"| {i} | {r['_short']} | {chapters_str} | {status} |")
    audit_lines += [
        "",
        "## Unmatched citation tokens (parsed but not in metadata table)",
        "",
    ]
    if unmatched:
        for u in unmatched:
            audit_lines.append(f"- `{u}` — appears in: {', '.join(set(cite_locations[u]))}")
    else:
        audit_lines.append("- (none)")

    audit_lines += [
        "",
        "## Pre-submission action list",
        "",
        "Resolve these before sending the bibliography to the supervisor:",
        "",
    ]
    for r in ordered_refs:
        if r.get("_incomplete"):
            audit_lines.append(
                f"- **{r['_short']}**: verify exact title, full author list, journal, "
                f"volume, issue, pages, DOI."
            )
    if not any(r.get("_incomplete") for r in ordered_refs):
        audit_lines.append("- (no incomplete entries)")

    OUT_AUDIT.write_text("\n".join(audit_lines))
    print(f"==> wrote {OUT_AUDIT}")

    print()
    print("==> Done.")


if __name__ == "__main__":
    main()
