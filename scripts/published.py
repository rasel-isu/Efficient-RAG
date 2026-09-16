"""Numbers transcribed from published papers, kept in one place.

Both report generators compare against these, and a figure that disagrees
between two reports is worse than no comparison at all, so they are defined
once here rather than copied into each script.

Every entry is (EM, F1) exactly as printed in the cited table. Dataset keys
match this repo's loader names.
"""

# --- CompAct (Yoon et al., EMNLP 2024; arXiv 2407.09014) Table 2 -------------
# Reader LLaMA3-8B, Contriever-MSMARCO over DPR Wikipedia 2018, top-30, no
# reranking. This is the closest published setting to this project's grid: same
# retriever, same corpus, same depth. The reader differs (8B vs our 3B).
COMPACT_SOURCE = ("CompAct (EMNLP 2024) Table 2 — reader LLaMA3-8B, "
                  "Contriever top-30, no reranking")
COMPACT = {
    "hotpotqa": {"Raw Document": (29.4, 40.3), "RECOMP (extractive)": (29.7, 39.9),
                 "LongLLMLingua": (25.6, 35.3), "AutoCompressors": (18.4, 28.4),
                 "CompAct": (35.5, 46.9), "Oracle": (39.9, 51.2)},
    "musique":  {"Raw Document": (6.5, 15.6), "RECOMP (extractive)": (6.7, 15.7),
                 "LongLLMLingua": (4.8, 13.5), "AutoCompressors": (3.9, 11.9),
                 "CompAct": (8.7, 18.1), "Oracle": (14.2, 23.6)},
    "2wiki":    {"Raw Document": (25.4, 31.2), "RECOMP (extractive)": (29.9, 34.9),
                 "LongLLMLingua": (27.9, 32.9), "AutoCompressors": (19.0, 24.5),
                 "CompAct": (31.0, 37.1), "Oracle": (37.4, 43.2)},
    "nq_open":  {"Raw Document": (39.0, 51.3), "RECOMP (extractive)": (34.6, 45.1),
                 "LongLLMLingua": (27.7, 40.6), "AutoCompressors": (17.3, 31.8),
                 "CompAct": (38.4, 50.0)},
    "triviaqa": {"Raw Document": (68.9, 77.1), "RECOMP (extractive)": (67.6, 74.1),
                 "LongLLMLingua": (64.0, 70.8), "AutoCompressors": (55.3, 64.3),
                 "CompAct": (65.4, 74.9)},
}
# Compression rate column of the same table, per dataset (not one global number).
COMPACT_COMP = {
    "hotpotqa": {"Raw Document": 1.0, "RECOMP (extractive)": 34.3,
                 "LongLLMLingua": 3.4, "AutoCompressors": 35.4,
                 "CompAct": 47.6, "Oracle": 10.8},
    "musique":  {"Raw Document": 1.0, "RECOMP (extractive)": 32.7,
                 "LongLLMLingua": 3.4, "AutoCompressors": 34.7,
                 "CompAct": 37.2, "Oracle": 10.3},
    "2wiki":    {"Raw Document": 1.0, "RECOMP (extractive)": 35.9,
                 "LongLLMLingua": 3.6, "AutoCompressors": 36.2,
                 "CompAct": 51.2, "Oracle": 11.0},
    "nq_open":  {"Raw Document": 1.0, "RECOMP (extractive)": 32.7,
                 "LongLLMLingua": 3.5, "AutoCompressors": 34.4,
                 "CompAct": 48.5},
    "triviaqa": {"Raw Document": 1.0, "RECOMP (extractive)": 39.2,
                 "LongLLMLingua": 3.3, "AutoCompressors": 34.5,
                 "CompAct": 49.4},
}
# Row order for the comparison table; the uncompressed row comes first because
# every other row's retention is computed against it.
COMPACT_ORDER = ["Raw Document", "AutoCompressors", "LongLLMLingua",
                 "RECOMP (extractive)", "CompAct", "Oracle"]
COMPACT_UNCOMPRESSED = "Raw Document"
COMPACT_READER = "LLaMA3-8B"

# --- RECOMP (Xu, Shi & Choi, ICLR 2024; arXiv 2310.04408) Table 2 ------------
# Reader Flan-UL2 20B, Contriever top-5. A different reader AND depth, so this
# is context, not a like-for-like comparison. Its "T5 (off-the-shelf)" row is
# the closest published analogue of this project's method: an untrained
# summarisation model applied to retrieved passages.
RECOMP_SOURCE = ("RECOMP (ICLR 2024) Table 2 — reader Flan-UL2 20B, "
                 "Contriever top-5")
RECOMP_PAPER = {
    "nq_open":  {"Top 5 documents": (39.39, 48.28), "Top 1 document": (33.07, 41.45),
                 "no retrieval": (21.99, 29.38), "T5 (off-the-shelf)": (25.90, 34.63),
                 "RECOMP abstractive": (37.04, 45.47), "RECOMP extractive": (36.57, 44.22)},
    "triviaqa": {"Top 5 documents": (62.37, 70.09), "Top 1 document": (57.84, 64.94),
                 "no retrieval": (49.33, 54.85), "T5 (off-the-shelf)": (55.18, 62.34),
                 "RECOMP abstractive": (58.68, 66.34), "RECOMP extractive": (58.99, 65.26)},
    "hotpotqa": {"Top 5 documents": (32.80, 43.90), "Top 1 document": (28.80, 40.58),
                 "no retrieval": (17.80, 26.10), "T5 (off-the-shelf)": (23.20, 33.19),
                 "RECOMP abstractive": (28.20, 37.91), "RECOMP extractive": (30.40, 40.14)},
}
RECOMP_UNCOMPRESSED = "Top 5 documents"
RECOMP_READER = "Flan-UL2 20B"
