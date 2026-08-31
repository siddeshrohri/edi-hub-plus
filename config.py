# ── EDI Hub+ Pipeline Configuration ──────────────────────────────────────────
# All constants live here. Change settings in this file only — nothing else
# should have magic strings or numbers hardcoded.

# ── Target ────────────────────────────────────────────────────────────────────
RESOURCE_CENTRE_URL = "https://edihubplusstg.wpenginepowered.com/edi-resources-2/"

# ── HTTP Client ───────────────────────────────────────────────────────────────
USER_AGENT = (
    "EDI-Hub-Research-Bot/1.0 "
    "(University of Leeds research; edihub@leeds.ac.uk)"
)
REQUEST_TIMEOUT_SEC = 20
RATE_LIMIT_SEC      = 1.5   # pause between requests — be polite to servers

# ── Output ────────────────────────────────────────────────────────────────────
STAGE1_OUTPUT_FILE = "stage1_resources.json"
LOG_FILE           = "edi_hub_pipeline.log"
