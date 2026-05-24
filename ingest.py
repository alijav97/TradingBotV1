"""
ingest.py - Materials Ingestion Pipeline for TradingBotV1

This script fetches trading content from URLs and PDFs, uses Claude AI to
extract structured trading rules, and saves them to data/rules.json.
"""

import os
import json
import base64
from pathlib import Path
import requests
import urllib3
import anthropic
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Optional OCR dependencies — gracefully handled if Tesseract is not installed
try:
    from PIL import Image
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Optional Word document support
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

import time
from _progress import Spinner, ProgressBar, _bar, _fmt_time, OK, FAIL, SKIP, WARN

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError("ANTHROPIC_API_KEY not found. Check your .env file.")

# Suppress SSL verification warnings emitted by the verify=False fallback
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialise the Anthropic client once at module level
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── System prompt used for every extraction call ──────────────────────────────
EXTRACTION_SYSTEM_PROMPT = (
    "You are a trading rule extractor. Read the provided trading material and extract "
    "specific actionable trading rules with PRECISE numerical conditions wherever possible.\n\n"
    "For each rule return a JSON object with:\n"
    "- pattern_name: exact name of the pattern or strategy\n"
    "- entry_condition: precise trigger with numbers where possible "
    "(e.g. 'RSI below 30', 'price within 0.5% of EMA 200', 'close above 20-period high')\n"
    "- candle_pattern: hammer/engulfing/doji/pin_bar/none\n"
    "- trend_requirement: bullish/bearish/any\n"
    "- indicator_conditions: list of specific indicator states needed as strings\n"
    "- timeframe: M15/H1/H4/D1\n"
    "- direction: long/short/both\n"
    "- stop_loss_pips: number of pips, ATR multiplier (e.g. '1.5 ATR'), or descriptive level\n"
    "- take_profit_pips: number of pips or RR expression (e.g. '2R', '40 pips')\n"
    "- min_risk_reward: minimum acceptable RR ratio as a number (e.g. 1.5)\n"
    "- confidence_score: 1-10 based on how clear and well-defined the rule is\n"
    "- source: filename or URL the rule came from\n"
    "- key_levels: relevant support/resistance/EMA levels if mentioned\n\n"
    "Return ONLY a valid JSON array, no other text. "
    "Never truncate — always close the array with ]. "
    "Be as numerically specific as possible."
)

RULES_FILE = os.path.join("data", "rules.json")
RESOURCES_DIR = "resources"


# ── 1. fetch_url ──────────────────────────────────────────────────────────────
def fetch_url(url: str) -> str:
    """
    Fetch a webpage and return its clean plain text.

    Args:
        url: The full URL to retrieve.

    Returns:
        Extracted visible text from the page, or an empty string on failure.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        try:
            response = requests.get(url, headers=headers, timeout=20)
        except requests.exceptions.SSLError:
            # Some servers have TLS quirks — retry without certificate verification
            print(f"  [WARN] SSL error, retrying without cert verification...")
            response = requests.get(url, headers=headers, timeout=20, verify=False)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script, style, and nav noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # Collapse excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    except requests.RequestException as exc:
        print(f"  [ERROR] Could not fetch URL '{url}': {exc}")
        return ""


# ── 2. fetch_pdf ──────────────────────────────────────────────────────────────
def fetch_pdf(filepath: str) -> str:
    """
    Extract all text from a local PDF file.

    Args:
        filepath: Absolute or relative path to the PDF file.

    Returns:
        Combined text from every page, or an empty string on failure.
    """
    try:
        reader = PdfReader(filepath)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        return "\n".join(pages_text)

    except Exception as exc:
        print(f"  [ERROR] Could not read PDF '{filepath}': {exc}")
        return ""


# ── 3. download_pdf ──────────────────────────────────────────────────────────
def download_pdf(pdf_url: str) -> str:
    """
    Download a PDF from a URL into the data/ folder and return the local path.

    Args:
        pdf_url: The direct URL to the PDF file.

    Returns:
        Local file path of the downloaded PDF, or an empty string on failure.
    """
    os.makedirs("data", exist_ok=True)

    # Derive a safe filename from the URL
    filename = pdf_url.split("/")[-1].split("?")[0]  # strip query strings
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    # Sanitise the filename
    filename = "".join(c if c.isalnum() or c in ".-_" else "_" for c in filename)
    local_path = os.path.join("data", filename)

    # Skip download if file already exists
    if os.path.exists(local_path):
        print(f"  [CACHE] PDF already downloaded: {local_path}")
        return local_path

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*",
        "Referer": "/".join(pdf_url.split("/")[:3]) + "/",
    }
    try:
        print(f"  Downloading PDF from: {pdf_url}")
        try:
            response = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
        except requests.exceptions.SSLError:
            print(f"  [WARN] SSL error, retrying without cert verification...")
            response = requests.get(pdf_url, headers=headers, timeout=60, stream=True, verify=False)
        response.raise_for_status()

        with open(local_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)

        print(f"  [OK] Saved to: {local_path}")
        return local_path

    except requests.RequestException as exc:
        print(f"  [ERROR] Could not download PDF '{pdf_url}': {exc}")
        return ""


# ── 4. extract_trading_rules ──────────────────────────────────────────────────
def extract_trading_rules(text: str, source_name: str) -> list[dict]:
    """
    Send raw text to Claude and extract structured trading rules.

    Args:
        text:        The raw content to analyse.
        source_name: A label identifying where the content came from.

    Returns:
        A list of trading-rule dicts, or an empty list on failure.
    """
    if not text.strip():
        print(f"  [SKIP] No text content to process for '{source_name}'.")
        return []

    # Truncate to stay well within the model's context window (~150 k tokens)
    max_chars = 60_000
    if len(text) > max_chars:
        print(f"  [INFO] Content truncated to {max_chars} chars for '{source_name}'.")
        text = text[:max_chars]

    user_message = (
        f"Source: {source_name}\n\n"
        f"Content:\n{text}"
    )

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8096,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_json = message.content[0].text.strip()

        # Strip markdown code fences if Claude wraps the response in them
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
            raw_json = raw_json.strip()

        rules = json.loads(raw_json)

        # Ensure every rule carries the source label
        for rule in rules:
            rule.setdefault("source", source_name)

        return rules

    except json.JSONDecodeError as exc:
        print(f"  [ERROR] Claude returned invalid JSON for '{source_name}': {exc}")
        return []
    except anthropic.APIError as exc:
        print(f"  [ERROR] Anthropic API error for '{source_name}': {exc}")
        return []
    except Exception as exc:
        print(f"  [ERROR] Unexpected error during extraction for '{source_name}': {exc}")
        return []


# ── 5. save_rules ─────────────────────────────────────────────────────────────
def save_rules(rules: list[dict], source_name: str) -> None:
    """
    Append new trading rules to data/rules.json.

    Creates the file (and the data/ directory) if they do not yet exist.

    Args:
        rules:       List of rule dicts to persist.
        source_name: Label used in the printed summary.
    """
    if not rules:
        print(f"  [INFO] No rules to save for '{source_name}'.")
        return

    os.makedirs("data", exist_ok=True)

    # Load existing rules if the file already exists
    existing_rules: list[dict] = []
    if os.path.exists(RULES_FILE):
        try:
            with open(RULES_FILE, "r", encoding="utf-8") as fh:
                existing_rules = json.load(fh)
        except (json.JSONDecodeError, IOError):
            print(f"  [WARN] Could not read existing '{RULES_FILE}'. Starting fresh.")
            existing_rules = []

    combined = existing_rules + rules

    with open(RULES_FILE, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, ensure_ascii=False)

    print(f"  [SAVED] {len(rules)} new rule(s) from '{source_name}'. "
          f"Total rules in file: {len(combined)}.")


# ── 6. extract_rules_from_image ──────────────────────────────────────────────
def extract_rules_from_image(image_path: str, source_name: str) -> list[dict]:
    """
    Extract trading rules from an image file.

    Tries pytesseract OCR first (requires Tesseract installed on Windows:
    https://github.com/UB-Mannheim/tesseract/wiki).
    Falls back to sending the image directly to Claude vision if OCR fails
    or if Tesseract is not installed.

    Args:
        image_path:  Path to the .png / .jpg / .jpeg file.
        source_name: Label for the source (used in rules + logs).

    Returns:
        List of trading-rule dicts.
    """
    # ── Attempt 1: pytesseract OCR ────────────────────────────────────────────
    ocr_text = ""
    if PYTESSERACT_AVAILABLE:
        try:
            img = Image.open(image_path)
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                print("  [OCR] Text extracted via pytesseract.")
            else:
                print("  [OCR] pytesseract returned empty text — falling back to Claude vision.")
        except Exception as exc:
            print(f"  [WARN] pytesseract failed ({exc}) — falling back to Claude vision.")
            ocr_text = ""
    else:
        print(
            "  [INFO] pytesseract not available (Tesseract not installed on Windows). "
            "See https://github.com/UB-Mannheim/tesseract/wiki — "
            "falling back to Claude vision."
        )

    # If OCR produced text, use the normal text-based extraction pipeline
    if ocr_text:
        return extract_trading_rules(ocr_text, source_name)

    # ── Attempt 2: Claude vision fallback ────────────────────────────────────
    print("  [VISION] Sending image to Claude for visual rule extraction...")
    try:
        with open(image_path, "rb") as fh:
            image_data = base64.standard_b64encode(fh.read()).decode("utf-8")

        # Determine media type from extension
        ext = Path(image_path).suffix.lower()
        media_type_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        media_type = media_type_map.get(ext, "image/png")

        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8096,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Source: {source_name}\n\n"
                                "Examine this trading chart or document image. "
                                "Extract all trading rules, patterns, or strategies you can identify."
                            ),
                        },
                    ],
                }
            ],
        )

        raw_json = message.content[0].text.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
            raw_json = raw_json.strip()

        rules = json.loads(raw_json)
        for rule in rules:
            rule.setdefault("source", source_name)
        return rules

    except json.JSONDecodeError as exc:
        print(f"  [ERROR] Claude vision returned invalid JSON for '{source_name}': {exc}")
        return []
    except Exception as exc:
        print(f"  [ERROR] Claude vision failed for '{source_name}': {exc}")
        return []


# ── 7. process_local_resources ────────────────────────────────────────────────
def process_local_resources(rules_so_far: int = 0) -> tuple[int, int, int]:
    """
    Scan the resources/ folder and extract trading rules from every supported file.

    Returns:
        (total_rules_extracted, files_processed, files_skipped)
    """
    os.makedirs(RESOURCES_DIR, exist_ok=True)

    unrelated_file = Path(RESOURCES_DIR) / "unrelated_files.txt"
    skipped_names: set[str] = set()
    if unrelated_file.exists():
        for line in unrelated_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                filename_part = line.split("→")[0].strip()
                if filename_part:
                    skipped_names.add(filename_part)

    supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".docx"}
    all_files = sorted(
        f for f in Path(RESOURCES_DIR).iterdir()
        if f.is_file()
        and f.suffix.lower() in supported_extensions
        and f.name != "unrelated_files.txt"
    )

    # Count by type for header
    pdfs   = [f for f in all_files if f.suffix.lower() == ".pdf"]
    images = [f for f in all_files if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    txts   = [f for f in all_files if f.suffix.lower() in {".txt", ".docx"}]

    print(f"  Scanning resources/ folder...")
    print(f"  Found: {len(pdfs)} PDFs | {len(images)} images | {len(txts)} text/docx files")

    if not all_files:
        print("  [INFO] No supported files found in resources/ folder.")
        return 0, 0, 0

    to_process = [f for f in all_files if f.name not in skipped_names]
    to_skip    = [f for f in all_files if f.name in skipped_names]

    # Print skip notices first
    for f in to_skip:
        print(f"\n  {SKIP} [SKIPPED] {f.name}")
        print(f"     Reason: listed in unrelated_files.txt")

    # Counters per type
    pdf_idx = img_idx = txt_idx = 0
    total_rules  = 0
    processed    = 0
    start_all    = time.time()

    for file_path in to_process:
        ext         = file_path.suffix.lower()
        source_name = f"local:{file_path.name}"
        file_start  = time.time()

        if ext == ".pdf":
            pdf_idx += 1
            type_label = f"PDF {pdf_idx}/{len(pdfs)}"
        elif ext in {".png", ".jpg", ".jpeg"}:
            img_idx += 1
            type_label = f"IMAGE {img_idx}/{len(images)}"
        else:
            txt_idx += 1
            type_label = f"TEXT {txt_idx}/{len(txts)}"

        print(f"\n  [{type_label}] {file_path.name}")
        content = ""
        rules   = []

        try:
            if ext == ".pdf":
                # Show page-reading progress
                try:
                    reader      = PdfReader(str(file_path))
                    total_pages = len(reader.pages)
                    pages_text  = []
                    pbar        = ProgressBar(total_pages, indent=4)
                    for p_idx, page in enumerate(reader.pages, 1):
                        txt = page.extract_text()
                        if txt:
                            pages_text.append(txt)
                        pbar.update(f"page {p_idx}")
                    pbar.finish("pages read")
                    content = "\n".join(pages_text)
                except Exception as exc:
                    print(f"    {WARN} PDF read error: {exc}")
                    content = ""

                if content:
                    with Spinner("Sending to Claude API...", indent=4) as sp:
                        rules = extract_trading_rules(content, source_name)

            elif ext in {".png", ".jpg", ".jpeg"}:
                if PYTESSERACT_AVAILABLE:
                    with Spinner("Running OCR...", indent=4) as sp:
                        try:
                            img      = Image.open(str(file_path))
                            ocr_text = pytesseract.image_to_string(img).strip()
                        except Exception:
                            ocr_text = ""
                    if ocr_text:
                        with Spinner("Sending to Claude API...", indent=4) as sp:
                            rules = extract_trading_rules(ocr_text, source_name)
                    else:
                        with Spinner("Sending to Claude vision...", indent=4) as sp:
                            rules = extract_rules_from_image(str(file_path), source_name)
                else:
                    with Spinner("Sending to Claude vision...", indent=4) as sp:
                        rules = extract_rules_from_image(str(file_path), source_name)

            elif ext == ".txt":
                content = file_path.read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    with Spinner("Sending to Claude API...", indent=4) as sp:
                        rules = extract_trading_rules(content, source_name)

            elif ext == ".docx":
                if DOCX_AVAILABLE:
                    doc     = DocxDocument(str(file_path))
                    content = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                    if content.strip():
                        with Spinner("Sending to Claude API...", indent=4) as sp:
                            rules = extract_trading_rules(content, source_name)
                else:
                    print(f"    {SKIP} python-docx not installed — skipping .docx")

            save_rules(rules, source_name)
            file_elapsed = time.time() - file_start
            total_rules += len(rules)
            icon = OK if rules else WARN
            print(f"    {icon} Extracted {len(rules)} rules  ({_fmt_time(file_elapsed)})")
            print(f"    Rules extracted so far: {rules_so_far + total_rules}")
            processed += 1

        except Exception as exc:
            print(f"    {FAIL} Failed: {exc}  — skipping")

    return total_rules, processed, len(to_skip)


# ── 8. Main ingestion loop ────────────────────────────────────────────────────
def run_full_ingest() -> int:
    """Run the full ingestion pipeline. Returns total rules extracted."""
    return _main_ingest()


def _main_ingest() -> int:
    """Internal entry point shared by __main__ and setup.py."""

    # ── Regular webpages ──────────────────────────────────────────────────────
    URLS_TO_FETCH = [
        # xs.com — candlestick reference pages
        "https://www.xs.com/en/blog/candlestick-patterns-cheat-sheet/",
        "https://www.xs.com/en/blog/candlestick-patterns-types/",
        # learnpriceaction.com — price action strategies
        "https://learnpriceaction.com/gold-trading-strategies-pdf/",
        "https://learnpriceaction.com/candlestick-patterns/",
        # StockCharts School — technical indicators (replaces blocked Investopedia)
        "https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi",
        "https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd",
        "https://school.stockcharts.com/doku.php?id=chart_analysis:fibonacci_retrace",
        "https://school.stockcharts.com/doku.php?id=technical_indicators:average_true_range_atr",
        "https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands",
        "https://school.stockcharts.com/doku.php?id=technical_indicators:relative_vigor_index",
        # StockCharts School — chart patterns & concepts (replaces blocked BabyPips)
        "https://school.stockcharts.com/doku.php?id=chart_analysis:support_and_resistance",
        "https://school.stockcharts.com/doku.php?id=technical_indicators:moving_averages",
        "https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo",
        "https://school.stockcharts.com/doku.php?id=chart_analysis:introduction_to_candlesticks",
        "https://school.stockcharts.com/doku.php?id=chart_analysis:chart_patterns",
    ]

    # ── PDF files to download then read ───────────────────────────────────────
    PDF_URLS_TO_DOWNLOAD_AND_READ = [
        # mql5.com — support & resistance strategies (confirmed working)
        "https://c.mql5.com/forextsd/forum/172/trade_forex_with_support_and_resistance_strategies.pdf",
        # goldtradingexperts.com — gold trading guide
        "http://www.goldtradingexperts.com/uploads/8/9/8/4/8984623/a_guide_to_successful_gold_trading.pdf",
        # CFTC (US regulator) — publicly available trading fundamentals guide
        "https://www.cftc.gov/sites/default/files/idc/groups/public/@customerprotection/documents/file/forex_info_customer.pdf",
        # SEC (US regulator) — technical analysis investor guide
        "https://www.sec.gov/investor/pubs/tenthingstoconsider.pdf",
        # World Bank open-access finance paper
        "https://documents1.worldbank.org/curated/en/099510103062226723/pdf/P17848904a98050780a4c20040ee3fc6671.pdf",
    ]

    # Counters for final summary
    local_files_processed  = 0
    urls_processed         = 0
    pdfs_processed         = 0
    total_rules_so_far     = 0
    ingest_start           = time.time()

    print("\n  INGESTING RESOURCES")
    print("  " + "═" * 52)

    print(f"  Sources: {len(URLS_TO_FETCH)} web URLs | {len(PDF_URLS_TO_DOWNLOAD_AND_READ)} remote PDFs | resources/ folder")

    ingest_start = time.time()

    # ═══════════════════════════════════════════════════════════
    # STEP 1 — Local resources/ folder
    # ═══════════════════════════════════════════════════════════
    print("\n  ── Local Resources ──────────────────────────────────")
    rules_from_local, local_processed, local_skipped = process_local_resources(0)
    total_rules_so_far += rules_from_local

    # ═══════════════════════════════════════════════════════════
    # STEP 2 — Web URLs
    # ═══════════════════════════════════════════════════════════
    print("\n  ── Web URLs ─────────────────────────────────────────")
    url_bar = ProgressBar(len(URLS_TO_FETCH), indent=2)
    for idx, url in enumerate(URLS_TO_FETCH, start=1):
        short = url.replace("https://", "").replace("http://", "").rstrip("/")[:50]
        url_bar.update(short)
        print()  # newline after inline bar
        print(f"  [URL {idx}/{len(URLS_TO_FETCH)}] {short}")

        with Spinner("Fetching page...", indent=4) as sp:
            content = fetch_url(url)

        if not content:
            print(f"    {SKIP} Empty content — skipped")
            continue

        with Spinner("Sending to Claude API...", indent=4) as sp:
            rules = extract_trading_rules(content, source_name=url)

        save_rules(rules, source_name=url)
        urls_processed     += 1
        total_rules_so_far += len(rules)
        icon = OK if rules else WARN
        print(f"    {icon} Extracted {len(rules)} rules  |  Running total: {total_rules_so_far}")

    url_bar.finish(f"URLs processed: {urls_processed}/{len(URLS_TO_FETCH)}")

    # ═══════════════════════════════════════════════════════════
    # STEP 3 — Remote PDF downloads
    # ═══════════════════════════════════════════════════════════
    print("\n  ── Remote PDFs ──────────────────────────────────────")
    pdf_bar = ProgressBar(len(PDF_URLS_TO_DOWNLOAD_AND_READ), indent=2)
    for idx, pdf_url in enumerate(PDF_URLS_TO_DOWNLOAD_AND_READ, start=1):
        label = pdf_url.split("/")[-1].split("?")[0] or pdf_url
        pdf_bar.update(label[:45])
        print()
        print(f"  [PDF {idx}/{len(PDF_URLS_TO_DOWNLOAD_AND_READ)}] {label}")

        with Spinner("Downloading PDF...", indent=4) as sp:
            local_path = download_pdf(pdf_url)

        if not local_path:
            print(f"    {FAIL} Download failed — skipped")
            continue

        with Spinner("Reading PDF pages...", indent=4) as sp:
            content = fetch_pdf(local_path)

        if not content:
            print(f"    {SKIP} No text extracted — skipped")
            continue

        with Spinner("Sending to Claude API...", indent=4) as sp:
            rules = extract_trading_rules(content, source_name=pdf_url)

        save_rules(rules, source_name=pdf_url)
        pdfs_processed     += 1
        total_rules_so_far += len(rules)
        icon = OK if rules else WARN
        print(f"    {icon} Extracted {len(rules)} rules  |  Running total: {total_rules_so_far}")

    pdf_bar.finish(f"PDFs processed: {pdfs_processed}/{len(PDF_URLS_TO_DOWNLOAD_AND_READ)}")

    # ═══════════════════════════════════════════════════════════
    # Final summary
    # ═══════════════════════════════════════════════════════════
    final_count = 0
    rules_before = 0
    if os.path.exists(RULES_FILE):
        try:
            with open(RULES_FILE, "r", encoding="utf-8") as fh:
                final_count = len(json.load(fh))
        except Exception:
            final_count = total_rules_so_far

    total_elapsed = time.time() - ingest_start

    print("\n  " + "═" * 52)
    print("  INGESTION COMPLETE")
    print(f"  Files processed  : {local_processed} local | {urls_processed} URLs | {pdfs_processed} PDFs")
    print(f"  Files skipped    : {local_skipped}")
    print(f"  Total rules extracted : {total_rules_so_far}")
    print(f"  rules.json updated {OK}  →  {final_count} total rules")
    print(f"  Total time       : {_fmt_time(total_elapsed)}")
    print("  " + "═" * 52)

    return final_count


if __name__ == "__main__":
    _main_ingest()
