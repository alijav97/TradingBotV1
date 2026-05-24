content = open('confluence_engine.py', encoding='utf-8').read()

# 1. Insert ML import guard right after _IND_OK = False block
OLD_IMPORT = (
    "    _IND_OK = False\n"
    "    def _get_indicators(df): return {}  # type: ignore[misc]\n"
    "\n"
    "# \u2500\u2500 Constants"
)

NEW_IMPORT = (
    "    _IND_OK = False\n"
    "    def _get_indicators(df): return {}  # type: ignore[misc]\n"
    "\n"
    "try:\n"
    "    from ml_engine import get_ml_confidence_adjustment as _get_ml_adj\n"
    "    _ML_OK = True\n"
    "except Exception:\n"
    "    _ML_OK = False\n"
    "    def _get_ml_adj(*a, **kw): return {'adjustment': 0.0, 'available': False}  # type: ignore[misc]\n"
    "\n"
    "# \u2500\u2500 Constants"
)

if OLD_IMPORT in content:
    content = content.replace(OLD_IMPORT, NEW_IMPORT, 1)
    print("ML import: OK")
else:
    print("ML import anchor NOT FOUND, searching...")
    idx = content.find("_IND_OK = False")
    print(repr(content[idx-5:idx+200]))

# 2. Insert ML adjustment block after indicators except block, before FINAL SCORE
OLD_BLOCK = (
    "            detail_lines.append(f'~ Indicators unavailable: {str(_ind_e)[:50]}')\n"
    "            _ind_data = {}\n"
    "\n"
    "    # \u2500\u2500 FINAL SCORE"
)

ML_ADJ_BLOCK = '''\
    # -- ML Confidence Adjustment (learns from paper trades) -------------------
    if _ML_OK:
        try:
            from datetime import datetime, timezone, timedelta
            _gst_zone  = timezone(timedelta(hours=4))
            _hour_uae  = datetime.now(_gst_zone).hour

            _sess_raw  = raw_checks.get("session") or {}
            _sess_name = (
                _sess_raw.get("session") if isinstance(_sess_raw, dict)
                else str(_sess_raw)
            )
            _reg_raw   = raw_checks.get("regime") or {}
            _reg_name  = (
                _reg_raw.get("regime") if isinstance(_reg_raw, dict)
                else str(_reg_raw)
            )

            ml_adj = _get_ml_adj(
                session    = str(_sess_name or "Unknown"),
                regime     = str(_reg_name  or "Unknown"),
                strategy   = str(symbol),
                confidence = float(weighted_raw),
                hour_uae   = int(_hour_uae),
            )

            if ml_adj.get("available") and ml_adj.get("adjustment", 0.0) != 0.0:
                weighted_raw += ml_adj["adjustment"]
                adj_sign = "+" if ml_adj["adjustment"] > 0 else ""
                detail_lines.append(
                    f"\U0001f916 ML adj ({ml_adj.get('model_size',0)} trades): "
                    f"{adj_sign}{ml_adj['adjustment']:.1f}"
                )
                for reason in ml_adj.get("reasons", [])[:2]:
                    detail_lines.append(f"   \u2192 {reason}")
                raw_checks["ml_adjustment"] = ml_adj
        except Exception:
            pass

'''

NEW_BLOCK = (
    "            detail_lines.append(f'~ Indicators unavailable: {str(_ind_e)[:50]}')\n"
    "            _ind_data = {}\n"
    "\n"
    + ML_ADJ_BLOCK
    + "    # \u2500\u2500 FINAL SCORE"
)

if OLD_BLOCK in content:
    content = content.replace(OLD_BLOCK, NEW_BLOCK, 1)
    print("ML adjustment block: OK")
else:
    print("ML adjustment anchor NOT FOUND, searching...")
    idx = content.find("_ind_data = {}")
    if idx > 0:
        print(repr(content[idx-10:idx+300]))

open('confluence_engine.py', 'w', encoding='utf-8').write(content)
print("File written.")
