import sys
errors = []

# Test 1: All new modules import correctly
try:
    from macro_scorer import MacroScorer
    ms = MacroScorer()
    print("✅ macro_scorer imported")
except Exception as e:
    errors.append(f"❌ macro_scorer: {e}")

try:
    from instrument_data import (get_ohlcv,
        get_market_context, get_instrument_summary)
    print("✅ instrument_data imported")
except Exception as e:
    errors.append(f"❌ instrument_data: {e}")

try:
    from sector_rotation import SectorRotation
    sr = SectorRotation()
    print("✅ sector_rotation imported")
except Exception as e:
    errors.append(f"❌ sector_rotation: {e}")

try:
    from open_interest import OpenInterestAnalyzer
    oi = OpenInterestAnalyzer()
    print("✅ open_interest imported")
except Exception as e:
    errors.append(f"❌ open_interest: {e}")

try:
    from instrument_confluence import InstrumentConfluence
    ic = InstrumentConfluence("XAUUSD")
    print("✅ instrument_confluence imported")
except Exception as e:
    errors.append(f"❌ instrument_confluence: {e}")

try:
    from ml_engine import MLEngine
    ml = MLEngine("XAUUSD")
    print("✅ ml_engine (XAUUSD) imported")
    ml2 = MLEngine("NAS100")
    print("✅ ml_engine (NAS100) imported")
except Exception as e:
    errors.append(f"❌ ml_engine: {e}")

try:
    from paper_trader import (open_paper_trade,
        get_open_trades, get_paper_summary)
    print("✅ paper_trader imported")
except Exception as e:
    errors.append(f"❌ paper_trader: {e}")

try:
    from settings import INSTRUMENT_SETTINGS
    assert len(INSTRUMENT_SETTINGS) == 6
    print(f"✅ INSTRUMENT_SETTINGS: "
          f"{list(INSTRUMENT_SETTINGS.keys())}")
except Exception as e:
    errors.append(f"❌ settings: {e}")

try:
    from mt5_sync import (get_price_for_instrument,
        INSTRUMENT_SYMBOLS)
    print(f"✅ mt5_sync instrument-aware: "
          f"{list(INSTRUMENT_SYMBOLS.keys())}")
except Exception as e:
    errors.append(f"❌ mt5_sync: {e}")

# Test 2: MacroScorer logic
try:
    ms = MacroScorer()
    usd = ms.score_currency("USD")
    assert "score" in usd
    assert "grade" in usd
    gbpusd = ms.score_pair("GBPUSD")
    assert "bias" in gbpusd
    assert "score_diff" in gbpusd
    eurusd = ms.score_pair("EURUSD")
    assert "bias" in eurusd
    print(f"✅ MacroScorer: USD={usd['score']} "
          f"GBPUSD={gbpusd['bias']} "
          f"EURUSD={eurusd['bias']}")
except Exception as e:
    errors.append(f"❌ MacroScorer logic: {e}")

# Test 3: InstrumentConfluence per instrument
try:
    for instr in ["XAUUSD","NAS100","US30",
                  "GBPUSD","EURUSD","WTI"]:
        ic = InstrumentConfluence(instr)
        rules = ic.check_hard_rules({})
        extra = ic.get_extra_confluence_score()
        assert "blocked" in rules
        assert "extra_score" in extra
    print("✅ InstrumentConfluence: all 6 instruments OK")
except Exception as e:
    errors.append(
        f"❌ InstrumentConfluence logic: {e}")

# Test 4: Paper trader per-instrument files
try:
    from paper_trader import _get_trades_file
    assert _get_trades_file("XAUUSD") == \
           "data/paper_trades.json"
    assert _get_trades_file("NAS100") == \
           "data/paper_trades_NAS100.json"
    assert _get_trades_file("GBPUSD") == \
           "data/paper_trades_GBPUSD.json"
    print("✅ paper_trader: per-instrument files OK")
except Exception as e:
    errors.append(
        f"❌ paper_trader file routing: {e}")

# Test 5: ML engine per-instrument files
try:
    from ml_engine import _get_ml_files
    xau = _get_ml_files("XAUUSD")
    nas = _get_ml_files("NAS100")
    assert xau["model"] != nas["model"]
    assert xau["paper"] != nas["paper"]
    print(f"✅ ml_engine file routing: "
          f"XAUUSD={xau['model']} "
          f"NAS100={nas['model']}")
except Exception as e:
    errors.append(
        f"❌ ml_engine file routing: {e}")

# Test 6: Backtest instrument awareness
try:
    from backtest import YF_BACKTEST_TICKERS
    assert "NAS100" in YF_BACKTEST_TICKERS
    assert "GBPUSD" in YF_BACKTEST_TICKERS
    assert "WTI" in YF_BACKTEST_TICKERS
    print(f"✅ backtest tickers: "
          f"{list(YF_BACKTEST_TICKERS.keys())}")
except Exception as e:
    errors.append(f"❌ backtest tickers: {e}")

# Final report
print("\n" + "="*50)
print(f"TESTS PASSED: {9 - len(errors)}/9")
if errors:
    print("\nFAILURES:")
    for err in errors:
        print(err)
else:
    print("✅ ALL TESTS PASSED — Bot ready for "
          "multi-instrument trading")
print("="*50)
