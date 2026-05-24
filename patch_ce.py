import sys

content = open('confluence_engine.py', encoding='utf-8').read()

INDICATOR_BLOCK = r"""
    # -- FACTOR 12+: 14 Technical Indicators (combined weight: max +/-3.0) -------
    _ind_data: dict = {}
    if _IND_OK:
        try:
            inds = _get_indicators(df)
            _ind_data = inds
            ind_score = 0.0
            ind_lines_local: list[str] = []

            def _aligned(bias: str, dir_: str) -> bool:
                b = bias.lower()
                return ('bull' in b) if dir_ == 'long' else ('bear' in b)

            def _opposed_ind(bias: str, dir_: str) -> bool:
                b = bias.lower()
                return ('bear' in b) if dir_ == 'long' else ('bull' in b)

            al_i = inds.get('alligator', {})
            if al_i.get('sleeping'):
                ind_score -= 0.5
                ind_lines_local.append('\u26a0 Alligator SLEEPING  -0.5')
            elif _aligned(al_i.get('bias', 'neutral'), direction):
                w = 1.0 if 'EATING' in al_i.get('state', '') else 0.5
                ind_score += w
                ind_lines_local.append(f'\u2713 Alligator {al_i.get("state","")} +{w}')

            adx_i = inds.get('adx', {})
            if adx_i.get('trending') and _aligned(adx_i.get('bias', ''), direction):
                w = 0.8 if adx_i.get('strength') == 'STRONG' else 0.4
                ind_score += w
                ind_lines_local.append(f'\u2713 ADX {adx_i.get("adx",0):.0f} {adx_i.get("strength","")}  +{w}')
            elif not adx_i.get('trending'):
                ind_score -= 0.3
                ind_lines_local.append('~ ADX weak  -0.3')

            mac_i = inds.get('macd', {})
            if mac_i.get('bullish_cross') and direction == 'long':
                ind_score += 1.0
                ind_lines_local.append('\u2713 MACD bullish crossover  +1.0')
            elif mac_i.get('bearish_cross') and direction == 'short':
                ind_score += 1.0
                ind_lines_local.append('\u2713 MACD bearish crossover  +1.0')
            elif _aligned(mac_i.get('bias', ''), direction):
                ind_score += 0.5
                ind_lines_local.append('~ MACD aligned  +0.5')

            st_i = inds.get('stoch_rsi', {})
            if st_i.get('bullish_cross') and st_i.get('oversold') and direction == 'long':
                ind_score += 1.0
                ind_lines_local.append('\u2713 StochRSI oversold bullish cross  +1.0')
            elif st_i.get('bearish_cross') and st_i.get('overbought') and direction == 'short':
                ind_score += 1.0
                ind_lines_local.append('\u2713 StochRSI overbought bearish cross  +1.0')
            elif _aligned(st_i.get('bias', ''), direction):
                ind_score += 0.4
                ind_lines_local.append('~ StochRSI aligned  +0.4')

            ich_i = inds.get('ichimoku', {})
            if _aligned(ich_i.get('bias', ''), direction):
                w = 1.0 if 'strongly' in ich_i.get('bias', '') else 0.6
                ind_score += w
                ind_lines_local.append(f'\u2713 Ichimoku cloud aligned  +{w}')
            elif ich_i.get('in_cloud'):
                ind_lines_local.append('~ Ichimoku in cloud  0.0')

            vw_i = inds.get('vwap', {})
            if _aligned(vw_i.get('bias', ''), direction):
                ind_score += 0.5
                ind_lines_local.append('\u2713 VWAP aligned  +0.5')

            sq_i = inds.get('squeeze', {})
            if sq_i.get('squeeze_off') and _aligned(sq_i.get('bias', ''), direction):
                ind_score += 0.8
                ind_lines_local.append('\u2713 BB Squeeze fired  +0.8')
            elif sq_i.get('squeeze_on'):
                ind_score += 0.3
                ind_lines_local.append('~ Squeeze building  +0.3')

            su_i = inds.get('supertrend', {})
            if _aligned(su_i.get('bias', ''), direction):
                flipped_note = " (flipped)" if su_i.get('just_flipped') else ""
                w = 1.0 if su_i.get('just_flipped') else 0.7
                ind_score += w
                ind_lines_local.append(f'\u2713 Supertrend aligned{flipped_note}  +{w}')
            elif _opposed_ind(su_i.get('bias', ''), direction):
                ind_score -= 0.5
                ind_lines_local.append(f'\u2717 Supertrend opposes {direction}  -0.5')

            km_i = inds.get('kama', {})
            if _aligned(km_i.get('bias', ''), direction) and km_i.get('trending'):
                ind_score += 0.5
                ind_lines_local.append('\u2713 KAMA adaptive trend aligned  +0.5')

            kz_i = inds.get('killzones', {})
            if kz_i.get('in_killzone'):
                w = 1.0 if kz_i.get('high_quality') else 0.5
                zones = ', '.join(kz_i.get('active_zones', []))
                ind_score += w
                ind_lines_local.append(f'\u2713 ICT Kill Zone: {zones}  +{w}')

            wy_i = inds.get('wyckoff', {})
            if _aligned(wy_i.get('bias', ''), direction):
                ind_score += 0.8
                ind_lines_local.append(f'\u2713 Wyckoff {wy_i.get("phase","")}  +0.8')

            rr_i = inds.get('real_rate', {})
            if rr_i.get('available') and _aligned(rr_i.get('bias', ''), direction):
                w = 0.8 if 'strongly' in rr_i.get('bias', '') else 0.4
                ind_score += w
                ind_lines_local.append(f'\u2713 Real rate {rr_i.get("real_rate",0):.1f}%  +{w}')

            mc_i = inds.get('market_cipher', {})
            if mc_i.get('bullish_cross') and mc_i.get('oversold') and direction == 'long':
                ind_score += 1.0
                ind_lines_local.append('\u2713 Market Cipher oversold bullish cross  +1.0')
            elif mc_i.get('bearish_cross') and mc_i.get('overbought') and direction == 'short':
                ind_score += 1.0
                ind_lines_local.append('\u2713 Market Cipher overbought bearish cross  +1.0')
            elif _aligned(mc_i.get('bias', ''), direction):
                ind_score += 0.5
                ind_lines_local.append('~ Market Cipher aligned  +0.5')

            ob_i = inds.get('obv', {})
            if ob_i.get('divergence') == 'bullish_divergence' and direction == 'long':
                ind_score += 0.8
                ind_lines_local.append('\u2713 OBV bullish divergence  +0.8')
            elif ob_i.get('divergence') == 'bearish_divergence' and direction == 'short':
                ind_score += 0.8
                ind_lines_local.append('\u2713 OBV bearish divergence  +0.8')
            elif _aligned(ob_i.get('bias', ''), direction):
                ind_score += 0.4
                ind_lines_local.append('~ OBV aligned  +0.4')

            # Normalise: indicators contribute max +/-3.0
            ind_score_norm = min(max(ind_score, -1.5), 3.0)
            weighted_raw  += ind_score_norm
            for line in ind_lines_local:
                detail_lines.append(line)
            check_weights_earned['Indicators'] = round(ind_score_norm, 2)

        except Exception as _ind_e:
            detail_lines.append(f'~ Indicators unavailable: {str(_ind_e)[:50]}')
            _ind_data = {}

"""

FINAL_SCORE_LINE = (
    '    # \u2500\u2500 FINAL SCORE '
    + '\u2500' * 58
    + '\n    confidence  = min(10.0,'
)

if FINAL_SCORE_LINE in content:
    content = content.replace(
        FINAL_SCORE_LINE,
        INDICATOR_BLOCK + '    # \u2500\u2500 FINAL SCORE ' + '\u2500' * 58 + '\n    confidence  = min(10.0,',
        1
    )
    print('Scoring block: OK')
else:
    # find whatever is there
    idx = content.find('    # \u2500\u2500 FINAL SCORE')
    if idx >= 0:
        snippet = content[idx:idx+200]
        print('Found at idx', idx, 'but line mismatch, repr:')
        print(repr(snippet))
    else:
        print('FINAL SCORE line not found at all')
    sys.exit(1)

# Also add "indicators": _ind_data to raw_checks return dict
COT_LINE  = '"cot":            _cot_result,\n        },'
IND_ENTRY = '"cot":            _cot_result,\n            "indicators":     _ind_data,\n        },'
if COT_LINE in content:
    content = content.replace(COT_LINE, IND_ENTRY, 1)
    print('raw_checks indicators entry: OK')
else:
    print('WARNING: cot line not found for raw_checks patch')

open('confluence_engine.py', 'w', encoding='utf-8').write(content)
print('File written.')
