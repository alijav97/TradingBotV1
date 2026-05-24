content = open('confluence_engine.py', encoding='utf-8').read()
print('ml import:', 'from ml_engine' in content)
print('IND_OK:', '_IND_OK' in content)
print('factor12:', '# -- FACTOR 12' in content)
idx = content.find('_IND_OK = False')
if idx > 0:
    print('IND area:', repr(content[idx-10:idx+100]))
idx2 = content.find("check_weights_earned['Indicators']")
if idx2 > 0:
    print('After inds:', repr(content[idx2:idx2+250]))
