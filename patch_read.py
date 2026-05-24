content = open('confluence_engine.py', encoding='utf-8').read()
idx = content.find('"raw_checks": {')
if idx < 0:
    idx = content.find("'raw_checks': {")
print(repr(content[idx:idx+800]))
