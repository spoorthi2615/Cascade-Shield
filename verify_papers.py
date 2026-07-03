import re
import urllib.request
import urllib.parse
import json
import difflib

file_path = r"d:\projects\cascade sheild\docs\related_work.md"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
new_lines = []
flagged = []

title_pattern = re.compile(r'- \*\*\[.*?\] "(.*?)".*\*\*')

for line in lines:
    match = title_pattern.search(line)
    if match and "(DOI:" not in line and "(arXiv:" not in line:
        title = match.group(1)
        query = urllib.parse.quote(title)
        url = f"https://api.crossref.org/works?query.title={query}&select=title,DOI&rows=3"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Cascade Shield/1.0 (mailto:test@example.com)'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                items = data.get('message', {}).get('items', [])
                best_match = None
                best_ratio = 0
                
                for item in items:
                    item_title = item.get('title', [''])[0]
                    ratio = difflib.SequenceMatcher(None, title.lower(), item_title.lower()).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = item
                
                if best_ratio > 0.85 and best_match:
                    doi = best_match['DOI']
                    line = line + f" (DOI: {doi})"
                    print(f"FOUND: {title[:50]}... -> {doi}")
                else:
                    flagged.append(title)
                    print(f"FLAGGED: {title}")
        except Exception as e:
            flagged.append(title)
            print(f"ERROR on {title}: {e}")
            
    new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))

print("\n--- FLAGGED PAPERS ---")
for p in flagged:
    print(p)
