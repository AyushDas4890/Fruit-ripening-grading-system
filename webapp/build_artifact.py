"""Bundle the presentation into ONE self-contained HTML file for sharing as a Claude artifact.

    python webapp/build_artifact.py   -> webapp/artifact/ripeness_presentation.html

Inlines style.css + presentation.css + presentation.js, embeds every results image as a
JPEG data URI and the results JSON in place of the /api/results call (no server needed).
Re-run after src/evaluate.py to refresh the numbers.
"""
import base64
import io
import json
import os
import re

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S = os.path.join(HERE, "static")
R = os.path.join(ROOT, "results")


def data_uri(path, max_w=1800):
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=84, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


html = open(os.path.join(S, "presentation.html"), encoding="utf-8").read()
css = open(os.path.join(S, "style.css"), encoding="utf-8").read() + "\n" + \
    open(os.path.join(S, "presentation.css"), encoding="utf-8").read()
js = open(os.path.join(S, "presentation.js"), encoding="utf-8").read()
results = json.load(open(os.path.join(R, "web_results.json")))

# keep only <head> fonts + <body> content; the artifact host adds its own skeleton
head = re.search(r"<head>(.*?)</head>", html, re.S).group(1)
body = re.search(r"<body[^>]*>(.*?)</body>", html, re.S).group(1)
head = re.sub(r'<link rel="stylesheet" href="/static/[^"]+">\s*', "", head)
head = re.sub(r"<meta[^>]*>\s*", "", head)
head = head.replace("<title>Ripeness Grading – Presentation</title>", "<title>Ripe or Not Slides</title>")

# images -> data URIs
for name in set(re.findall(r'/results/([\w.]+\.png)', body)):
    body = body.replace(f"/results/{name}", data_uri(os.path.join(R, name)))

# no demo server here: logo and back links become plain text, CTA explains how to run it
body = body.replace('<a class="logo" href="/">', '<span class="logo">').replace(
    'Ripe<span>/</span>or<span>/</span>Not</a>', 'Ripe<span>/</span>or<span>/</span>Not</span>', 1)
body = re.sub(r'<a class="back hand" href="/">.*?</a>', '<span class="back hand">use ← → keys</span>', body)
body = body.replace('<p class="hand big-hand a-up">questions? try the live demo →</p>',
                    '<p class="hand big-hand a-up">questions? the live demo runs locally:</p>')
body = re.sub(r'<a class="cta note-sun a-pop" href="/"><span class="tape"></span>Open the demo</a>',
              '<div class="cta note-sun a-pop"><span class="tape"></span><span class="mono" style="font-size:20px">python webapp/server.py</span></div>',
              body)
body = re.sub(r'<script src="/static/presentation.js"></script>', "", body)

# results come inline instead of from the server
js = js.replace('results(await (await fetch("/api/results")).json());',
                "results(window.RESULTS);")

out = f"""<meta name="claude:viewport-fit-sides" content="cover">
{head.strip()}
<style>
{css}
/* artifact: single deliberate light "paper" look; side gutter clears phone system bars */
.slide {{ padding-left: max(72px, env(safe-area-inset-left, 0px)); padding-right: max(72px, env(safe-area-inset-right, 0px)); }}
@media (max-width: 1000px) {{ .slide {{ padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }} }}
.deck-nav {{ padding-left: max(32px, env(safe-area-inset-left, 0px)); padding-right: max(32px, env(safe-area-inset-right, 0px)); top: env(safe-area-inset-top, 0px); }}
.deck-body, .deck {{ height: 100%; }}
html {{ height: 100%; }}
</style>
{body.strip()}
<script>window.RESULTS = {json.dumps(results)};</script>
<script>
{js}
</script>
"""
os.makedirs(os.path.join(HERE, "artifact"), exist_ok=True)
p = os.path.join(HERE, "artifact", "ripeness_presentation.html")
open(p, "w", encoding="utf-8").write(out)
print(p, f"{len(out) / 1e6:.2f} MB")
