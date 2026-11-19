"""Build the PowerPoint version of the presentation.

    python deploy/build_pptx.py <folder with hd1.png ... hd14.png>

Each slide is a full-bleed 2880x1620 capture of the web presentation (?slide=N&clean),
so the deck keeps the notebook template exactly. The speaker notes on each slide are the
matching section of docs/scripts/slides_script.html.
Output: docs/Ripe_or_Not_presentation.pptx
"""
import html
import os
import re
import sys

from pptx import Presentation
from pptx.util import Emu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shots = sys.argv[1]
script = open(os.path.join(ROOT, "docs", "scripts", "slides_script.html"), encoding="utf-8").read()

notes = {}
for num, body in re.findall(r"<h2>Slide (\d+) · .*?</h2>(.*?)</div></div>", script, flags=re.S):
    paras = re.findall(r"<p>(.*?)</p>", body, flags=re.S)
    notes[int(num)] = "\n\n".join(html.unescape(re.sub(r"<[^>]+>", "", t)).strip() for t in paras)

TITLES = ["Title", "Agenda", "Ripeness is a ladder, not a list", "Real photos, two fruits, four stages",
          "From detection boxes to crops", "Same fruit must never sit in train and test",
          "Ask three yes/no questions", "The CORAL head", "A fair, controlled comparison",
          "Check the plumbing first", "What the numbers say", "How wrong are the wrong answers?",
          "Findings & limits", "Thank you"]

prs = Presentation()
prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)   # 13.333 x 7.5 in, 16:9
blank = prs.slide_layouts[6]
for i in range(1, 15):
    s = prs.slides.add_slide(blank)
    pic = s.shapes.add_picture(os.path.join(shots, f"hd{i}.png"), 0, 0, prs.slide_width, prs.slide_height)
    pic.name = f"Slide {i} – {TITLES[i - 1]}"
    pic._element.nvPicPr.cNvPr.set("descr", TITLES[i - 1])   # alt text for screen readers
    s.notes_slide.notes_text_frame.text = notes.get(i, "")
out = os.path.join(ROOT, "docs", "Ripe_or_Not_presentation.pptx")
prs.save(out)
print(out, f"{os.path.getsize(out) / 1e6:.1f} MB", "| slides with notes:", sum(1 for i in range(1, 15) if notes.get(i)))
