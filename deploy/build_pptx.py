"""Build the PowerPoint version of the presentation (13 slides).

    python deploy/build_pptx.py <folder with hd1.png ... hd14.png>

Each slide is a full-bleed 2880x1620 capture of the web presentation (?slide=N&clean),
so the deck keeps the notebook template exactly. The web deck's slide 12 (error distances +
confusion matrices) is left out of the PowerPoint. Speaker notes come from
docs/scripts/presenter_script.md (the presenter's own script, numbered for the 13 slides).
Output: docs/Ripe_or_Not_presentation.pptx
"""
import os
import re
import sys

from pptx import Presentation
from pptx.util import Emu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 13
shots = sys.argv[1]
md = open(os.path.join(ROOT, "docs", "scripts", "presenter_script.md"), encoding="utf-8").read()

notes = {}
for num, body in re.findall(r"^### Slide (\d+) — .*?\n(.*?)(?=^---|^### Slide|\Z)", md, flags=re.S | re.M):
    notes[int(num)] = "\n\n".join(p.strip() for p in body.strip().split("\n\n") if p.strip())
WEB = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]   # web slide shown on each PowerPoint slide
TITLES = ["Introduction", "Agenda", "Ripeness is a ladder, not a list", "Real photos, two fruits, four stages",
          "From detection boxes to crops", "Same fruit must never sit in train and test",
          "Ask three yes/no questions", "The CORAL head", "A fair, controlled comparison",
          "Check the plumbing first", "What the numbers say", "Findings & limits", "Thank you"]

prs = Presentation()
prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)   # 13.333 x 7.5 in, 16:9
for i in range(1, N + 1):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    pic = s.shapes.add_picture(os.path.join(shots, f"hd{WEB[i - 1]}.png"), 0, 0, prs.slide_width, prs.slide_height)
    pic.name = f"Slide {i} – {TITLES[i - 1]}"
    pic._element.nvPicPr.cNvPr.set("descr", TITLES[i - 1])   # alt text for screen readers
    s.notes_slide.notes_text_frame.text = notes[i]
out = os.path.join(ROOT, "docs", "Ripe_or_Not_presentation.pptx")
prs.save(out)
print(out, f"{os.path.getsize(out) / 1e6:.1f} MB | slides: {len(prs.slides)} | notes on all:",
      all(notes.get(i) for i in range(1, N + 1)))
