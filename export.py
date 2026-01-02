import json
import csv
from io import StringIO

def export_as_json(data):
    return json.dumps(data, indent=2)

def export_as_markdown(segments, summary):
    md = "# Meeting Summary\n\n"
    md += summary + "\n\n---\n\n## Transcript\n"
    for s in segments:
        md += f"- **{s['speaker']}** ({s['start']}s–{s['end']}s): {s['text']}\n"
    return md

def export_as_csv(segments):
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=["speaker", "start", "end", "text"])
    writer.writeheader()
    for s in segments:
        writer.writerow(s)
    return buf.getvalue()
#