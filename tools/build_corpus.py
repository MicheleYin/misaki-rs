"""Assemble the parity corpus: adversarial cases + real prose.

Two halves, kept separate on purpose:
  * adversarial -- hand-built to hit the paths most likely to diverge. Finds
    bugs. NOT representative of real text.
  * prose -- public-domain books. Says what fraction of real input is affected.

Report both. An implementation can score well on prose while failing every
heteronym, which is exactly the failure that sounds worst to a listener.
"""
import re
import sys
import urllib.request

# Straight from hexgrad/misaki examples/usage.py -- the one canonical case
# upstream publishes, including its [word](/phoneme/) link syntax.
UPSTREAM = [
    "[Misaki](/misˈɑki/) is a G2P engine designed for [Kokoro](/kˈOkəɹO/) models.",
]

ADVERSARIAL = [
    # --- heteronyms: POS decides pronunciation (790 such lexicon entries) ---
    "Please read the report I read yesterday.",
    "He will lead the team with a lead pipe.",
    "They record a record every week.",
    "We live near a live volcano.",
    "The bandage was wound around the wound.",
    "The dove dove into the bushes.",
    "I object to that object.",
    "The farm was used to produce produce.",
    "He did not desert his comrades in the desert.",
    "The soldier decided to desert in the desert.",
    "She was too close to the door to close it.",
    "The buck does funny things when the does are present.",
    "A bass was painted on the bass drum.",
    "They were too separate to separate the pieces.",
    "The insurance was invalid for the invalid.",
    "I had to subject the subject to a test.",
    "The wind was too strong to wind the sail.",
    "He wound the clock after the wound healed.",
    "Do not refuse to take out the refuse.",
    "The contract did not contract as expected.",
    # --- lookahead context: the/to/a before vowel vs consonant ---
    "The apple and the pear were on the table.",
    "The elephant ate the entire thing.",
    "I want to eat and to sleep.",
    "She went to Ohio to open an account.",
    "A unicorn and an umbrella.",
    "A hour is not an hour.",
    # --- acronyms and plurals (our b22fe80 fix) ---
    "I have three PDFs and two URLs to send you.",
    "The CEOs of several NGOs met.",
    "He bought two CDs and three DVDs.",
    "NASA and the FBI and the CIA.",
    "An API, two APIs, and some APIs' documentation.",
    "The GPUs cost more than the CPUs.",
    # --- numbers, ordinals, dates, times, currency ---
    "Chapter 12 begins on page 347 in 1999.",
    "It cost $1,250.75 and weighed 3.5 kg.",
    "Dr. Smith met Mr. Jones at 5:30 p.m. on Jan. 3rd.",
    "She finished 1st, he finished 2nd, and I was 23rd.",
    "In 1776 and again in 1812 and once more in 2026.",
    "The temperature was -40 degrees.",
    "Add 1/2 cup and 3/4 teaspoon.",
    "It is 100% certain that 50% will fail.",
    "Call 555-1234 or 1-800-555-0199.",
    "The price rose from £5 to €7 to $9.",
    "He ran 26.2 miles in 3:42:15.",
    "Room 101 on the 4th floor of building 7A.",
    # --- punctuation, quotes, dashes ---
    'She said "don\'t go" and then left.',
    "It was -- how shall I say it -- unexpected.",
    "Wait... what? No! Really?",
    "The cat's toys; the dogs' bones.",
    "(Parenthetical) [bracketed] {braced}",
    "A semicolon; a colon: and a comma, too.",
    "Hyphenated-words and re-entry and co-operate.",
    "'Single quotes' and “curly doubles” and ‘curly singles’.",
    # --- casing and symbols ---
    "ALL CAPS SENTENCE HERE.",
    "iPhone and eBay and McDonald's.",
    "e.g. and i.e. and etc. and vs.",
    "The naive cafe served souffle.",
    "Mr O'Brien and Mrs McCarthy-Jones.",
    # --- stress / multi-token words ---
    "A well-known fact about ice cream.",
    "The New York Times said so.",
    "He is a self-made man.",
    # --- link syntax (misaki's own markup) ---
    "[Kokoro](/kˈOkəɹO/) is fast.",
    "[hello](+2) and [hello](-1) and [hello](0.5).",
    # --- edge cases ---
    "A.",
    "I",
    "...",
    "a",
    "The",
    "Hmm.",
    "Shhh!",
    "Ok.",
]

BOOKS = [
    # (gutenberg id, start marker, how many sentences)
    (2554, r"\bPART\s+I\b", 200),      # Crime and Punishment
    (1342, r"\bChapter\s+1\b", 200),   # Pride and Prejudice
    (84, r"\bLetter\s+1\b", 150),      # Frankenstein
    (11, r"\bCHAPTER\s+I\b", 150),     # Alice in Wonderland
]

# Deliberately NOT used to develop against -- a held-out set, so the gate can
# be checked for overfitting to the corpora above.
HELDOUT = [
    (1661, r"\bI\.\s", 200),           # Sherlock Holmes
    (98, r"\bBook\s+the\s+First\b", 200),  # A Tale of Two Cities
    (345, r"\bCHAPTER\s+I\b", 200),    # Dracula
    (1232, r"\bCHAPTER\s+I\b", 150),   # The Prince
    (74, r"\bCHAPTER\s+I\b", 150),     # Tom Sawyer
]


def fetch_prose(gid, marker, limit):
    url = "https://www.gutenberg.org/cache/epub/%d/pg%d.txt" % (gid, gid)
    raw = urllib.request.urlopen(url, timeout=90).read().decode("utf-8", "ignore")
    m = re.search(r"\*\*\* START OF.*?\*\*\*(.*?)\*\*\* END OF", raw, re.S)
    body = m.group(1) if m else raw
    mm = re.search(marker, body)
    body = body[mm.end():] if mm else body
    text = re.sub(r"\s+", " ", body).strip()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    out = []
    for s in sents:
        if len(s.split()) < 4 or len(s) > 400:
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    lines = []
    if which in ("all", "adversarial"):
        lines += UPSTREAM + ADVERSARIAL
    if which == "heldout":
        for gid, marker, lim in HELDOUT:
            try:
                got = fetch_prose(gid, marker, lim)
                sys.stderr.write("  gutenberg %-6d %d sentences\n" % (gid, len(got)))
                lines += got
            except Exception as e:
                sys.stderr.write("  gutenberg %-6d FAILED: %s\n" % (gid, e))
    if which in ("all", "prose"):
        for gid, marker, lim in BOOKS:
            try:
                got = fetch_prose(gid, marker, lim)
                sys.stderr.write("  gutenberg %-6d %d sentences\n" % (gid, len(got)))
                lines += got
            except Exception as e:
                sys.stderr.write("  gutenberg %-6d FAILED: %s\n" % (gid, e))
    # de-dup, keep order
    seen, uniq = set(), []
    for l in lines:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    sys.stdout.write("\n".join(uniq) + "\n")
    sys.stderr.write("total %d lines\n" % len(uniq))


if __name__ == "__main__":
    main()
