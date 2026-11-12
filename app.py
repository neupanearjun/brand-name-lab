"""Brand Name Lab — candidate generation and preliminary screening.
Run: pip install -r requirements.txt && streamlit run app.py
"""
import csv
import json
import uuid
from pathlib import Path
import io
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from collections import Counter
import string
from difflib import SequenceMatcher
from urllib.parse import quote

import requests
import streamlit as st

st.set_page_config(page_title="Brand Name Lab", page_icon="✳", layout="wide")

# All domain results are preliminary; 'not found' cannot guarantee registration eligibility.
RDAP = {
    'com': 'https://rdap.verisign.com/com/v1/domain/{domain}',
    'ca': 'https://rdap.ca.fury.ca/rdap/domain/{domain}',
    'in': 'https://rdap.nixiregistry.in/rdap/domain/{domain}',
}
LINKS = {
    'Canada CIPO': 'https://ised-isde.canada.ca/cipo/trademark-search/',
    'USA USPTO': 'https://tmsearch.uspto.gov/',
    'India IP India': 'https://tmrsearch.ipindia.gov.in/',
    'WIPO Global': 'https://branddb.wipo.int/',
    'EU EUIPO': 'https://www.tmdn.org/tmview/',
    'UK IPO': 'https://trademarks.ipo.gov.uk/ipo-tmtext',
}
HEADERS = {'User-Agent': 'BrandNameLab/1.5', 'Accept': 'application/rdap+json, application/json;q=0.9'}

START = ['ba','be','bi','bo','da','de','di','do','fa','fe','fi','ga','ge','gi','go','ha','he','ka','ke','ki','ko','la','le','li','lo','ma','me','mi','mo','na','ne','ni','no','pa','pe','pi','po','ra','re','ri','ro','sa','se','si','so','ta','te','ti','to','va','ve','vi','vo','ya','ye']
MID = ['ra','ri','ro','la','le','li','lo','ma','me','mi','mo','na','ne','ni','no','va','ve','vi','vo','pa','pe','pi','po','ta','te','ti','to','ka','ke','ki','ko','da','de','di','do','sa','se','si','so']
END = ['na','ne','ni','no','ra','re','ri','ro','la','le','li','lo','ma','me','mi','mo','va','ve','vi','vo','ka','ke','ki','ko','ta','te','ti','to','pa','pe','pi','po']
CONSONANTS = 'bcdfghjklmnprstv'
VOWELS = 'aeiou'


def pronounceable(name):
    """A light heuristic, not a linguistic or legal guarantee."""
    if re.search(r'[^a-z]', name): return False
    if re.search(r'[bcdfghjklmnpqrstvwxyz]{3,}', name): return False
    if re.search(r'[aeiou]{3,}', name): return False
    if len(set(name)) < 4: return False
    return True


def parse_letter_pool(raw):
    """Parse the exact letter tiles provided by the user; duplicate entries are tiles."""
    if not raw.strip():
        return ()
    entries = [part.strip().lower() for part in raw.split(',')]
    if any(not re.fullmatch(r'[a-z]', item) for item in entries):
        raise ValueError('Enter single letters separated by commas, e.g. m, l, a, e, y.')
    if len(entries) > 40:
        raise ValueError('Use no more than 40 letter tiles at once.')
    return tuple(entries)


def pool_matches(name, pool):
    """Shorter names use only pool tiles; longer names use every pool tile."""
    if not pool:
        return True
    name_count, pool_count = Counter(name), Counter(pool)
    if len(name) <= len(pool):
        return all(count <= pool_count.get(ch, 0) for ch, count in name_count.items())
    return all(name_count.get(ch, 0) >= count for ch, count in pool_count.items())


def letter_pool_extra(rng, existing):
    """Add soft, familiar letters only when the target length exceeds pool size."""
    letters = 'aaaaeeeeiiiioooouuulllmmnnnrrrsssvvttkdp'
    vowel_count = sum(ch in 'aeiou' for ch in existing)
    # Avoid a hard-to-say result when all supplied tiles are consonants.
    if vowel_count < max(1, round((len(existing) + 1) * .34)):
        return rng.choice('aaeeeeiiiiooouu')
    return rng.choice(letters)


def soft_name_score(name, comment=''):
    """Simple pronunciation heuristic, not proof of language or brand suitability."""
    vowels = sum(ch in 'aeiou' for ch in name)
    score = 0.0
    score += 5 if 0.28 <= vowels / len(name) <= 0.65 else -5
    score += 1.5 if name[-1] in 'aeio' else 0
    score += 1 if name[0] in 'lmnrsv' else 0
    score += sum(0.6 for ch in name if ch in 'lmnrsv')
    score -= 9 * len(re.findall(r'[^aeiou]{3,}', name))
    score -= 7 * len(re.findall(r'[aeiou]{3,}', name))
    score -= 2 * len(re.findall(r'([a-z])\1', name))
    score -= 6 * len(re.findall(r'[^aeiou]{2}', name))
    score -= 3 if len(name) > 1 and name[:2].count('a') == 0 and all(ch not in 'aeiou' for ch in name[:2]) else 0
    score += 0.8 * sum((name[i] in 'aeiou') != (name[i+1] in 'aeiou')
                       for i in range(len(name)-1))
    # For a recognized concept, favor sounds also seen in its historical roots,
    # but NEVER relax letter-pool constraints just to follow a theme.
    category = theme_category(comment) if comment else None
    if category:
        roots = [r for r, _ in THEME_ROOTS[category]]
        for root in roots:
            for idx in range(len(root) - 1):
                if root[idx:idx+2] in name:
                    score += 0.18
    return score


def generate_from_pool(count, min_len, max_len, excluded, pool, first_letter='', comment='', seed=None):
    """Use only supplied letters when enough; add only deficit otherwise.

    If a fixed first letter is selected, it consumes one corresponding tile.
    Lengths shorter than the supplied pool draw a subset without replacement.
    """
    rng = random.Random(seed)
    pool = tuple(pool)
    if not pool:
        raise ValueError('A letter pool is required for this generation mode.')
    if first_letter and first_letter not in pool:
        raise ValueError('Choose a starting letter from your supplied letter pool.')
    seen = {str(x).strip().lower() for x in excluded}
    candidates = {}
    attempts = 0
    attempt_limit = min(70000, max(7000, count * 200))
    while attempts < attempt_limit:
        attempts += 1
        target = rng.randint(min_len, max_len)
        # A selected first letter MUST be drawn from the pool and never added.
        available = list(pool)
        if first_letter:
            available.remove(first_letter)
            slots = target - 1
        else:
            slots = target
        if slots <= len(available):
            remaining = rng.sample(available, slots)
        else:
            remaining = available.copy()
            for _ in range(slots - len(available)):
                remaining.append(letter_pool_extra(rng, remaining + ([first_letter] if first_letter else [])))
        rng.shuffle(remaining)
        candidate = first_letter + ''.join(remaining)
        if candidate in seen or candidate in candidates or candidate in LOCAL_LANGUAGE_RISKS:
            continue
        if not pool_matches(candidate, pool):
            continue
        # Discard the least pronounceable results, but don't require four
        # distinct characters when users intentionally supply duplicates.
        score = soft_name_score(candidate, comment)
        if score < -7:
            continue
        candidates[candidate] = (score, rng.random())
        # For smaller pools, near-exhaustive searching avoids premature repeats.
        if len(candidates) >= count * 5 and attempts >= count * 30:
            break
    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:count]]


def force_first_letter(candidate, letter):
    if not letter:
        return candidate
    return letter + candidate[1:] if candidate else candidate


def generate(count=50, min_len=5, max_len=8, seed=None, excluded=(), letter_pool=(), first_letter=''):
    if letter_pool:
        return generate_from_pool(count, min_len, max_len, excluded, letter_pool, first_letter, seed=seed)
    r = random.Random(seed)
    out, seen = [], set(x.lower().strip() for x in excluded)
    attempts = 0
    while len(out) < count and attempts < count * 350:
        attempts += 1
        mode = r.choice(['syllables', 'syllables', 'compact', 'flip'])
        if mode == 'syllables':
            candidate = r.choice(START) + r.choice(MID) + r.choice(END)
            if r.random() < 0.25: candidate = candidate[:-1]
        elif mode == 'compact':
            candidate = r.choice(START) + r.choice(MID) + r.choice(END)
            candidate = candidate[:r.choice([5,6,7])]
        else:
            candidate = r.choice(START) + r.choice(END) + r.choice(MID)
        candidate = force_first_letter(candidate, first_letter)
        if not min_len <= len(candidate) <= max_len or not pronounceable(candidate): continue
        if candidate in seen or candidate in LOCAL_LANGUAGE_RISKS: continue
        if any(candidate.endswith(s) for s in ['lumi','nova','wear','style','fashion']): continue
        if candidate[:2] == candidate[2:4] or candidate[2:4] == candidate[4:6]: continue
        seen.add(candidate)
        out.append(candidate)
    return out


# Small, transparent multilingual inspiration bank. These roots are inspiration,
# not trademark clearance, and their adapted forms are invented brand candidates.
THEME_ROOTS = {
    'clothing': [
        ('tela','Spanish/Italian: fabric'), ('fil','French: thread'),
        ('ito','Japanese: thread'), ('nuno','Japanese: cloth'),
        ('kapda','Hindi: cloth'), ('buna','Nepali/Hindi: weaving'),
        ('vastra','Sanskrit: clothing'), ('pata','Sanskrit: cloth'),
        ('kuma','Swahili: weaving (verify local usage)'),
        ('lin','English: linen association'), ('sil','English: silk association'),
        ('suta','Hindi/Nepali: yarn association'), ('weva','English: weave association'),
        ('dora','Nepali/Hindi: thread/string association'),
    ],
    'nature': [('mori','Japanese: forest'), ('sora','Japanese: sky'), ('terra','Latin: earth'),
               ('nadi','Hindi/Nepali: river'), ('vana','Sanskrit: forest'),
               ('flor','Latin: flower'), ('luna','Latin: moon'), ('sol','Latin: sun')],
    'comfort': [('sukha','Sanskrit: ease/happiness'), ('calma','Spanish/Italian: calm'),
                ('molu','soft-sounding invented root'), ('nido','Spanish: nest'),
                ('soma','Greek: body'), ('mira','soft-sounding root')],
    'travel': [('yatra','Sanskrit: journey'), ('via','Latin: road'),
               ('safar','Hindi/Urdu: journey'), ('roam','English: roaming'),
               ('tavi','invented travel-sounding root')],
}
THEME_TERMS = {
    'clothing': ('clothing','clothes','apparel','fashion','fabric','textile','thread','weave','garment','fleece','outfit','wear','cotton','silk','tailor','knit'),
    'nature': ('nature','earth','forest','water','river','ocean','flower','green','plant','mountain'),
    'comfort': ('comfort','cozy','cosy','soft','warm','relax','calm','sleep'),
    'travel': ('travel','journey','trip','wander','adventure'),
}


def theme_category(comment):
    cleaned = comment.casefold().strip()
    for key, words in THEME_TERMS.items():
        if any(re.search(r'\b' + re.escape(word) + r'\b', cleaned) for word in words):
            return key
    return None


def themed_generate(count, min_len, max_len, excluded, letter_pool, first_letter, comment, seed=None):
    """Offline thematic blends. Returns [] for an unknown topic (no fake relevance)."""
    category = theme_category(comment)
    if not category:
        return []
    if letter_pool:
        # A letter pool is a hard constraint: a theme only nudges ranking.
        return generate_from_pool(count, min_len, max_len, excluded, letter_pool,
                                  first_letter, comment=comment, seed=seed)
    rng = random.Random(seed)
    roots = [root for root, _ in THEME_ROOTS[category]]
    endings = ['ra','ri','ro','la','li','lo','na','ni','no','ma','mi','mo','va','vi','ka','ki','ko','re','el','en','io']
    beginnings = ['a','e','o','ka','ma','ne','va','lo','mi','ta']
    found = []
    seen = {x.strip().lower() for x in excluded}
    for _ in range(count * 600):
        root = rng.choice(roots)
        mode = rng.randrange(5)
        if mode == 0:
            candidate = root + rng.choice(endings)
        elif mode == 1:
            candidate = rng.choice(beginnings) + root
        elif mode == 2:
            other = rng.choice(roots)
            candidate = root[:rng.randint(2, min(4, len(root)))] + other[-rng.randint(2, min(3, len(other))):]
        elif mode == 3:
            candidate = root[:rng.randint(2, min(4, len(root)))] + rng.choice(END)
        else:
            candidate = rng.choice(START) + root[-rng.randint(2,min(4,len(root))):] + rng.choice(['a','i','o',''])
        candidate = re.sub('[^a-z]', '', candidate.lower())
        candidate = force_first_letter(candidate, first_letter)
        if candidate in seen or candidate in LOCAL_LANGUAGE_RISKS: continue
        if not min_len <= len(candidate) <= max_len: continue
        if not pronounceable(candidate): continue
        seen.add(candidate)
        found.append(candidate)
        if len(found) >= count: break
    return found


def ai_themed_generate(comment, count, min_len, max_len, excluded, letter_pool, first_letter, api_key):
    """Optional paid model-based creative generation for arbitrary user topics/languages."""
    if not api_key:
        return [], 'No OpenAI API key — only built-in themes available'
    prompt = (
        'Invent original, easy-to-spell, pronounceable apparel/lifestyle brand candidates inspired by '
        'the following user concept. Draw creative inspiration from languages worldwide, including '
        'Nepali, Sanskrit and other regional languages when appropriate, but never claim words are '
        'verified translations. Blend and change roots into coined words, not exact dictionary words. '
        'Do not imply domain or trademark availability. Output a JSON object with key names, an array '
        f'of up to {min(count*2, 250)} lowercase alphabetic names, each {min_len} to {max_len} letters. '
        f'Letter pool: {list(letter_pool)} (repeated letters are separate tiles). '
        'HARD CONSTRAINT: when a candidate is no longer than the pool size, use '
        'ONLY supplied letter tiles, each at most as often as it occurs in the pool. '
        'When longer than the pool, use EVERY supplied tile and add only the '
        'number of extra letters needed. '
        f'Every name MUST start with {first_letter} when that letter is nonempty. '
        f'User concept: {comment[:500]}'
    )
    try:
        resp = requests.post('https://api.openai.com/v1/chat/completions',
            headers={'Authorization': 'Bearer ' + api_key, 'Content-Type':'application/json'},
            json={'model':'gpt-4.1-mini', 'messages':[{'role':'user','content':prompt}],
                  'response_format':{'type':'json_object'}, 'temperature':1}, timeout=55)
        if resp.status_code != 200:
            return [], f'OpenAI generation API error HTTP {resp.status_code}'
        raw = json.loads(resp.json()['choices'][0]['message']['content']).get('names', [])
        if not isinstance(raw, list): return [], 'AI returned an unexpected format'
        seen = {str(x).lower() for x in excluded}
        out = []
        for entry in raw:
            n = str(entry).strip().lower()
            if (n in seen or n in LOCAL_LANGUAGE_RISKS or not min_len <= len(n) <= max_len
                or not pronounceable(n) or not pool_matches(n, letter_pool)
                or (first_letter and not n.startswith(first_letter))):
                continue
            out.append(n)
            seen.add(n)
            if len(out) >= count: break
        return out, 'AI themed generation completed — names are not cleared'
    except (requests.RequestException, KeyError, ValueError, TypeError, IndexError) as exc:
        return [], f'AI generation failed ({type(exc).__name__})'


def rdap_check(domain):
    tld = domain.rsplit('.',1)[-1]
    url = RDAP.get(tld, 'https://rdap.org/domain/{domain}').format(domain=domain)
    try:
        response = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get('objectClassName') == 'domain' or data.get('ldhName') or data.get('unicodeName'):
                    return 'REGISTERED', url
            except ValueError: pass
            return 'UNKNOWN', url
        if response.status_code == 404:
            return 'NOT FOUND — verify at registrar', url
        if response.status_code in (403, 429):
            return 'RATE-LIMITED / BLOCKED', url
        return f'UNKNOWN (HTTP {response.status_code})', url
    except requests.RequestException as exc:
        return f'UNKNOWN ({type(exc).__name__}: {str(exc)[:90]})', url


def serper_search(name, api_key):
    """Uses the user's own Serper.dev API key for Google search results."""
    if not api_key: return None, 'Not run — add a Serper API key', []
    try:
        response = requests.post('https://google.serper.dev/search',
            headers={'X-API-KEY': api_key, 'Content-Type': 'application/json'},
            json={'q': f'"{name}"', 'num': 10}, timeout=12)
        if response.status_code != 200: return None, f'API error {response.status_code}', []
        data = response.json()
        results = data.get('organic', [])
        hits = []
        for item in results:
            snippet = (item.get('title','') + ' ' + item.get('snippet','')).lower()
            matched = name.lower() in re.sub(r'[^a-z0-9]', '', snippet)
            hits.append({'title':item.get('title',''), 'url':item.get('link',''), 'snippet':item.get('snippet',''), 'exact_hint':matched})
        return len(results), 'Search completed — manually inspect matches', hits
    except requests.RequestException as exc:
        return None, 'API request failed', []


# Curated EXACT spellings only. Matching substrings would produce many false flags.
# This is an intentionally small list; absence is NOT a language or slang clearance.
LOCAL_LANGUAGE_RISKS = {
    'muji': [('Nepali', 'VERY HIGH')],
    'lado': [('Nepali', 'VERY HIGH')],
    'lund': [('Hindi', 'VERY HIGH'), ('Urdu', 'VERY HIGH')],
    'randi': [('Hindi', 'VERY HIGH'), ('Nepali', 'VERY HIGH')],
    'gandu': [('Hindi', 'VERY HIGH'), ('Nepali', 'VERY HIGH')],
    'pendejo': [('Spanish', 'HIGH')],
    'putain': [('French', 'HIGH')],
    'merde': [('French', 'MEDIUM')],
    'culo': [('Spanish', 'MEDIUM')],
}
RISK_ORDER = {'MEDIUM': 1, 'HIGH': 2, 'VERY HIGH': 3}


def local_language_risk(name):
    normalized = re.sub(r'[^a-z]', '', name.lower())
    return [{'language': lang, 'risk': risk, 'source': 'curated local list'}
            for lang, risk in LOCAL_LANGUAGE_RISKS.get(normalized, [])]


def ai_language_risk(name, api_key):
    """Optional, paid preliminary language review; NOT a verified dictionary lookup."""
    if not api_key:
        return [], 'Not run — no AI API key'
    prompt = (
        'Review this exact coined clothing brand name for vulgar, sexual, insulting, '
        'offensive or severely embarrassing meanings and COMMON near-homophones in '
        'languages worldwide, specifically including Nepali romanization and Devanagari, '
        'Hindi, Urdu, English, Spanish, Portuguese, French, German, Italian, Arabic, '
        'Turkish, Japanese, Korean and Chinese. Do not invent etymologies or claim exhaustive coverage. '
        'Return ONLY JSON object with key "findings", an array of objects containing '
        '"language", "risk" (one of MEDIUM, HIGH, VERY HIGH), and "note" (short reason). '
        'Omit low risk and uncertain guesses. When no reliable high-confidence findings, use empty array. '
        f'Brand name: {name}'
    )
    try:
        response = requests.post('https://api.openai.com/v1/chat/completions',
            headers={'Authorization': 'Bearer ' + api_key, 'Content-Type':'application/json'},
            json={'model':'gpt-4.1-mini', 'messages':[{'role':'user','content':prompt}],
                  'response_format':{'type':'json_object'}, 'temperature':0}, timeout=35)
        if response.status_code != 200:
            return [], f'AI API error {response.status_code}'
        raw = response.json()['choices'][0]['message']['content']
        parsed = json.loads(raw)
        findings = []
        for item in parsed.get('findings', [])[:12]:
            language = str(item.get('language', '')).strip()[:45]
            risk = str(item.get('risk', '')).strip().upper()
            if language and risk in RISK_ORDER:
                findings.append({'language':language, 'risk':risk,
                    'source':'AI suggestion — verify with native speaker',
                    'note':str(item.get('note',''))[:240]})
        return findings, 'AI preliminary review completed — not independently verified'
    except (requests.RequestException, KeyError, ValueError, TypeError, IndexError) as exc:
        return [], f'AI review failed ({type(exc).__name__})'


def combine_risks(local, ai):
    combined = {}
    for item in local + ai:
        lang = item['language'].strip()
        k = lang.casefold()
        if k not in combined or RISK_ORDER[item['risk']] > RISK_ORDER[combined[k]['risk']]:
            combined[k] = item
    return sorted(combined.values(), key=lambda x: (-RISK_ORDER[x['risk']], x['language']))


def language_remarks(findings, ai_status):
    if findings:
        return '; '.join(f"{x['language']}: {x['risk']}" for x in findings)
    if ai_status.startswith('AI preliminary review completed'):
        return 'No medium+ risk flagged (limited AI/local check)'
    return 'NOT CHECKED globally — no local match'


def candidate_worker(name, key, domains, ai_key=""):
    result = {'name': name, 'searched_at_utc':datetime.now(timezone.utc).isoformat()}
    for tld in domains:
        status, url = rdap_check(f'{name}.{tld}')
        result[f'.{tld}'] = status
        result[f'.{tld} RDAP'] = url
    count, status, hits = serper_search(name, key)
    result['web_hits'] = count if count is not None else 'not checked'
    result['web_status'] = status
    result['web_results'] = hits
    # The prioritization is intentionally conservative, not a legal clearance score.
    domain_registered = any(result.get('.'+t) == 'REGISTERED' for t in domains)
    domain_unknown = any('UNKNOWN' in result.get('.'+t,'') or 'BLOCKED' in result.get('.'+t,'') or 'RATE-LIMITED' in result.get('.'+t,'') for t in domains)
    exact_web = any(h['exact_hint'] for h in hits)
    if domain_registered or exact_web:
        result['priority'] = 'REVIEW CONFLICT — registered domain or web use'
    elif not key or domain_unknown or count is None:
        result['priority'] = 'INCOMPLETE — more checks needed'
    else:
        result['priority'] = 'REVIEW — no obvious hit in limited checks'
    result['trademark'] = 'NOT CHECKED — official manual clearance required'
    ai_findings, ai_status = ai_language_risk(name, ai_key)
    findings = combine_risks(local_language_risk(name), ai_findings)
    result['language_risk'] = findings
    result['language_review_status'] = ai_status
    result['remarks'] = language_remarks(findings, ai_status)
    result['language_risk_rank'] = max((RISK_ORDER[f['risk']] for f in findings), default=0)
    return result


def export_csv(records):
    cols = ['name', '.com', '.ca', '.in', 'remarks', 'language_review_status', 'web_hits', 'web_status', 'trademark', 'priority', 'searched_at_utc']
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=cols, extrasaction='ignore')
    writer.writeheader(); writer.writerows(records)
    return out.getvalue().encode('utf-8-sig')


st.title('✳ Brand Name Lab')
st.caption('Invent names → screen domains and web usage → flag medium/high/very-high language risks → review trademarks.')
st.warning('Global slang screening is NOT exhaustive. AI findings may be inaccurate, and no flag is not proof of cultural safety. Domain checks and trademarks also require independent verification.')

with st.sidebar:
    st.header('Generate')
    count = st.slider('Number of names', 10, 200, 50, step=10)
    sizes = st.slider('Letter length', 4, 10, (5, 7))
    requested_letter_pool = st.text_input(
        'Letter pool (optional; separate each letter with a comma)',
        placeholder='e.g. m, l, a, y, e',
        help='Use ONLY these letters when the name length is at most the pool size. '
             'When the name must be longer, the app uses every supplied letter and '
             'adds only enough extra letters to reach the requested length. '
             'Repeated letters count as separate tiles.'
    )
    # Derive the dropdown live so a fixed first letter always comes from
    # the pool, rather than accidentally adding a foreign letter.
    try:
        preview_pool = parse_letter_pool(requested_letter_pool)
        pool_error = ''
    except ValueError as exc:
        preview_pool, pool_error = (), str(exc)
    if pool_error:
        st.error(pool_error)
    first_options = ['Any'] + (sorted(set(preview_pool)) if preview_pool else list(string.ascii_lowercase))
    preferred_first = st.selectbox(
        'Start names with (optional)', first_options,
        format_func=lambda x: 'Any starting letter' if x == 'Any' else x.upper(),
        help='Choose the first character. When you entered a pool, only letters '
             'from that pool can be chosen, and the chosen letter is consumed from it.'
    )
    first_letter = '' if preferred_first == 'Any' else preferred_first
    if preview_pool:
        lo, hi = sizes
        if lo == hi:
            explanation = (f'{len(preview_pool)} supplied tiles → names of {lo} letters: '
                           + ('use only a subset of your tiles' if lo < len(preview_pool)
                              else 'use all of your tiles' if lo == len(preview_pool)
                              else f'use all tiles plus {lo-len(preview_pool)} new letters'))
        else:
            explanation = (f'{len(preview_pool)} supplied tiles → names of {lo}–{hi} letters; '
                           'use only pool tiles when enough, otherwise add the exact deficit.')
        st.caption(explanation)
    inspiration = st.text_area(
        'Name inspiration / comments (optional)',
        placeholder='e.g. clothing, soft fleece, premium everyday apparel, inspired by Nepali and Japanese textile words',
        help='Blank = completely random. With text, use topic-inspired roots. An optional OpenAI API key supports custom themes and broader language inspiration.'
    )
    st.caption('Blank inspiration = random names. Text = related invented names (not verified words or trademarks).')
    st.caption('Example: m, l, a, y, e + length 5 uses exactly those five letters. Length 7 adds two extra letters. Twelve supplied letters + length 5 takes a five-letter subset.')
    st.caption('The letter pool and starting letter apply only to NEW batches. Saved batches and screening history are unchanged.')
    excluded = st.text_area('Names to exclude (one per line)', value='milay\nvilonuma\nvinoluma\naritzia\nlululemon')
    st.divider()
    st.header('Automatic checks')
    st.caption('Domain records use RDAP. A “not found” response is NOT guaranteed availability.')
    selected_domains = st.multiselect('Domain extensions', ['com','ca','in'], default=['com','ca','in'])
    serper_key = st.text_input('Serper.dev API key (optional)', value=os.environ.get('SERPER_API_KEY',''), type='password', help='Needed for automated indexed Google web results. Key is used only for API requests while this app runs.')
    st.caption('Without a web API key, the app can still generate names, run RDAP domain checks, and provide manual search links.')
    st.divider()
    st.header('Language & slang risks')
    ai_key = st.text_input('OpenAI API key (optional; paid usage)', value=os.environ.get('OPENAI_API_KEY',''), type='password', help='Enables an AI preliminary review across major languages. Not a verified dictionary or native-speaker clearance. Separate from a ChatGPT subscription.')
    st.caption('Without this API key, ONLY a tiny curated list of known offensive exact words is checked. No flag does NOT mean a name is safe.')
    workers = st.slider('Concurrent checks (use modestly)', 1, 5, 2)

# Persistent, local data: survives browser refreshes and app restarts.
DATA_DIR = Path(__file__).resolve().parent / 'data'
DATA_FILE = DATA_DIR / 'workspace.json'


def load_workspace():
    try:
        if DATA_FILE.exists():
            state = json.loads(DATA_FILE.read_text(encoding='utf-8'))
            if isinstance(state, dict):
                return {
                    'batches': state.get('batches', []),
                    'results': state.get('results', []),
                    'excluded': state.get('excluded', []),
                }
    except (OSError, ValueError, TypeError) as err:
        st.error(f'Could not open saved workspace: {err}')
    return {'batches': [], 'results': [], 'excluded': []}


def save_workspace():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {k: st.session_state.workspace[k] for k in ('batches', 'results', 'excluded')}
    temporary = DATA_FILE.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(DATA_FILE)


def all_names(workspace):
    return list(dict.fromkeys(name for batch in workspace['batches'] for name in batch['names']))


def fully_not_found(result, domains):
    # Never treat missing, unknown, errors or blocked statuses as available.
    return bool(domains) and all(result.get('.' + d, '') == 'NOT FOUND — verify at registrar' for d in domains)


def result_sort_key(result, domains):
    # Sort preliminary unregistered-domain candidates first, then all others.
    return (0 if fully_not_found(result, domains) else 1, result['name'].lower())


if 'workspace' not in st.session_state:
    st.session_state.workspace = load_workspace()
if 'screen_selection' not in st.session_state:
    st.session_state.screen_selection = []
if 'chosen_batches' not in st.session_state:
    st.session_state.chosen_batches = []
if 'last_screened_batch_ids' not in st.session_state:
    st.session_state.last_screened_batch_ids = []


def on_batch_change():
    # Crucial: old multiselect names must not carry over to a different batch.
    st.session_state.screen_selection = []


def scoped_names(workspace, chosen_ids):
    allowed_ids = set(chosen_ids)
    return list(dict.fromkeys(
        name for batch in workspace['batches'] if batch['id'] in allowed_ids
        for name in batch['names']
    ))


def build_queue(chosen_names, selected_names, result_map, include_previously_checked):
    # Hard safety boundary: ONLY explicitly selected names from selected batches.
    allowed = set(chosen_names)
    seen = set()
    return [name for name in selected_names
            if name in allowed and (include_previously_checked or name not in result_map)
            and not (name in seen or seen.add(name))]


ws = st.session_state.workspace
names = all_names(ws)
prior_results = {r['name']: r for r in ws['results']}

left, right = st.columns([1, 2])
with left:
    if st.button('Generate fresh batch', type='primary', use_container_width=True):
        exclusions = set(excluded.lower().splitlines()) | set(names) | set(ws['excluded'])
        try:
            pool = parse_letter_pool(requested_letter_pool)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        topic = inspiration.strip()
        generation_mode = 'random'
        if topic:
            # An AI key enables arbitrary topics and multilingual inspiration. Built-in
            # offline themes work without a key and avoid adding latency/API cost.
            if ai_key:
                fresh, generation_status = ai_themed_generate(topic, count, sizes[0], sizes[1], exclusions, pool, first_letter, ai_key)
                generation_mode = 'AI thematic'
                if not fresh:
                    st.warning(generation_status)
            else:
                fresh = themed_generate(count, sizes[0], sizes[1], exclusions, pool, first_letter, topic)
                generation_mode = 'built-in thematic'
                if not theme_category(topic):
                    st.warning('This topic is not in the offline inspiration bank. Add an OpenAI API key for arbitrary topics/language-inspired names, or try clothing, nature, comfort, or travel. No unrelated random names were substituted.')
        else:
            fresh = generate(count, sizes[0], sizes[1], seed=None, excluded=exclusions,
                             letter_pool=pool, first_letter=first_letter)
        if fresh:
            batch = {
                'id': uuid.uuid4().hex[:12],
                'label': f'Batch {len(ws["batches"]) + 1} — {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'names': fresh,
                'letter_pool': list(pool),
                'first_letter': first_letter,
                'inspiration': topic,
                'generation_mode': generation_mode,
            }
            ws['batches'].append(batch)
            st.session_state.chosen_batches = [batch['id']]
            st.session_state.screen_selection = fresh.copy()
            save_workspace()
            if len(fresh) < count:
                st.warning(f'Only {len(fresh)} new unique names generated. Try a longer name length or a larger letter pool; a small fixed pool has limited unique combinations.')
            st.rerun()
        st.warning('No new names generated. Try a larger letter pool, a wider letter range, or a different inspiration topic. If all your supplied letters are consonants and no extra letters are permitted, there may be no easily pronounceable names.')
    custom = st.text_area('Or paste your own names, one per line', height=90)
    if st.button('Save pasted names as a batch', use_container_width=True):
        additions = list(dict.fromkeys(re.sub('[^a-z]', '', x.lower()) for x in custom.splitlines()))
        additions = [x for x in additions if len(x) >= 3 and x not in names and x not in ws['excluded']]
        if additions:
            batch = {'id': uuid.uuid4().hex[:12],
                     'label': f'Custom batch {len(ws["batches"]) + 1} — {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                     'created_at_utc': datetime.now(timezone.utc).isoformat(), 'names': additions}
            ws['batches'].append(batch)
            st.session_state.chosen_batches = [batch['id']]
            st.session_state.screen_selection = additions.copy()
            save_workspace()
            st.rerun()
        st.info('No new valid names to add.')

with right:
    st.subheader('Saved batches')
    if ws['batches']:
        label_to_id = {b['label'] + f' ({len(b["names"])} names)': b['id'] for b in ws['batches']}
        batch_ids = list(label_to_id.values())
        st.session_state.chosen_batches = [x for x in st.session_state.chosen_batches if x in batch_ids]
        chosen_batch_ids = st.multiselect('Choose batches to view and screen',
                          options=batch_ids,
                          format_func=lambda v: next((label for label, value in label_to_id.items() if value == v), v),
                          key='chosen_batches', on_change=on_batch_change)
        current_names = scoped_names(ws, chosen_batch_ids)
        st.caption(f'{len(ws["batches"])} saved batches · {len(names)} unique names · {len(prior_results)} screened names')
        if current_names:
            st.text_area('Names in selected batches', '\n'.join(n.upper() for n in current_names), height=160)
        st.download_button('Download all names (TXT)', '\n'.join(names), 'all_brand_names.txt', 'text/plain')
        st.download_button('Backup all batches + screenings (JSON)',
                           json.dumps(ws, ensure_ascii=False, indent=2), 'brand_name_lab_backup.json', 'application/json')
        with st.expander('Restore workspace from a backup'):
            uploaded = st.file_uploader('Choose a Brand Name Lab JSON backup', type=['json'])
            if st.button('Restore backup', disabled=uploaded is None):
                try:
                    parsed = json.loads(uploaded.getvalue().decode('utf-8'))
                    assert isinstance(parsed.get('batches'), list) and isinstance(parsed.get('results'), list)
                    st.session_state.workspace = {'batches': parsed['batches'],
                                                  'results': parsed['results'], 'excluded': parsed.get('excluded', [])}
                    st.session_state.chosen_batches = []
                    st.session_state.screen_selection = []
                    save_workspace()
                    st.rerun()
                except (ValueError, UnicodeDecodeError, AssertionError, KeyError) as exc:
                    st.error(f'Invalid backup: {exc}')
    else:
        current_names = []
        st.info('Generate a batch. It will automatically be saved to this Mac.')

st.divider()
st.subheader('Screen candidates')
if ws['batches']:
    st.caption('Only the batches selected under **Saved batches** will be screened. Previously screened names are skipped by default.')
    chosen_labels = [b['label'] for b in ws['batches'] if b['id'] in st.session_state.chosen_batches]
    if chosen_labels:
        st.info('Screening scope: ' + ' · '.join(chosen_labels) + f' — {len(current_names)} total names')
    else:
        st.warning('No batch selected. Choose one or more saved batches above to enable screening.')
    already_checked = [n for n in current_names if n in prior_results]
    unchecked = [n for n in current_names if n not in prior_results]
    show_checked = st.checkbox('Include previously screened names in selection (to rerun them)', value=False)
    available_names = current_names if show_checked else unchecked
    valid = set(available_names)
    st.session_state.screen_selection = [n for n in st.session_state.screen_selection if n in valid]
    controls = st.columns(3)
    with controls[0]:
        if st.button('Select all in chosen batches', use_container_width=True):
            st.session_state.screen_selection = available_names.copy()
            st.rerun()
    with controls[1]:
        if st.button('Select unchecked only', use_container_width=True):
            st.session_state.screen_selection = unchecked.copy()
            st.rerun()
    with controls[2]:
        if st.button('Clear selection', use_container_width=True):
            st.session_state.screen_selection = []
            st.rerun()
    names_to_check = st.multiselect('Select names to screen', available_names, key='screen_selection')
    st.caption(f'{len(names_to_check)} selected · {len(unchecked)} never screened · {len(already_checked)} previously screened in selected batches')
    queued = build_queue(current_names, names_to_check, prior_results, show_checked)
    st.write(f'**Ready to screen: {len(queued)} names from {len(chosen_labels)} selected batch(es).**')
    with st.expander('Preview the exact names that will be screened'):
        st.code('\n'.join(queued) if queued else '(none)', language=None)
    if show_checked:
        st.warning('Previously screened selections will be checked again and their old results replaced.')
    if len(queued) > 50:
        st.warning('Checking many names may cause RDAP rate limits or consume paid web-search requests.')
    if st.button(f'Run preliminary screening ({len(queued)} names)',
                 disabled=not queued or not selected_domains, type='primary'):
        # Make a fixed snapshot at click time; never use all saved names here.
        run_queue = build_queue(scoped_names(ws, st.session_state.chosen_batches),
                                queued, prior_results, show_checked)
        if not run_queue:
            st.error('No names in the selected batch(es) are queued.')
            st.stop()
        progress = st.progress(0, text=f'Checking {len(run_queue)} names in selected batches…')
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(candidate_worker, n, serper_key, selected_domains, ai_key) for n in run_queue]
            for i, future in enumerate(as_completed(futures)):
                try:
                    rows.append(future.result())
                except Exception as exc:
                    st.error(f'Unable to process a name: {exc}')
                progress.progress((i + 1)/len(futures), text=f'Checked {i+1}/{len(futures)} names')
        st.session_state.last_screened_batch_ids = list(st.session_state.chosen_batches)
        prior_results.update({r['name']: r for r in rows})
        ws['results'] = list(prior_results.values())
        save_workspace()
        st.success(f'Saved screening results for {len(rows)} names.')
        st.rerun()

if ws['results']:
    st.subheader('Results: language-risk flags and domain findings')
    st.caption('Remarks shows language and MEDIUM/HIGH/VERY HIGH risk where flagged. Names without a complete language check are not labeled safe. Domain NOT FOUND still needs registrar verification.')
    view = st.radio('Display results', ['Selected batches only', 'All saved results'], horizontal=True, index=0)
    st.caption('This table is historical saved results, not the current screening queue. Switch to All saved results only if you want to browse other batches.')
    allowed = set(current_names) if view == 'Selected batches only' else None
    rows = [r for r in ws['results'] if allowed is None or r['name'] in allowed]
    # No medium-or-higher language flags first, then all-domain NOT FOUND.
    # Old history remains visibly NOT CHECKED rather than silently treated as clean.
    for r in rows:
        if 'remarks' not in r:
            r['remarks'] = 'NOT CHECKED — pre-v1.5 result (rerun screening)'
    rows = sorted(rows, key=lambda r: (
        0 if fully_not_found(r, ['com', 'ca', 'in']) else 1,
        1 if r.get('language_risk_rank',0) > 0 else 0,
        r['name'].lower()))
    matches = sum(fully_not_found(r, ['com', 'ca', 'in']) for r in rows)
    st.write(f'**{matches}** of **{len(rows)}** shown have NOT FOUND for all three: .com, .ca and .in.')
    cols = ['name', '.com', '.ca', '.in', 'remarks', 'web_hits', 'priority', 'trademark', 'searched_at_utc']
    st.dataframe([{k: r.get(k, '—') for k in cols} for r in rows], use_container_width=True, hide_index=True)
    st.download_button('Download sorted screening CSV', export_csv(rows), 'brand_name_screening.csv', 'text/csv')
    if rows:
        chosen = st.selectbox('Inspect name evidence', [r['name'] for r in rows])
        record = next(r for r in rows if r['name'] == chosen)
        st.write(f"**Web:** {record.get('web_status', 'not checked')}")
        st.write(f"**Language screen:** {record.get('language_review_status', 'NOT CHECKED — old result')}")
        for finding in record.get('language_risk', []):
            st.write(f"- {finding['language']}: {finding['risk']} — {finding.get('source','unverified')}")
            if finding.get('note'):
                st.caption(finding['note'])
        for hit in record.get('web_results', []):
            st.markdown(f"- [{hit['title']}]({hit['url']}) — {hit['snippet'][:180]}")
        for t in selected_domains:
            rdap_link = record.get('.' + t + ' RDAP', '')
            st.markdown(f"- **{chosen}.{t}** — {record.get('.' + t, 'not checked')}" +
                        (f' · [RDAP record]({rdap_link})' if rdap_link else ''))

st.divider()
st.subheader('Official trademark review — REQUIRED before choosing a name')
st.write('This step is manual, because public trademark registers are not all suitable or licensed for automated worldwide name searching. Check exact names, spelling variants, phonetic similarities, translations, and relevant goods/services (especially Nice Class 25).')
for name,url in LINKS.items():
    st.markdown(f'- [{name}]({url})')

st.subheader('Search links for any finalist')
query_name = st.text_input('Name to investigate manually', value=names[0] if names else '')
if query_name:
    q = quote('"'+query_name+'"')
    st.markdown(f'[Google exact-name search](https://www.google.com/search?q={q}) · [Bing exact-name search](https://www.bing.com/search?q={q}) · [Google similar apparel search](https://www.google.com/search?q={quote(query_name+" clothing brand")})')

st.info('DECISION RULE: shortlist only after clean preliminary checks. Then verify all domains at a registrar checkout and perform professional trademark clearance in target markets. “No hits” is never proof of legal availability.')
