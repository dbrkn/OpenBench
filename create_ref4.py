import csv, json, os, shutil

# Additional terms to remove on top of the original 50
extra_remove_en = {
    "allergic", "cell", "eyelid", "fatigue", "hormonal", "veins",
    "syndrome", "surgery", "sunburn", "solar", "scar", "rash",
    "pacemaker", "needle", "liquid", "crusting", "counseling",
    "drainage", "cough", "fever", "injection", "joint", "led",
    "morphology", "breath",
}

# Original 50
original_remove_en = {
    "anxiety", "appearing", "au", "baldness", "behavior", "bite", "body",
    "breast", "cafe", "canceled", "contact", "corn", "cosmetic", "damage",
    "dress", "exam", "extractions", "facial", "failure", "full", "hair",
    "heart", "insect", "loss", "male", "massage", "monitor", "mosquito",
    "nitrogen", "oily", "pain", "patch", "pattern", "punch", "quote",
    "recommendations", "recurrent", "referral", "removal", "scaling",
    "shave", "shortness", "skin", "tag", "tags", "tattoo", "thinning",
    "uncertain", "unspecified", "waxing",
}

all_remove_en = original_remove_en | extra_remove_en
print(f"Total EN terms to remove: {len(all_remove_en)} (50 original + {len(extra_remove_en)} new)")

# Load CSV for ES translations
en_to_es = {}
with open("/Users/berkin/Downloads/terms_translated_v3 (1).csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        en = row["term"].strip().lower()
        es = row["term_spanish"].strip()
        if " " not in en:
            en_to_es[en] = es

all_remove_es = set()
for t in all_remove_en:
    if t in en_to_es:
        all_remove_es.add(en_to_es[t].lower())

print(f"Total ES phrases to remove: {len(all_remove_es)}")


def collect_global_keywords(ref_dir):
    all_terms = []
    seen = set()
    for fn in sorted(os.listdir(ref_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(ref_dir, fn)) as f:
            d = json.load(f)
        for t in d.get("dictionary", []):
            if t.lower() not in seen:
                all_terms.append(t)
                seen.add(t.lower())
    return all_terms


def build_excluded(src_dir, out_dir, remove_set):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(src_dir, out_dir)

    total_before = 0
    total_after = 0
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(out_dir, fn)
        with open(path) as f:
            d = json.load(f)
        old = d.get("dictionary", [])
        new = [t for t in old if t.lower() not in remove_set]
        total_before += len(old)
        total_after += len(new)
        d["dictionary"] = new
        with open(path, "w") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    gk = collect_global_keywords(out_dir)
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(out_dir, fn)
        with open(path) as f:
            d = json.load(f)
        d["keywords"] = gk
        with open(path, "w") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    unique = set()
    for fn in os.listdir(out_dir):
        if fn.endswith(".json"):
            with open(os.path.join(out_dir, fn)) as f:
                d = json.load(f)
            unique.update(t.lower() for t in d.get("dictionary", []))

    print(f"  Before: {total_before}, After: {total_after}, Removed: {total_before - total_after}")
    print(f"  Unique: {len(unique)}, Global keywords: {len(gk)}")


# English
print("\n=== EN reference_medical_only ===")
build_excluded(
    "/Users/berkin/Desktop/mystique/reference_og",
    "/Users/berkin/Desktop/mystique/reference_medical_only",
    all_remove_en,
)

# Spanish
print("\n=== ES reference_medical_only ===")
build_excluded(
    "/Users/berkin/Desktop/mystique_spanish/reference_og",
    "/Users/berkin/Desktop/mystique_spanish/reference_medical_only",
    all_remove_es,
)
