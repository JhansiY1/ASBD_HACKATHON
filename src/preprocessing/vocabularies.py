"""
Controlled vocabularies of the UCI Movie dataset.

Every code table below is transcribed from the dataset documentation
(data/raw/movie/doc.html, sections 3.13, 4.3, 4.4 and 4.5, and the preamble
of main.html).  They are used for category standardisation and for detecting
inconsistent values: any token that is not in a vocabulary (after
case-folding and the explicit typo map) is reported as inconsistent.
"""

# --- Movie categories (doc.html 3.13 / 4.4, main.html preamble) ---------
CATEGORY_NAMES = {
    "Actn": "Action/violence", "Advt": "Adventure", "AvGa": "Avant-garde",
    "BioP": "Biographical", "Camp": "Camp", "Cart": "Cartoon",
    "CnR": "Cops & robbers", "Comd": "Comedy", "Disa": "Disaster",
    "Docu": "Documentary", "Dram": "Drama", "Epic": "Epic", "Faml": "Family",
    "Fant": "Fantasy", "Hist": "History", "Horr": "Horror", "Musc": "Musical",
    "Myst": "Mystery", "Noir": "Film noir", "Porn": "Adult", "Romt": "Romantic",
    "ScFi": "Science fiction", "Surl": "Surreal", "Susp": "Suspense/thriller",
    "West": "Western", "TV": "TV show",
}
# Explicit, documented corrections of misspelt category codes found in the
# data (detected by the inconsistency scan; each one is unambiguous).
CATEGORY_TYPOS = {
    "ctxx": None, "ctxxx": None, "txx": None,          # 'uncategorised' = missing
    "biop": "BioP", "biog": "BioP", "scifi": "ScFi", "scif": "ScFi",
    "s.f.": "ScFi", "sf": "ScFi", "surr": "Surl", "surl": "Surl",
    "muscl": "Musc", "muusc": "Musc", "ducu": "Docu", "dramd": "Dram",
    "comdx": "Comd", "cond": "Comd", "com": "Comd", "hor": "Horr",
    "adct": "Advt", "romt.": "Romt", "west1": "West", "crim": "CnR",
    "religous": None, "satire": "Comd", "f": None, "aan": None,
    # single-character slips found by the inconsistency scan (edit distance 1-2,
    # unambiguous); ambiguous tokens (e.g. 'CRomt', 'Sctn') are left unmapped
    "c0md": "Comd", "omd": "Comd", "dramn": "Dram", "drama": "Dram", "drm": "Dram",
    "d<ram": "Dram", "cmr": "CnR", "epicx": "Epic", "horr<": "Horr", "hhorr": "Horr",
    "duco": "Docu", "dicu": "Docu", "bio": "BioP", "biob": "BioP", "biopp": "BioP",
    "biopx": "BioP", "biopp.": "BioP", "mystp": "Myst", "porm": "Porn", "porb": "Porn",
    "ront": "Romt", "romtx": "Romt", "rom": "Romt", "axtn": "Actn", "act": "Actn",
    "sxfi": "ScFi", "disaster": "Disa", "dist": "Disa", "sati": "Comd",
    "avant": "AvGa", "garde": "AvGa", "sup": "Susp",
}

# --- Colour / film process codes (doc.html 3.14 / 4.5) -----------------
# Mapped to a coarse, analysable "colour type".
PROCESS_GROUP = {
    "bws": "silent_bw", "sbw": "silent_bw", "bnw": "bw", "bnwc": "bw",
    "cld": "colorized", "col": "color", "car": "color", "cart": "color",
    "tcol": "color", "deluxe": "color", "duart": "color", "ecol": "color",
    "mcol": "color", "mlab": "color", "movielab": "color", "wcol": "color",
    "acol": "color", "agcol": "color", "fcol": "color", "cfcol": "color",
    "cficol": "color", "tr": "color", "cs": "color", "trama": "color",
    "pan": "color", "vst": "color", "tv": "tv",
}
PROCESS_MISSING = {"prc", ""}

# --- Country / origin codes (doc.html 4.3) -----------------------------
COUNTRY_CODES = {
    "Am": "USA", "Br": "UK", "GB": "UK", "Fr": "France", "Ge": "Germany",
    "It": "Italy", "Ja": "Japan", "Ar": "Argentina", "Au": "Australia",
    "Be": "Belgium", "Bz": "Brazil", "Ca": "Canada", "Ch": "China",
    "Cz": "Czechoslovakia", "Da": "Denmark", "Gr": "Greece", "Du": "Netherlands",
    "Hu": "Hungary", "In": "India", "Ir": "Ireland", "Me": "Mexico",
    "Os": "Austria", "Pe": "Peru", "Ru": "Russia/USSR", "Sp": "Spain",
    "SA": "South Africa", "Yu": "Yugoslavia", "Zw": "Switzerland",
    "Sw": "Sweden", "Pl": "Poland", "Po": "Poland", "Pg": "Portugal",
    "Fi": "Finland", "No": "Norway", "Is": "Israel", "Eg": "Egypt",
    "Tu": "Turkey", "Ko": "Korea", "Tw": "Taiwan", "HK": "Hong Kong",
    "NZ": "New Zealand", "Cu": "Cuba", "Ph": "Philippines",
}
COUNTRY_MISSING = {"Un", "UN", "un", ""}

# --- Award codes (main.html preamble, awtypes.html) --------------------
# H...: stars in Halliwell's Film Guide (0-4), AA: Academy Award,
# AAN: Academy Award nomination, CG: Consumers Guide stars, RE: Roger Ebert
# stars, NFR: US National Film Registry, W50: "50 worst films" list, etc.
AWARD_MISSING = {"aw", ""}
