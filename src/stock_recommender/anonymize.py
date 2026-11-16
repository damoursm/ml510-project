"""Hide the companies and the dates of a news article from the LLM that scores it.

The LLM was trained on text written after most of the articles it scores, so it may know what happened next to a
company it recognizes, or after a date it reads. mask() replaces each scored ticker, and the names its company goes
by, with a placeholder (Company A, Company B...); the other companies it finds get the next placeholders; and every
date goes: "March 15, 2023" becomes [date], "in March" becomes in [month], 2023 becomes 20XX, and the dateline of a
press release (city, date, wire) is dropped. What refers to a time without dating it stays: last year, next
quarter, Q3, Monday. Tickers of one company (GOOG and GOOGL) share a placeholder. What stays recognizable: products,
people, places, and a company that neither the table below nor the article itself names as one.

The names come from a table of aliases, one row per ticker and alias (build_aliases, scripts/build_ticker_aliases.py):
- name: the name Polygon gives the ticker (scripts/download_ticker_names.py), with and without its legal ending:
  Ford Motor Company gives Ford Motor Company and Ford Motor;
- start, word, initials: that name's first words (Ford), its other words (Disney) and its initials (TSMC, P&G);
- seen: the capitalized words the news write just before the ticker in parentheses, "Coke (KO)";
- brand: a name neither gives, in BRANDS (Google for Alphabet);
- ticker: the ticker itself where it stands alone in the text ("AMD rose").
An alias is used only if the news of 2021-2025 confirm it (use()). Of the articles that contain it on its own (not
inside a longer alias of the same ticker), a large enough share must be tagged with the ticker: "American" fails
for AAL and "Solar" for FSLR, while "Ford" and "Delta" pass. And the articles tagged with the ticker must write it
with a capital: "Gold" fails for Barrick Gold, whose articles are about gold; a name its own articles write small
(lululemon, e.l.f. Beauty) is hidden in any case. A one-word alias that most articles write about something else
(Target, Dow, Apollo) is hidden unless it touches a word it forms another name with in the other articles: "Price
Target", "Dow Jones". An alias the news never contain is used.

A ticker is hidden wherever it is written as one: "NYSE: F", "$F", "Ford (F)", "TECK.B", in a list with other
companies ("Ford, F and GM"), before "stock", "shares" or "earnings call". Where it stands alone ("AMD rose") it is
hidden too, unless most articles that write it use it for something else (AI, IT, CEO, LNG): those stay where the
text uses the word, and in brackets where the words before spell it ("artificial intelligence (AI)"). A ticker of
one letter is hidden alone too, except where it is a letter: Plan B, Class C, Vitamin D, F-150, John F. Kennedy.

The article itself gives names the table lacks: the capitalized words just before a ticker in parentheses ("Apollo
(APO)", "Li Auto Inc. (NASDAQ: LI)"), the short names a press release defines ("Evergold" or the "Company") and the
first word of a full name it writes (Montero, for Montero Mining and Exploration) are hidden wherever the article
repeats them. That works for companies the article is not tagged with too: any ticker after a market ("NASDAQ: LI"),
any cashtag, a ticker in a list with the article's companies, and the aliases the table marks wide (a full name of
several words that the news do not write small, a one-word name that is rarely anything else, a ticker nearly always
used as one) are hidden in every article, whatever its day.

A ticker passes from one company to another (GOLD from Barrick Gold to Gold.com in 2025): an alias from a Polygon
name applies from SINCE_SLACK before the first date Polygon gave that name, or before the news first wrote it
before the ticker in parentheses (Polygon dates some names late, or misspells them for years), and stays after the
name changes (Facebook). Two tickers are one company (same_company) when they share a name at the article's date
and either look like two classes of shares (GOOG and GOOGL, BRK.A and BRK.B) or one took the other's name, legal
form included (SQ and XYZ are both Block, Inc.).
"""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import re
import unicodedata
import pandas as pd

TOKEN = re.compile(r"[^\W_]+(?:[.&'’][^\W_]+)*")  # Words: AT&T, Amazon.com and Lowe's are one each.
MAX_WORDS = 6  # Words in the longest alias.
# The share of the articles containing an alias on its own that must be tagged with its ticker, for an alias of one
# word by kind, and for a longer one. Banks are named in many articles that are not tagged with them (Goldman Sachs
# analysts), so a share far below one still tells a name from a common word. A name of several words needs none.
SHARES = {"name": 0.02, "seen": 0.05, "brand": 0.05, "start": 0.35, "ticker": 0.5, "word": 0.7, "initials": 0.5}
PHRASE_SHARE = 0.15
NAME_LIFT = 5  # A one-word name (Target) must be this many times more frequent in its ticker's articles than elsewhere.
CAPITAL_SHARE = 0.7  # Of the ticker's articles that contain the alias's words, the share that give them a capital.
MIN_SMALL, SMALL_SHARE = 2, 0.5  # A name written small is its ticker's when its articles hold this share of them.
MIN_APART = 3  # Other articles that must write a word beside a one-word alias for the two to be another name.
MIN_WORD = 10  # Other articles that must write a ticker for it to be a word too.
# Places that start companies' names: alone, they are the place.
PLACES = {"taiwan", "china", "chinese", "japan", "japanese", "korea", "korean", "india", "indian", "canada", "canadian",
          "mexico", "brazil", "brazilian", "chile", "peru", "argentina", "colombia", "israel", "turkey", "egypt",
          "germany", "german", "france", "french", "italy", "italian", "spain", "spanish", "britain", "british",
          "england", "london", "ireland", "irish", "scotland", "europe", "european", "asia", "asian", "africa",
          "african", "australia", "australian", "singapore", "hong", "kong", "swiss", "dutch", "nordic", "texas",
          "california", "florida", "ohio", "virginia", "carolina", "georgia", "nevada", "arizona", "colorado",
          "oregon", "washington", "boston", "chicago", "houston", "dallas", "atlanta", "denver", "miami", "york",
          "jersey", "pacific", "atlantic", "rio", "paris", "tokyo", "shanghai", "beijing", "dubai", "qatar",
          "saudi", "russia", "russian", "ukraine", "vietnam", "thailand", "malaysia", "indonesia", "philippines",
          "manhattan", "brooklyn", "hawaii", "alaska", "montana", "utah", "idaho", "iowa", "kansas", "kentucky",
          "tennessee", "alabama", "louisiana", "oklahoma", "michigan", "indiana", "wisconsin", "minnesota",
          "missouri", "maryland", "delaware", "vermont", "maine", "pennsylvania", "illinois", "silicon", "wall", "southwest", "northeast", "northwest", "southeast", "midwest"}
NAME_LINKS = {"de", "of", "del", "da", "do", "la", "du", "van", "von", "der", "di", "dos", "das"}  # Rio de Janeiro.
PLACE_STARTS = {"new", "north", "south", "east", "west", "san", "santa", "los", "las", "saint", "st", "fort", "port",
                "mount", "lake", "upper", "lower", "greater", "great", "little", "old", "le", "el"}
# First names: a company's one-word name after one is a surname (Harrison Ford, Mike Matson), and one that is a
# first name before a surname is a person (Lindsay Fitzgerald, Devon Sawa).
FIRST_NAMES = set("""aaron adam alan albert alex alexander alice amanda amy andrew andy angela anna anne anthony
    barbara ben benjamin beth betty bill billy bob bobby brad brandon brian bruce carl carlos carol caroline catherine
    charles charlie chris christian christina christine christopher chuck cindy claire craig dan daniel danny dave
    david dean deborah dennis devon diana diane don donald donna doug douglas ed eddie edward elizabeth ellen emily
    emma eric erik eva frank fred gary george gerald grace greg gregory harrison harry heather helen henry ian jack
    jacob james jamie jane janet jason jeff jeffrey jennifer jeremy jerry jessica jill jim jimmy joan joe joel john
    johnny jon jonathan jordan joseph josh joshua joyce juan judy julia julie justin karen kate katherine kathleen
    kathy keith kelly ken kenneth kevin kim kyle larry laura lauren lawrence lee linda lindsay lisa liz louis lucy
    luke margaret maria mark martha martin mary matt matthew megan melissa michael michelle mike nancy nathan neil
    nicholas nick nicole noah pamela pat patricia patrick paul peter phil philip rachel ralph randy ray raymond
    rebecca richard rick robert roger ron ronald rose russell ruth ryan sam samantha samuel sandra sara sarah scott
    sean sharon shirley simon stephanie stephen steve steven susan ted terry thomas tim timothy todd tom tony tyler
    victoria vincent walter wayne william willie""".split())
MIN_PHRASE, PHRASE_RATE = 5, 0.02  # Articles, and their share of those that write the alias small, for a phrase.
# An alias is wide (hidden in the articles not tagged with its ticker too) when it is rarely anything but the
# company: a ticker nearly always tagged; a name of several words that is not the end of other companies' names
# (Bancorp Inc); a one-word name that the news rarely write small. Below PLAIN_SHARE, a one-word name is hidden
# only where no capitalized word touches it (New Brunswick).
WIDE_TICKER, WIDE_PHRASE, WIDE_WORD, PLAIN_SHARE, WIDE_SMALL, WIDE_INSIDE = 0.95, 0.02, 0.15, 0.5, 0.1, 0.5
ROLES = {"ceo", "cfo", "coo", "cto", "chief", "chairman", "president", "stock", "stocks", "shares", "inc", "corp"}
MIN_SEEN = 3  # Times a name must be written before its ticker in parentheses, and
MIN_SEEN_SHARE = 0.05  # its share of that ticker's such mentions, to be an alias;
OWN_SEEN_SHARE = 0.2  # this share if it ends with one of the name's other words ("Why Disney" against "U.S. Steel").
BRANDS = {"GOOGL": ["google"], "GOOG": ["google"]}  # Names the news use that neither Polygon's name nor "(GOOGL)" give.
SINCE_SLACK = pd.Timedelta(days=100)  # Names are read once a quarter, and the news use a new name before it is one.
ALIAS_COLUMNS = ["ticker", "alias", "kind", "articles", "tagged", "alone", "lowercase", "use", "since", "any_case",
                 "wide", "apart"]
ALWAYS, ALONE, NEVER = "always", "alone", "never"  # An alias's use: where mask() hides it.
CAPITALS = "capitals"  # How initials are hidden: where the text writes them in capitals.
RANK = {"name": 0, "start": 1, "brand": 2, "seen": 3, "word": 4, "initials": 5}  # Whose alias it is, of two tickers.
NAMES = ("name", "start", "brand", "seen")  # The kinds of alias that name the company.
# Markets a ticker is written after: "NYSE: F", "(TSX-V: ABC)", "OTC Pink: XYZ". The short ones only in capitals.
_MARKETS = (r"NYSE(?:\s?American|\s?Arca|\s?MKT|\s?Euronext)?|NASDAQ(?:/(?:GS|GM|CM)|\s?(?:GS|GM|CM"
            r"|Global\sSelect(?:\sMarket)?"
            r"|Global\sMarket|Capital\sMarket|Stockholm|Helsinki|Copenhagen|First\sNorth(?:\sGrowth\sMarket)?"
            r"|Baltic|Iceland))?|AMEX|OTC(?:\s?(?:QB|QX|MKTS|BB|PK|Markets|Pink\sOpen\sMarket|Pink\sSheets|Pink"
            r"|Expert\sMarket))?|Pink\sSheets|TSX(?:\s?[-.]?\s?V(?:enture)?(?:\sExchange)?)?|CNSX|XETRA|HKEX|SEHK"
            r"|KOSDAQ|KRX|TASE|OTCMKTS|Euronext(?:\s(?:Paris|Amsterdam|Brussels|Lisbon|Dublin|Oslo|Milan|Growth"
            r"(?:\s(?:Paris|Oslo|Milan))?))?|Oslo\sB[øo]rs|Cboe(?:\sCanada|\sBZX)?|BATS")
_SHORT_MARKETS = (r"ASX|AIM|CSE|NEO|SIX|SWX|BME|BIT|OSE|STO|HEL|CPH|EPA|ETR|AMS|HKG|SHA|SHE|TYO|TSE|NSE|BSE|IDX|JSE|NZX"
                  r"|SGX|BMV|LSE|FSE|FRA|FWB|OMX|B3|SSE|Frankfurt|Stuttgart|Xetra|Börse\sFrankfurt")
EXCHANGE = rf"(?:(?i:{_MARKETS})|{_SHORT_MARKETS})"
_ABROAD = (r"(?:TSX(?:\s?[-.]?\s?V(?:enture)?)?|CSE|CNSX|NEO|ASX|LSE|AIM|(?i:Euronext)(?:\s[A-Z]\w+)?|Oslo(?:\sB[øo]rs)?"
           r"|OSE|XETRA|FSE|FRA|FWB|Frankfurt|SIX|SWX|BMV|HKEX|SEHK|KOSDAQ|KRX|TASE|NSE|BSE|JSE|NZX|SGX|OMX"
           r"|(?i:Nasdaq)\s(?:Stockholm|Helsinki|Copenhagen|Iceland|Baltic|First\sNorth)|B3|EPA|ETR|AMS|BIT|BME"
           r"|TYO|TSE|STO|HEL|CPH|HKG)")
_HOME = (r"(?i:NYSE(?:\s?American|\s?Arca|\s?MKT)?|NASDAQ(?:\s?(?:GS|GM|CM)|/(?:GS|GM|CM))?|AMEX|OTC\w*"
         r"|Cboe(?:\sBZX)?|BATS)")
_FOREIGN = lru_cache(maxsize=4096)(lambda ticker: re.compile(
    rf"(?<![\w.]){_ABROAD}\s*[:;–-]\s*[“\"]?{re.escape(ticker)}(?![\w])"))
_AT_HOME = lru_cache(maxsize=4096)(lambda ticker: re.compile(
    rf"(?<![\w.]){_HOME}(?!\s(?:Stockholm|Helsinki|Copenhagen|Iceland|Baltic|First))(?:\s*[/&,]\s*{EXCHANGE})*"
    rf"\s*[:;–-]\s*{re.escape(ticker)}(?![\w])|\${re.escape(ticker)}(?![\w])"))
# AAPL, BRK.B, 4WE, FEMSAUBD; a code of digits (HKEX: 1179) only where a bracket or a list ends it.
SYMBOL = (r"(?:[A-Z][A-Z0-9]{0,7}(?:[.\-][A-Z]{1,2})?|\d[A-Z0-9]{0,4}[A-Z][A-Z0-9]{0,3}"
          r"|\d{1,6}(?=\s*[)\],;/(]|\s+(?:and|or)\b))")
AFTER_MARKET = r"(?:\s*[:;]\s*|\s+[-–]\s+)"  # "NYSE: F", "Euronext Paris : IPH", "NYSE - F", "(NASDAQ; TSLA)".
# "NYSE: F", "NYSE/TSX: ABX", "Nasdaq: HCDI, HCDIP": the tickers written after one or several markets.
MARKETS = rf"{EXCHANGE}(?:\s*(?:[/&,]|(?i:and|or)\b)\s*{EXCHANGE})*{AFTER_MARKET}"  # "NYSE, NASDAQ: VZ".
LISTED = re.compile(rf"(?<![\w.]){EXCHANGE}(?:\s*(?:[/&,]|(?i:and|or)\b)\s*{EXCHANGE})*{AFTER_MARKET}"
                    rf"({SYMBOL}(?:\s*[,/]\s*{SYMBOL}(?![\w.]|\s*:))*)(?![\w])")
CASHTAG = re.compile(r"\$([A-Z]{1,6}(?:\.[A-Z])?)(?![\w])")
CURRENCIES = {"US", "USD", "CAD", "CDN", "AUD", "NZD", "HKD", "SGD", "MXN", "EUR", "GBP", "M", "MM", "B", "BN", "K"}
# A security's codes name its company: "(CUSIP: 38502HAA3 / G4066TAA0; ISIN: US38502HAA32)", "ID code
# DK0061027356" (an ISIN: two letters, nine signs, a digit), "002475.SZ".
CODES = re.compile(r"\b(?:WKN|ISIN|CUSIP|SEDOL)\b\s*:?\s*[A-Z0-9]{6,12}\b(?:\s*/\s*[A-Z0-9]{6,12}\b)*"
                   r"|\b[A-Z]{2}(?=(?:[A-Z]*\d){3})[A-Z0-9]{9}\d\b|\b\d{4,6}\.(?:SZ|SS|SH|HK|T|KS|KQ|TW|TWO)\b")
# A ticker with its market's suffix: SHOP.TO, BHP.AX, VOD.L.
SUFFIXED = re.compile(r"\b([A-Z][A-Z0-9]{1,5})\.(?:TO|V|CN|NE|AX|L|PA|DE|F|MI|AS|BR|ST|OL|CO|HE|SW|KS|KQ|TW|SI|NS"
                      r"|BO|MX|SA|HK)\b(?![\w]|\.\w)")
# "under the ticker symbol “BTTX”", "its trading symbol XYZ", "ticker: XYZ"; not a headline all in capitals.
NAMED_SYMBOL = re.compile(r"(\b(?:(?:[Tt]icker|[Tt]rading|[Ss]tock)\s+[Ss]ymbols?|[Ss]ymbols?|[Tt]icker)\s*"
                          r"(?:of\s+|is\s+|will\s+be\s+|:\s*)?[“\"„«'‘]?)([A-Z]{2,6}(?:\.[A-Z])?)(?![\w])")
# Addresses carry the company's name or ticker: ir.lxp.com, www.nninc.com, @TroyTeslike, #tesla.
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
LINK = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"“”)\]]*[^\s<>\"“”)\].,;:!?]|\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\."
                  r"(?:com|net|org|io|ai|gov|edu|ca|uk|eu|info|biz|tv|app|xyz|fr|cn|jp|gg|de|es|nl|se|dk|fi|ch|au"
                  r"|nz|sg|hk|kr|br|mx|ly|co)\b"
                  r"(?:/[^\s<>\"“”)\]]*[^\s<>\"“”)\].,;:!?])?")
HANDLE = re.compile(r"(?<![\w.&])[@#](?=[A-Za-z_])\w{2,30}")
SIGNS = re.compile("[®™©℠​]")  # Trademark signs stand between a name and what follows it.
SPACES = str.maketrans({" ": " ", " ": " ", " ": " ", " ": " "})
# Words that start a sentence: after "Ford Motor Co." they tell that the period ends one.
SENTENCE = (r"(?:The|This|That|It|Its|In|A|An|He|She|They|We|Our|On|At|As|For|But|And|However|Meanwhile|Shares|Here"
            r"|There|These|Those|With|While|When|After|If|To)")
# First words too common to name a company on their own: First Quantum is not every "First".
COMMON = {"first", "american", "united", "national", "general", "international", "global", "advanced", "applied",
          "north", "south", "east", "west", "northern", "southern", "eastern", "western", "central", "pacific",
          "atlantic", "golden", "silver", "blue", "green", "red", "black", "white", "royal", "great", "grand",
          "premier", "standard", "universal", "digital", "smart", "clean", "pure", "bright", "energy", "capital",
          "bank", "banco", "banque", "community", "federal", "home", "liberty", "pioneer", "summit", "alpha",
          "omega", "delta", "prime", "core", "next", "one", "open", "true", "star", "sun", "city", "state",
          "world", "group", "holdings", "resources", "mining", "gold", "oil", "gas", "power", "health", "life",
          "medical", "bio", "tech", "technology", "technologies", "financial", "trust", "fund", "old", "big"}
LEGAL = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "llc", "lp", "l.p",
         "nv", "n.v", "sa", "s.a", "se", "ag", "ab", "asa", "spa", "s.p.a", "bhd", "pte", "pty", "gmbh", "kgaa", "cos",
         "companies", "aktiengesellschaft", "n.a", "the", "de", "i", "ii", "iii", "iv", "new", "reit"}
SAME_LEGAL = {"inc": "incorporated", "incorporated": "inc", "corp": "corporation", "corporation": "corp",
              "co": "company", "company": "co", "ltd": "limited", "limited": "ltd"}
LINKS = {"of", "and", "the", "de", "la", "for", "in", "at", "on", "du", "del", "van", "von", "a", "an", "to"}
# Words a headline writes with a capital before a name: "Why Montero (MON) Rose", "Is Bonanza (BCEI) A Buy".
HEADLINE = {"why", "is", "are", "was", "were", "will", "can", "could", "should", "would", "does", "did", "do", "has",
            "have", "had", "how", "what", "when", "where", "here's", "heres", "here", "but", "or", "as", "with",
            "from", "by", "buy", "sell", "hold", "stock", "stocks", "shares", "trending", "top", "best", "about",
            "vs", "versus", "than", "after", "before", "while", "if", "that", "this", "these", "those", "its", "it",
            "be", "been", "not", "no", "more", "most", "up", "down", "over", "under", "into", "out", "off", "amid",
            "against", "between", "during", "may", "might", "must", "just", "now", "still", "also", "then", "so",
            "too", "very", "like", "including", "says", "said", "see", "watch", "meet", "new", "analyst", "analysts",
            "shareholder", "shareholders", "investor", "investors", "alert", "update", "breaking", "report"}
# Words that end a headline's phrase, not a company's name: "Before April 23 Deadline (NYSE: ATKR)".
NOT_NAMES = {"deadline", "alert", "alerts", "update", "updates", "reminder", "notice", "lawsuit", "action", "investors",
             "shareholders", "losses", "loss", "gains", "suffered", "results", "report", "reports", "earnings",
             "stock", "stocks", "shares", "interest",
             "news", "today", "week", "rating", "ratings", "analysis", "conference", "call", "transcript", "says",
             "quarter", "year", "estimates", "guidance", "outlook", "dividend", "deal", "offering", "price", "target",
             "sale", "buy", "sell", "hold", "upgrade", "downgrade", "review", "preview", "recap", "roundup"}
_MONTH_WORDS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
                "dec"}
# Words a headline writes with capitals around a company's name: "Citigroup Maintains", "Why NVDA Is Soaring".
TITLE_WORDS = {"maintains", "reiterates", "initiates", "upgrades", "downgrades", "lowers", "raises", "boosts", "cuts",
               "adjusts", "sets", "assumes", "resumes", "keeps", "earnings", "stock", "stocks", "shares", "share",
               "rally", "rallies", "soars", "jumps", "falls", "drops", "slides", "surges", "gains", "rises", "rose",
               "fell", "beats", "misses", "tops", "lags", "results", "revenue", "sales", "guidance", "outlook",
               "dividend", "deal", "rating", "target", "price", "report", "reports", "announces", "unveils",
               "launches", "plans", "posts", "sees", "gets", "wins", "faces", "nears", "hits", "plunges", "tumbles",
               "climbs", "sinks", "slumps", "rebounds", "extends", "expands", "acquires", "buys", "sells", "signs",
               "partners", "invests", "stake", "options", "short", "q1", "q2", "q3", "q4", "inc", "corp", "ltd",
               "plc", "co", "and", "vs", "versus", "or", "says", "said", "to", "of", "on", "in", "at", "for", "from",
               "with", "by", "after", "as", "amid", "ahead", "over", "into", "up", "down", "higher", "lower", "big",
               "new", "top", "this", "that", "here", "here's", "what", "how", "why", "is", "are", "will", "could",
               "should", "would", "can", "may", "might", "must", "has", "have", "had", "be", "been", "being", "its"}
TITLE = HEADLINE | ROLES | TITLE_WORDS
PRONOUNS = {"you", "your", "i", "we", "our", "they", "their", "he", "she", "his", "her", "my", "me", "us", "them"}
# The words of a headline that a company's name does not hold: a name the table lacks starts after the last.
# Words that end a company's name after its first ones: "Deere & Company", "Closed-End Opportunity Fund, Inc.".
CORPORATE = {"company", "companies", "co", "corp", "corporation", "inc", "incorporated", "group", "holdings", "holding",
             "ltd", "limited", "plc", "llc", "lp", "sa", "ag", "nv", "se", "bancorp", "bancshares", "trust", "fund",
             "etf", "partners", "industries", "technologies", "brands", "electric", "international", "financial",
             "capital", "resources", "energy", "systems", "solutions", "services", "pharmaceuticals", "therapeutics",
             "bank", "reit", "income", "portfolio"}
_HEADLINE_ONLY = (TITLE | HEADLINE | PRONOUNS) - LINKS - LEGAL - CORPORATE - {"and", "or", "new", "big", "top"}
# Polygon writes over-the-counter names short: AKER SOLTNS ASA, NATL AUSTRALIA BK, TENCENT HLDGS.
ABBREVIATIONS = {"hldgs": "Holdings", "hldg": "Holding", "techs": "Technologies", "bk": "Bank", "bks": "Banks",
                 "grp": "Group", "gp": "Group", "intl": "International", "svcs": "Services", "sys": "Systems",
                 "pharm": "Pharmaceuticals", "finl": "Financial", "natl": "National", "mgmt": "Management",
                 "inds": "Industries", "engy": "Energy", "res": "Resources", "soltns": "Solutions", "solns": "Solutions",
                 "commun": "Communications", "elec": "Electric", "mfg": "Manufacturing", "dev": "Development",
                 "invt": "Investment", "assoc": "Associates", "amer": "American", "hlth": "Health", "ins": "Insurance",
                 "props": "Properties", "ptnrs": "Partners", "autombl": "Automobil", "minl": "Mineral",
                 "minls": "Minerals", "expl": "Exploration", "petrol": "Petroleum", "chem": "Chemical",
                 "chems": "Chemicals", "inv": "Investment"}
# What a press release calls itself in a defined term: (the "Company"), ("we", "us" or "our").
GENERIC = {"company", "corporation", "corp", "group", "fund", "trust", "bank", "partnership", "issuer", "registrant",
           "we", "us", "our", "parent", "firm", "reit", "offeror", "purchaser", "borrower", "buyer", "seller",
           "acquirer", "holdings", "shares", "stock", "units", "notes", "warrants", "securities", "offering",
           "etf", "etfs", "etn", "index", "spdr", "adr", "ads", "portfolio"}
# Where the description of the security starts in Polygon's name: "Alphabet Inc. | Class A Common Stock".
SECURITY = re.compile(
    r"\b(?:Class\s+[A-Z]\b|Series\s+[A-Z0-9-]+\b|(?:New\s+)?(?:Common|Ordinary|Capital|Preferred|Preference|Voting"
    r"|Subordinate|Restricted|Limited\s+Voting|Non-Voting)\s+(?:Stock|Shares?|Voting)\b|American\s+Deposit[ao]ry\b"
    r"|Deposit[ao]ry\s+(?:Shares?|Receipts?|Units?)\b|ADSs?\b|ADRs?\b|Shares\s+of\b|each\s+representing\b"
    r"|\d+(?:\.\d+)?\s?%|due\s+20\d\d|Warrants?\b|Units?\s*(?:$|,|each\b|consisting\b)|Rights?\s*$|\((?:DE|MD|NV)\)"
    # Over-the-counter names: TENCENT HLDGS UNSP/ADR, DANONE S/ADR, AAC TECHS HLDGS INC ORD.
    r"|(?:\b(?:UNSP(?:ON(?:S(?:ORED)?)?)?|SPON(?:S(?:ORED)?)?)[\s/.-]+|(?-i:\bS/))AD[RS]S?\b|\bORD\b|\bSHS\b|\bAKT\b"
    r"|\bNPV\b)", re.IGNORECASE)

# Dates. A month and a day or a year is a [date], a month alone a [month], a year 20XX.
_FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
         "November", "December"]
_SHORT = ["Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sept", "Sep", "Oct", "Nov", "Dec"]
_cased = lambda names: "|".join(names + [name.upper() for name in names])  # noqa: E731
# A month as a date writes it: its name, or its first letters with or without a period.
_MONTH = rf"(?:(?:{_cased(_FULL)})\b|(?:{_cased(_SHORT)})\b\.?)"
_NOT_MAY = rf"(?:(?:{_cased([name for name in _FULL if name != 'May'])})\b|(?:{_cased(_SHORT)})\b\.?)"
_DAY = r"(?:[12]\d|3[01]|0?[1-9])(?:st|nd|rd|th)?"
_YEAR = r"(?:19|20)\d\d"
_WHEN = (r"(?i:in|of|since|until|till|through|during|late|early|mid|by|from|last|next|ended|ending|before|after"
         r"|between)")
# "March 15, 2023", "Sept. 9-11, 2024", "June 26 and 27", "Dec.14, 2021"; not "May 5%" or "March 4.5".
MONTH_DAY = re.compile(rf"\b{_MONTH}(?:(?<=\.)\s*|\s+){_DAY}\b(?!\s?%|\.\d|,\d(?!\d\d\d\b)"
                       rf"|\s+(?i:percent|million|billion)\b)"
                       rf"(?:\s*(?:-|–|—|to|through|and|&)\s*{_DAY}\b(?!\s?%|[.,]\d))?(?:\s*,?\s*{_YEAR}\b)?")
# "15 March 2023", "31st of March"; "3 May" only before a year or the end of its phrase ("These 3 May Soar").
DAY_MONTH = re.compile(rf"(?<![\w/.,$€£-])(?<!Phase )(?<!phase )(?<!PHASE ){_DAY}\s+(?:of\s+)?(?:{_NOT_MAY}(?!\w)|(?:May|MAY)\b(?=\s*,?\s*{_YEAR}\b"
                       rf"|\s*[.,;:)|–—-]|\s*$))(?:\s*,?\s*{_YEAR}\b)?")
MONTH_YEAR = re.compile(rf"\b{_MONTH}\s*,?\s*(?:of\s+)?{_YEAR}\b")  # "March 2023", "Sept. of 2021".
MONTH_COMMA_DAY = re.compile(rf"\b{_MONTH},\s+{_DAY},?\s+{_YEAR}\b")  # "as of March, 31, 2023".
DAY_YEAR = re.compile(rf"\b(?:[12]\d|3[01]|0?[1-9])(?:st|nd|rd|th),?\s+{_YEAR}\b")  # "to the 11th, 2024".
SPACED_DATE = re.compile(r"(?<![\d,.])(?:[0-2]\d|3[01]) (?:0[1-9]|1[0-2]) (?:(?:19|20)\d\d|\d\d)(?![\d,.])")  # 30 06 23.
WEEK = re.compile(r"\b(?i:week|semaine|uge|vecka|uke|woche|viikko|semana|settimana)\s+(?:nr\.?\s*|no\.?\s*)?"
                  r"(?:[1-4]\d|5[0-3]|0?[1-9])\b(?!\s*(?:%|days|weeks))")  # "Semaine 23", "uge 22": weeks of a year.
# "3/15/2023", "10.9.2025", "2021-06-04"; not a version (v1.2.21) or an address (10.9.20.1).
NUMERIC_DATE = re.compile(r"(?<![\w/.-])(?:\d{1,2}([/.-])\d{1,2}\1(?:19|20)?\d\d|(?:19|20)\d\d([/.-])\d{1,2}\2\d{1,2})"
                          r"(?![\w/-]|\.\d)")
SHORT_DATE = re.compile(r"\b((?i:on|dated|since|until|through)\s+)(?:0?[1-9]|1[0-2])/(?:[12]\d|3[01]|0?[1-9])\b(?![/\d])")
_OTHER_MONTHS = (  # French, German, Spanish, Italian, Dutch, Portuguese, Nordic and Finnish, where they differ.
    "janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre"
    "|janv|févr|fevr|avr|juil|déc|januar|februar|märz|maerz|april|juni|juli|august|september|oktober|november"
    "|dezember|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
    "|gennaio|febbraio|aprile|maggio|giugno|luglio|settembre|ottobre|dicembre|januari|februari|maart|mei|augustus"
    "|maj|augusti|marts|desember|janeiro|fevereiro|março|maio|junho|julho|setembro|outubro|novembro|dezembro"
    "|tammikuuta|helmikuuta|maaliskuuta|huhtikuuta|toukokuuta|kesäkuuta|heinäkuuta|elokuuta|syyskuuta|lokakuuta"
    "|marraskuuta|joulukuuta")
OTHER_DATE = re.compile(rf"(?<![\w/.,$€£-])(?:1er|\d{{1,2}})\.?\s+(?:de\s+|d[’'])?(?i:{_OTHER_MONTHS})\b\.?"
                        rf"(?:\s+(?:de\s+)?{_YEAR}\b)?|\b(?i:{_OTHER_MONTHS})\b\.?\s+(?:de\s+)?{_YEAR}\b")
CJK_DATE = re.compile(rf"(?:{_YEAR}\s*年\s*)?\d{{1,2}}\s*月\s*\d{{1,2}}\s*日|{_YEAR}\s*年(?:\s*\d{{1,2}}\s*月)?")
MONTH_ALONE = re.compile(rf"\b(?:{_cased([name for name in _FULL if name != 'May'])})\b"
                         r"(?!\s+(?:Higher|Lower|On|Toward|Towards|Forward|Ahead|Into|Back)\b)")  # Shares March Higher.
# "May" is a month after a word of time, or before one a month goes with; not in "This May Be".
MAY = re.compile(rf"\b({_WHEN}[\s-]+)(?:May|MAY)\b(?!\s+(?i:be|not|have|also|still|soon|well|even|never|just)\b)"
                 r"|\b(?:May|MAY)\b(?=[\s-](?:quarter|meeting|report|sales|data|deliveries|jobs|contract|futures"
                 r"|Quarter|Meeting|Sales|Data|Deliveries|Jobs|CPI|PPI|Retail|Payrolls|Inflation|Production"
                 r"|Traffic|Shipments|Housing|Futures|Contract|payrolls|retail|inflation|production|traffic"
                 r"|shipments|housing|cpi)\b"
                 r"|\s*(?:,|and|or|through|to|-|–|/)\s*\[(?:month|date)\])")
SHORT_MONTH = re.compile(rf"\b({_WHEN}[\s-]+)(?:{_cased(_SHORT)})\b\.?(?![\w-])")
THE_MONTH = re.compile(rf"\b((?i:the)\s+)(?:{_cased(_SHORT)})\.(?=\s)")  # "the Mar. CPI".
RANGE_END = re.compile(rf"(\[date\]\s*(?:to|through|until|and|-|–|—)\s*)(?:the\s+)?{_DAY}\b(?:,?\s*{_YEAR}\b)?")
_ORDINALS = ("first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth"
             "|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty-\\w+|thirtieth"
             "|thirty-first")
HOLIDAY = re.compile(rf"\b(?:the\s+)?(?i:{_ORDINALS})\s+of\s+\[month\]|\[month\]\s+(?i:{_ORDINALS})\b"
                     r"(?!\s*(?i:quarter|half|time|round|day of))")  # "Fourth of July", "July Fourth".
_NOT_COUNTED = (r"(?!\s?(?i:shares|units|employees|stores|points|people|workers|customers|patients|locations|sites"
                r"|miles|km|kg|tons|tonnes|mw|gw|kw|mah|bps|barrels|bbl|oz|ounces|acres|hectares|feet|ft|meters"
                r"|metres|square|sq)\b)")
YEAR_RANGE = re.compile(rf"(?<![\d$€£#])(?<!\d[.,])({_YEAR})\s?[-–/]\s?(\d\d)(?![\w%]|[-/.,]\d)")  # 2024-25, 2023/24.
# A year: 1900 to 2099, not an amount ($2023, 2,023, 2024%, 2030.5), an index (Russell 2000) or a count.
YEAR = re.compile(rf"(?<![\d$€£#])(?<!\d[.,])(?<!Russell )({_YEAR})(?![\d%]|[.,]\d|®){_NOT_COUNTED}")
OLD_YEAR = re.compile(r"\b((?i:in|since|from|until|before|after|circa|founded\sin|established\sin|back\sto)\s+)"
                      r"1[5-8]\d\d\b(?![\d%]|[.,]\d)")
# FY23, Q3'23, 1Q23, F3Q22: a year from 2015 to 2035, so that H100 and AEC-Q200 stay.
PERIOD_YEAR = re.compile(r"\b(F?Q[1-4]|F?[1-4]Q|H[12]|[12]H|FY|CY)(?:\s?['’]|-)?(?:1[5-9]|2\d|3[0-5])(?=[EeA]?\b)")
SHORT_YEAR = re.compile(r"(?<![0-9])['’]\d{2}(?=s?\b)")  # '21, Hikes'21, the '90s.
# "40th Annual Conference", "67th American Society of Hematology Annual Meeting", "21st US Government Shutdown":
# an edition counts the years.
EDITION = re.compile(r"(?<![\d,.])\b(?:\d{1,3}(?i:st|nd|rd|th)\s+(?=(?i:annual)\b)|\d{2,3}(?i:st|nd|rd|th)\s+"
                     r"(?=[A-Z]))")
# The start of a press release: "DUNMORE, Pa., April 21, 2023 (GLOBE NEWSWIRE) -- ", "NEW YORK, April 21, 2023
# /PRNewswire/ -- ", "SAN FRANCISCO--(BUSINESS WIRE)--", "CITY, ST / ACCESSWIRE / April 21, 2023 / ".
_WIRES = (r"GLOBE\s?NEWSWIRE|BUSINESS\s?WIRE|PR\s?Newswire(?:-[\w-]+)?|ACCESS\s?WIRE|ACCESS\s?Newswire"
          r"|Newsfile\sCorp\.?|NewMediaWire|TheNewswire|Marketwired?|EQS[\s-]News|CNW(?:\sGroup)?|PRWEB|InvestorWire")
DATELINE = re.compile(rf"^\s*[^\n|]{{0,140}}?(?:[(/]\s*(?i:{_WIRES})[^\n|/)]{{0,40}}[)/]\s*(?:--+|—|–|-)+"
                      rf"(?:\s*[A-Z][a-z]{{2,8}}\.?\s+\d{{1,2}},\s+{_YEAR}\s*--+)?"
                      rf"(?:\s*via\s+[A-Z][\w-]+(?:\s[A-Z][\w-]+)?\s*[–—-]+)?"
                      rf"|/\s*(?i:ACCESS\s?WIRE|ACCESS\s?Newswire)\s*/[^\n|/]{{0,40}}/)\s*")
# A dateline further in, once its date is gone: "... -- MIAMI BEACH, Fla., [date] (GLOBE NEWSWIRE) -- ".
_CITY = r"[A-Z][\w.'’-]*(?:[ \t]+[A-Z][\w.'’-]*){0,3}"
INNER_DATELINE = re.compile(rf"(?<![\w,]){_CITY}(?:,[ \t]*{_CITY}){{0,3}},?[ \t]*\[date\][ \t]*[(/][ \t]*"
                            rf"(?i:{_WIRES})[^\n|/)]{{0,40}}[)/][ \t]*(?:--+|—|–)?[ \t]*"
                            rf"|^[ \t]*{_CITY}(?:,[ \t]*{_CITY}){{0,3}},?[ \t]*(?:[–—-]+[ \t]*)?\[date\][ \t]*[–—-]+[ \t]*")

_MARK = ""  # A placeholder's place in a text being masked: _MARK and a private-use character for its index.
_FIRST, _LAST = 0xE100, 0xF7FF
_INDEX = "[-]"
_SPLIT = ""  # Between the title and the summary, masked as one text.
_PRIVATE = re.compile("[-]")
_ANY = f"{_MARK}{_INDEX}"
# "Ford Motor Co.", "Eli Lilly and Company", "BP p.l.c.", "Baidu, inc", "Acquisition Corp. II", "XPeng Inc. Sponsored
# ADR": what follows the placeholder of a name is still the name. The period stays before a new sentence.
_LEGAL_AFTER = re.compile(
    rf"({_ANY})(?:,?[ \t]+(?:(?:&|and)[ \t]+)?(?:(?i:inc|incorporated|corp|corporation|ltd|limited|llc|plc|p\.l\.c)"
    rf"|Co|Company|Companies|Holdings?|Hldgs|Group|Grp|Gr|Bancorp|Bancshares|Solns|Industries|Technologies|Brands"
    rf"|Electric|Sa|S\.?A\.?B\.?(?:[ \t]+de[ \t]+C\.?V)?"
    rf"|S\.?A\.?[ \t]+de[ \t]+C\.?V|L\.?P|N\.?V|S\.?A|SE|AG|AB|ASA|A/S|Oyj|S\.p\.A|SpA|GmbH|KGaA)\b"
    rf"(?:\.(?![ \t]+{SENTENCE}\b))?"
    rf"(?![-\w]))*(?:[ \t]+(?:II|III|IV|VI?)\b(?![-\w]))?(?:\.?,?[ \t]+(?:Sponsored|Unsponsored)?[ \t]*(?:ADR|ADS)\b)?"
    rf"(?:[ \t]*\(publ\))?")
# "Ford (F)", "Ford's (F)", "(F; FpB)", "Ford Ford" and "(F) (F)" are one mention; on one line: "About Comcast" and
# the "Comcast is..." of the next line are two.
_REPEATED = re.compile(rf"({_ANY})(['’]s)?(?:[ \t]*[(\[][ \t]*\1(?:[ \t]*[,;/][ \t]*\1)*[ \t]*[)\]]"
                       rf"|[ \t]*[,;/][ \t]*\1(?=[ \t]*[)\],;/])|[ \t]+\1(?![-\w]))+")
_REPEATED_BRACKETS = re.compile(rf"([(\[][ \t]*({_ANY})[ \t]*[)\]])(?:[ \t]*[(\[][ \t]*\2[ \t]*[)\]])+")
_BRACKETS = re.compile(r"([(\[])([^()\[\]]{1,160})([)\]])")
# A company's mark, the rest of its name ("& Company", "SpA", "Companies") and up to two brackets, to the end.
_AFTER_NAME = re.compile(rf"{_MARK}({_INDEX})(?:['’]s)?(?:[ \t,.&;-]+(?:[A-Z0-9][\w.&'’-]*|(?:de|of|and|la|du|der"
                         rf"|del|y|e|et)\b)|[ \t,]*[(\[][^()\[\]]{{0,120}}[)\]]){{0,6}}[ \t,.;]*$")
# Brackets that hold nothing but tickers already replaced: "(A)", "(A, B)", "(A; [code])", after a label too:
# "(NYSE - A)", "(Oslo: A)".
_TICKER_BRACKET = (rf"[(\[]\s*(?:\[code\]\s*[,;/]\s*)?(?:[A-Za-z][A-Za-z .&]{{1,24}}\s*[:–-]\s*)?{_ANY}"
                   rf"(?:\s*(?:[,;/&|]|and|-)?\s*(?:{_ANY}|\[code\]))*\s*[)\]]")
_TICKERS = re.compile(rf"{_TICKER_BRACKET}(?:[ \t]*{_TICKER_BRACKET})*")
_UNTIDY = re.compile(r"[ \t]+([,.](?![\w.]))|[ \t]{2,}")  # What a name leaves when it goes: "A , B", "A  said".
_THE = re.compile(rf"\bThe[ \t]+(?={_ANY})")  # "The Cheesecake Factory": the article was the name's.
# A placeholder's name ending with a word of companies or funds: "Company A Closed-End Opportunity Fund, Inc." (FOF).
_FUND_BEFORE = re.compile(rf"{_ANY}(?:['’]s)?(?:[ \t,.&-]+[A-Z][\w.&'’-]*){{0,6}}[ \t,.]+(?:Fund|Trust|ETF|ETFs|Inc"
                          rf"|Corp|Corporation|Company|Co|Ltd|Limited|LLC|L\.?P|plc|PLC|Group|Holdings|Bancorp|Partners"
                          rf"|REIT|Capital|Income|Portfolio)\.?[ \t,]*$")
_FORMERLY = re.compile(rf"({_ANY})((?:['’]s)?,?[ \t]*\(?[ \t]*(?i:f/k/a|fka|formerly(?:[ \t]+known[ \t]+as)?"
                       rf"|previously[ \t]+known[ \t]+as|formerly[ \t]+called)[ \t]+)({_ANY})\)?")
# What the brackets of a defined term hold besides the names: ("A" or the "Company"), (« A » ou la « Société »).
_DEFINING = re.compile(r"[\s,;.“”\"„«»]+|\b(?:or|and|ou|et|oder|und|collectively|together|the|la|le|die|der|das"
                       r"|hereinafter|referred\s+to\s+as|as)\b", re.IGNORECASE)
_PIECE = re.compile(rf"{_ANY}(?:['’]s)?|[^\W_]+(?:[.&'’][^\W_]+)*")
_GAP = re.compile(r"[ \t ]*(?:[-&/+!][ \t ]*)?")
_QUOTED = re.compile(r"(\bthe\s+)?[“\"„«]\s*([^“”\"„«»]{1,60}?)\s*[,.]?\s*[”\"“»]", re.IGNORECASE)
_LIST = r"(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or|vs\.?|versus|&)\s+|\s*/\s*)"
_STOCK = r"(?:stock|shares|share\s+price|stockholders|shareholders|investors)"


def _once(match: re.Match) -> str:
    """One mention for a repeated one: "Ford's (F) sales" keeps the possessive; "Alphabet's Google Cloud" is the
    company's cloud, not the company's company's."""
    possessive = match.group(2) or ""
    rest = re.sub(r"[(\[][^()\[\]]*[)\]]", "", match.group(0)[len(match.group(1)) + len(possessive):])
    return match.group(1) + ("" if _MARK in rest else possessive)


def _mark(index: int) -> str:
    return _MARK + chr(_FIRST + index)


def _day(day) -> pd.Timestamp | None:
    """A day without its time zone, or None."""
    if day is None or pd.isna(day):
        return None
    day = pd.Timestamp(day)
    return day.tz_localize(None) if day.tzinfo else day


def words(text: str) -> list[tuple[str, int, int]]:
    """The words of a text, each with its start and end. A French article before a name is not its word: "d'Innate
    Pharma" gives Innate and Pharma."""
    found = []
    for match in TOKEN.finditer(text):
        word, start = match.group(), match.start()
        if len(word) > 2 and word[0] in "dDlL" and word[1] in "'’" and word[2].isupper():
            word, start = word[2:], start + 2
        found.append((word, start, match.end()))
    return found


@lru_cache(maxsize=1 << 20)
def lower(word: str) -> str:
    """A word as aliases are kept: small letters, a straight apostrophe, no accents (Económico, Société)."""
    word = word.lower().replace("’", "'")
    if word.isascii():
        return word
    return "".join(letter for letter in unicodedata.normalize("NFKD", word) if not unicodedata.combining(letter))


def clean(name: str) -> tuple[str, str]:
    """A company's name without the security it names or its legal form, and with its legal form: "Alphabet Inc.
    Class A Common Stock" gives Alphabet and Alphabet Inc; "The Walt Disney Company", Walt Disney and Walt Disney
    Company."""
    name = " ".join(str(name).split())
    for match in SECURITY.finditer(name):
        if TOKEN.search(name[:match.start()]):
            name = name[:match.start()]
            break
    parts = name.replace(",", " ").split()
    while len(parts) > 1 and parts[0].lower() == "the":
        parts = parts[1:]
    count = len(parts)
    while count > 1 and (parts[count - 1].strip(".()/").lower() in LEGAL or parts[count - 1] in ("&", "and", "-")):
        count -= 1
    legal = parts[count].strip(".()/") if count < len(parts) else ""
    return " ".join(parts[:count]), " ".join(parts[:count] + [legal]) if legal.lower() in SAME_LEGAL else ""


def name_aliases(name: str) -> dict[tuple[str, ...], str]:
    """The aliases a Polygon name gives, each as its lower-case words, with its kind: name, start, word or
    initials."""
    expanded = " ".join(ABBREVIATIONS.get(lower(word).strip(".,"), word) for word in str(name).split())
    if expanded != " ".join(str(name).split()):
        found = _name_aliases(expanded)  # TENCENT HLDGS UNSP/ADR is Tencent Holdings too.
        for key, kind in _name_aliases(name).items():
            found.setdefault(key, kind)
        return found
    return _name_aliases(name)


def _name_aliases(name: str) -> dict[tuple[str, ...], str]:
    cleaned, with_legal = clean(name)
    found: dict[tuple[str, ...], str] = {}

    def add(key, kind):
        key = tuple(key)[:MAX_WORDS]
        if key and any(len(word) > 1 for word in key) and not (len(key) == 1 and key[0] in LEGAL | LINKS):
            found.setdefault(key, kind)

    # As the news may write it: Fubo TV for FuboTV, W. P. Carey for W.P. Carey.
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", cleaned)
    dotted = re.sub(r"\b([A-Z])([A-Z])(?= [A-Z][a-z])", r"\1.\2.", spaced)  # J.P. Morgan.
    plain = re.sub(r"['’]s\b", "s", cleaned)  # BJs, Macys.
    initials = re.sub(r"\b([A-Z])\.(?=[A-Z]\b)", r"\1. ", cleaned)
    for text in (cleaned, cleaned.replace(" & ", " and "), cleaned.replace(" and ", " & "), spaced, initials, dotted,
                 plain):
        add([lower(word) for word, _, _ in words(text)], "name")
    key = [lower(word) for word, _, _ in words(cleaned)]
    if with_legal:
        legal = lower(with_legal.split()[-1])
        add(key + [legal], "name")  # News Corp, Southern Company.
        add(key + [SAME_LEGAL[legal]], "name")
    if len(key) > 2 and key[:2] == ["united", "states"]:
        add(["u.s"] + key[2:], "name")  # U.S. Steel.
        add(["us"] + key[2:], "name")
    if len(key) == 2:
        add(["".join(key)], "name")  # ExxonMobil.
    for word in key:
        if word.endswith(".com"):
            add([word[:-4]], "name")  # Amazon.
    for count in range(len(key) - 1, 0, -1):
        if key[count - 1] not in LINKS:
            add(key[:count], "start")
    for word in key[1:]:
        if len(word) >= 4 and word[0].isalpha():
            add([word], "word")
    initials = [word[0] for word in key if word not in LINKS and word[0].isalpha()]
    if len(initials) >= 2 and all("&" not in word and "." not in word for word in key):
        add(["".join(initials)], "initials")
        add(["".join(initials) + "c"], "initials")  # TSMC: with Company or Corporation.
        if len(initials) == 2 and re.search(r"\s(?:&|and)\s", cleaned):
            add(["&".join(initials)], "initials")  # P&G.
    return found


def _capital(word: str) -> bool:
    """Whether a word may start a name: it has a capital letter (Ford, eBay) or starts with a digit (3M)."""
    return word[0].isdigit() or any(letter.isupper() for letter in word)


def _joined(text: str, word: str, end: int, after: str, start: int, loose: bool = False) -> bool:
    """Whether two words that follow each other are written as one name: nothing between them but spaces and one of
    - & / + !, a period after an abbreviation (U.S. Steel, J.M. Smucker, a placeholder) or punctuation before a
    legal form (Amazon.com, Inc.). A sentence's end, a colon, a bracket or a new line parts them. A dash between
    spaces parts them too ("Short Interest - Ares Management"), and a period before a capital does not, if loose:
    an alias of several words is specific enough ("Warner Bros. Discovery")."""
    gap = text[end:start]
    if _GAP.fullmatch(gap):
        return loose or not re.fullmatch(r"[ 	 ]+[-–—][ 	 ]+", gap)
    gap = gap.strip(" \t ")
    if gap == ".":
        return (len(word) <= 3 or "." in word or word.startswith(_MARK) or lower(after) in LEGAL
                or (loose and after[:1].isupper()))
    return gap in (",", ".,") and lower(after) in LEGAL


def _find(text: str, found: list[tuple[str, int, int]], keys, starts, any_case: bool = False):
    """Yield (first word, word after the last, alias, possessive, capital) for every alias of keys in a text's words.
    starts holds every alias's first words; possessive tells that the text adds 's to the alias (Ford's); capital,
    that the text writes it as a name: each word but a link (of, and) has a capital letter, so "Tuesday Morning"
    and "Bank of America" are names and "Tuesday morning" is not. Without any_case, only names are found."""
    exact = [lower(word) for word, _, _ in found]
    named = [_capital(word) for word, _, _ in found]
    for first in range(len(found)):
        if not (named[first] or any_case):
            continue
        capital = named[first]
        for count in range(1, min(MAX_WORDS, len(found) - first) + 1):
            place = first + count - 1
            if count > 1 and not _joined(text, found[place - 1][0], found[place - 1][2], found[place][0],
                                         found[place][1], loose=True):
                break
            head, last = tuple(exact[first:place]), exact[place]
            capital = capital and (named[place] or last in LINKS)
            if not (capital or any_case):
                break
            if head + (last,) in keys:
                yield first, first + count, head + (last,), False, capital
            elif last.endswith("'s") and len(last) > 2 and head + (last[:-2],) in keys:
                yield first, first + count, head + (last[:-2],), True, capital
            if head + (last,) not in starts:
                break


def _near(text: str, found: list[tuple[str, int, int]], index: int) -> list[str]:
    """The capitalized words that touch a word, as in "Price Target" (price<), "Target Date" (>date) or "Kinross
    Gold": the one before it, marked <, and the one after it, marked >. A possessive before it ("Musk's Twitter")
    and a role after it ("Target CEO") do not count."""
    word, start, end = found[index]
    near = []
    if index and not text[found[index - 1][2]:start].strip(" "):
        before = found[index - 1][0]
        if _capital(before) and lower(before) not in LINKS and not lower(before).endswith("'s"):
            near.append(f"{lower(before)}<")
    if index + 1 < len(found) and not text[end:found[index + 1][1]].strip(" "):
        after = found[index + 1][0]
        if _capital(after) and lower(after) not in LINKS | ROLES:
            near.append(f">{lower(after)}")
    return near


def _beside(text: str, found: list[tuple[str, int, int]], index: int) -> list[str]:
    """The words that touch a word, whatever their case, marked as _near() marks them; not links or the words of a
    headline (its, new)."""
    word, start, end = found[index]
    beside = []
    if index and not text[found[index - 1][2]:start].strip(" ") and lower(found[index - 1][0]) not in LINKS | HEADLINE:
        beside.append(f"{lower(found[index - 1][0])}<")
    if index + 1 < len(found) and not text[end:found[index + 1][1]].strip(" ") and (
            lower(found[index + 1][0]) not in LINKS | HEADLINE):
        beside.append(f">{lower(found[index + 1][0])}")
    return beside


def _sentence(found: list[tuple[str, int, int]], index: int) -> bool:
    """Whether a word stands in a sentence, not in a headline that gives every word a capital: two of the eight
    words around it are written small."""
    around = found[max(0, index - 4):index] + found[index + 1:index + 5]
    return sum(word[0].islower() and lower(word) not in LINKS for word, _, _ in around) >= 2


def _alone(text: str, found: list[tuple[str, int, int]], index: int) -> bool:
    """Whether the word stands alone as a name: no capitalized word touches it."""
    return not _near(text, found, index)


def _standalone(text: str, found: list[tuple[str, int, int]], index: int, strong: bool = False) -> bool:
    """Whether another company's one-word name stands as its name. Not after a first name ("Harrison Ford", "Mike
    Matson"), not as a first name before a surname ("Lindsay Fitzgerald"), not before "de" and a capital ("Rio de
    Janeiro"). A strong name (Amazon, Pfizer: most articles that write it are its company's) stands as its name
    anywhere else. A weaker one does where it touches no capitalized word but those of a headline ("Citigroup
    Maintains"); in a headline, where every word has a capital, the capitals tell nothing."""
    word, start, end = found[index]
    previous = found[index - 1] if index and not text[found[index - 1][2]:start].strip(" ") else None
    following = found[index + 1] if index + 1 < len(found) and not text[end:found[index + 1][1]].strip(" ") else None
    if previous is not None and previous[0][:1].isupper() and lower(previous[0]) in FIRST_NAMES:
        return False
    if following is not None and _capital(following[0]) and lower(word) in FIRST_NAMES and (
            lower(following[0]) not in TITLE):
        return False
    if following is not None and lower(following[0]) in NAME_LINKS and index + 2 < len(found) and (
            _capital(found[index + 2][0])):
        return False
    if strong:
        return True
    if previous is not None and previous[0][:1].isupper() and lower(previous[0]) in PLACE_STARTS:
        return False  # New Brunswick, Fort Worth.
    if not _sentence(found, index):
        return True
    return all(near.strip("<>") in TITLE for near in _near(text, found, index))


def _bare_ticker(text: str, found: list[tuple[str, int, int]], index: int) -> bool:
    """Whether a token is written as a ticker, not as part of a name or a product: not joined to a word by - + & /
    ("CGI-animated", "IGE+XAO"), not before a number ("RTX 4070"), not touching a capitalized word but a headline's
    ("ROG Zephyrus", "RPT Alliance"; "NVDA Stock Soars" is fine), not in a text in capitals."""
    word, start, end = found[index]
    if text[end:end + 1] in "-+&/" and text[end + 1:end + 2].isalnum() or (
            start and text[start - 1] in "-+&/" and text[start - 2:start - 1].isalnum()):
        return False
    headline = not _sentence(found, index)  # Every word has a capital: they tell nothing.
    if index + 1 < len(found) and not text[end:found[index + 1][1]].strip(" "):
        after = found[index + 1][0]
        if after[0].isdigit() or (not headline and _capital(after) and lower(after) not in TITLE
                                  and not after.isupper()):
            return False
    if index and not headline and not text[found[index - 1][2]:start].strip(" "):
        before = found[index - 1][0]
        if _capital(before) and lower(before) not in TITLE | LINKS and not before.isupper() and not (
                lower(before).endswith("'s")):
            return False
    return not _shouting(text, start, end)


def _capitals_before(before: str) -> bool:
    """Whether a text ends with a name: one to six capitalized words, the last just before the bracket."""
    tail = re.search(r"((?:[A-Z][\w.&'’-]*[ \t]+(?:(?:of|and|de|&)[ \t]+)?){0,5}[A-Z][\w.&'’-]*)[ \t,]*$", before)
    return tail is not None and not all(lower(word) in TITLE | HEADLINE for word in tail.group(1).split())


def _shouting(text: str, start: int, end: int) -> bool:
    """Whether a word stands in a text in capitals: THE F WORD, ELECTRIC CARS SELL. Its clause (up to a period, a
    colon, a semicolon, a bar or a new line) has at least two other words of letters, none written small; a list of
    tickers in a headline ("Catalyst Watch: CPI, BABA, DIS, PLTR, UPS Preview") is not one."""
    marks = ".;:|\n" + _SPLIT
    left = max(text.rfind(mark, max(0, start - 160), start) for mark in marks)
    right = [position for position in (text.find(mark, end, end + 160) for mark in marks) if position >= 0]
    clause = text[left + 1:start] + " " + text[end:min(right, default=end + 160)]
    around = [found for found, _, _ in words(clause) if len(found) >= 2 and found.isalpha()]
    return (len(around) >= 2 or any(len(found) >= 5 for found in around)) and all(found.isupper() for found in around)


def _starts(keys) -> set:
    return {key[:count] for key in keys for count in range(1, len(key) + 1)}


def seen_names(text: str, tickers) -> list[tuple[str, tuple[str, ...]]]:
    """The (ticker, name) pairs a text writes as "Name (TICKER)" or "Name (NYSE: TICKER)": the capitalized words
    just before the parenthesis, without a legal ending."""
    tickers, pairs = set(tickers), []
    for match in re.finditer(rf"[(\[]\s*(?:{EXCHANGE}\s*:\s*)?([A-Z][A-Z.]*)\s*[)\],;]", text):
        if match.group(1) not in tickers:
            continue
        before = text[max(0, match.start() - 120):match.start()]
        found = words(before)
        if not found or before[found[-1][2]:].strip(" ,."):
            continue  # Something else than a name stands before the parenthesis.
        run: list[str] = []
        for index in range(len(found) - 1, -1, -1):
            word, _, end = found[index]
            if run:
                gap = before[end:found[index + 1][1]]
                short = "." in word or len(word) <= 2  # U.S. Steel, J.M. Smucker.
                if not (re.fullmatch(r"[ &-]*", gap) or (re.fullmatch(r"[ ,.]*", gap) and (run[0] in LEGAL or short))):
                    break
            if not _capital(word):
                break  # "Shares of Coke (KO)" names Coke; "Bank of America (BAC)" is known from Polygon's name.
            run.insert(0, lower(word))
        while len(run) > 1 and (run[-1] in LEGAL or run[-1] in LINKS):
            run.pop()
        if run and run[-1].endswith("'s") and len(run[-1]) > 2:
            run[-1] = run[-1][:-2]  # "Alphabet's (GOOGL) cloud": the name is Alphabet.
        if run and run[-1] not in LEGAL and len(run) <= MAX_WORDS:
            pairs.append((match.group(1), tuple(run)))
    return pairs


def strip_dates(text: str) -> str:
    """Replace the dates of a text: a day or a month with its year by [date] ("March 15, 2023", "15 March", "March
    2023", "3/15/2023", "2023-03-15", "15 mars 2023"), a month alone by [month], a year by 20XX or 19XX (2023,
    FY2023, FY23, Q3'23, 2024-25) and '23 by 'XX. What only refers to a time stays: last year, Q3, Monday."""
    for pattern in (NUMERIC_DATE, SPACED_DATE, MONTH_COMMA_DAY, MONTH_DAY, DAY_MONTH, MONTH_YEAR, OTHER_DATE,
                    CJK_DATE, DAY_YEAR, WEEK):
        text = pattern.sub("[date]", text)
    text = SHORT_DATE.sub(r"\1[date]", text)
    text = MONTH_ALONE.sub("[month]", text)
    text = MAY.sub(lambda match: f"{match.group(1) or ''}[month]", text)
    text = SHORT_MONTH.sub(r"\1[month]", text)
    text = THE_MONTH.sub(r"\1[month]", text)
    text = RANGE_END.sub(r"\1[date]", text)
    text = HOLIDAY.sub("[date]", text)
    text = EDITION.sub("", text)
    text = PERIOD_YEAR.sub(lambda match: f"{match.group(1)} 20XX", text)
    text = YEAR_RANGE.sub(lambda match: f"{match.group(1)[:2]}XX-XX", text)
    text = YEAR.sub(lambda match: f"{match.group(1)[:2]}XX", text)
    text = OLD_YEAR.sub(r"\g<1>1XXX", text)
    return SHORT_YEAR.sub("'XX", text)


def label(index: int) -> str:
    """The placeholder of an article's index-th company: Company A, Company B... Company AA."""
    letters, index = "", index + 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(65 + rest) + letters
    return f"Company {letters}"


def _classes(first: str, second: str) -> bool:
    """Whether two tickers look like two classes of one company's shares: GOOG and GOOGL, BRK.A and BRK.B, DISCA
    and DISCK."""
    first, second = first.split(".")[0], second.split(".")[0]
    short, long = sorted((first, second), key=len)
    return (first == second or (long.startswith(short) and len(long) == len(short) + 1)
            or (len(first) == len(second) == 5 and first[:4] == second[:4]))


class Aliases:
    """The aliases to hide. names maps each ticker to {alias words: [(how, since, kind, any case, apart)]}: how is
    CAPITALS for initials (hidden where written in capitals), ALONE (hidden unless a word of apart touches it) or
    ALWAYS; since, the first day the alias applies, or None; any case, whether it is hidden written small too.
    doubtful holds the tickers the news write for something else too ("AI"), hidden only where written as tickers;
    known, every ticker of the table. wide maps the aliases hidden in any article to [(ticker, how, since)], and
    wide_symbols holds the tickers hidden so; heads maps each ticker to the first words of its names, which tell
    that "Lindsay (LNN)" names LNN and "artificial intelligence (AI)" does not name AI. Aliases() hides only the
    tickers."""

    def __init__(self, table: pd.DataFrame | None = None):
        self.names: dict[str, dict[tuple[str, ...], list]] = defaultdict(dict)
        self.doubtful: set[str] = set()
        self.known: set[str] = set()
        self.companies: dict[str, dict] = defaultdict(dict)  # Ticker -> {name: since}.
        self.wide: dict[tuple[str, ...], list] = {}
        self.wide_starts: set = set()
        self.wide_symbols: set[str] = set()
        self.heads: dict[str, set[str]] = defaultdict(set)
        if table is None:
            return
        optional = {name: table[name] if name in table else [default] * len(table)
                    for name, default in (("any_case", False), ("wide", ""), ("apart", ""))}
        for ticker, alias, kind, how, since, any_case, wide, apart in zip(
                table.ticker, table.alias, table.kind, table.use, table.since, *optional.values()):
            since = None if pd.isna(since) else pd.Timestamp(since) - SINCE_SLACK
            key = tuple(alias.split(" "))
            wide = wide if isinstance(wide, str) else ""
            if kind == "ticker":
                self.known.add(ticker)
                if how == NEVER:
                    self.doubtful.add(ticker)
                if wide:
                    self.wide_symbols.add(ticker)
                continue
            if how != NEVER:
                apart = frozenset(apart.split()) if isinstance(apart, str) else frozenset()
                self.names[ticker].setdefault(key, []).append(
                    (CAPITALS if kind == "initials" else how, since, kind, bool(any_case), apart))
            if wide:
                self.wide.setdefault(key, []).append((ticker, wide, since))
            if kind in NAMES and len(key[0]) > 1 and key[0] not in LEGAL | LINKS:
                self.heads[ticker].add(key[0])
            if kind == "name" and len(key) < MAX_WORDS:  # A name cut at MAX_WORDS is shared by other companies.
                known = self.companies[ticker].get(key, since)
                self.companies[ticker][key] = None if since is None or known is None else min(since, known)
        for owners in self.wide.values():  # Several owners are nearly always one issuer: MS and MSpF, AIA's ADRs.
            owners.sort(key=lambda owner: (len(owner[0]), owner[0]))
        self.wide_starts = _starts(self.wide)

    @classmethod
    def read(cls, path: Path) -> "Aliases":
        """The table a file holds. ValueError if scripts/build_ticker_aliases.py made it before mask() hid dates
        and other companies: it lacks what tells which aliases are wide."""
        table = pd.read_parquet(path)
        missing = [name for name in ALIAS_COLUMNS if name not in table]
        if missing:
            raise ValueError(f"{path} lacks the columns {missing}: build it again with "
                             f"scripts/build_ticker_aliases.py.")
        return cls(table)

    def same_company(self, first: str, second: str, published: pd.Timestamp | None = None) -> bool:
        """Whether two tickers are one company on the article's day: they share a name then, and are two classes of
        its shares, or one took the other's name with its legal form (a new ticker, a merger). Two companies of one
        name (Blue Owl Capital Inc. and Corporation, the two Independent Bank Corp.) are not."""
        mine, theirs = self.companies.get(first, {}), self.companies.get(second, {})
        for key in mine.keys() & theirs.keys():
            dates = (mine[key], theirs[key])
            if published is not None and any(since is not None and published < since for since in dates):
                continue
            if _classes(first, second) or (key[-1] in SAME_LEGAL and any(since is not None for since in dates)):
                return True
        return False


# Words a capital letter stands after as a letter: Plan B, Class C, Vitamin D, Generation X.
LETTER_WORDS = {"class", "series", "plan", "vitamin", "vitamins", "hepatitis", "type", "grade", "phase", "tier",
                "model", "section", "title", "schedule", "form", "item", "part", "exhibit", "rule", "category",
                "option", "group", "unit", "building", "block", "gate", "hall", "wing", "line", "route", "level",
                "zone", "appendix", "annex", "chapter", "article", "no", "generation", "gen", "cohort", "arm", "stage",
                "round", "team", "division", "channel", "size", "row", "seat", "floor", "terminal", "pier", "lot",
                "area", "sector", "region", "program", "version", "variant", "strain", "letter", "pillar", "box",
                "studio", "mr", "dr", "ms", "mrs", "big", "plus", "pro", "max"}
_LETTER = {letter: re.compile(rf"(?<![\w.&'’$/-]){letter}(?=['’]s\b|(?![\w&'’/-]|\.\w))")
           for letter in "BCDEFGHJKLMNOPQRSTUVWXYZ"}
_IN_LIST = re.compile(
    # Zacks compares with a foreign company by its OTC ticker, which Polygon may not name: "MARUY vs. Company A:",
    # "Company A or UOVEY:".
    rf"(?P<versus>(?<![\w.&'’$-])(?P<symbol3>[A-Z]{{3,4}}[YF])(?=\s+(?:vs\.?|or|versus)\s+{_MARK})"
    rf"|(?P<before2>{_ANY}\s+(?:vs\.?|or|versus)\s+)(?P<symbol4>[A-Z]{{3,4}}[YF])(?![\w&'’-]|\.\w))"
    rf"|(?P<before>{_ANY}(?:['’]s)?{_LIST})(?P<symbol>[A-Z][A-Z0-9]{{1,5}}(?:\.[A-Z])?)(?![\w&'’-]|\.\w)"
    rf"|(?P<none>)(?<![\w.&'’$-])(?P<symbol2>[A-Z][A-Z0-9]{{1,5}}(?:\.[A-Z])?)(?={_LIST}{_MARK})")


def _letter(match: re.Match, mark: str) -> str:
    """The mark for a one-letter ticker written alone, or the letter where it is one: after Plan or Class, the
    initial of a name (John F. Kennedy), an abbreviation (E. coli), after a number ("$5 M"), in a text in capitals."""
    text, start, end = match.string, match.start(), match.end()
    if text[start - 1:start] in ("(", "[") and text[end:end + 1] in (")", "]"):
        return match.group(0)  # In brackets, brackets() has decided: "Plan F (F)" spells it.
    before, after = text[max(0, start - 40):start], text[end:end + 3]
    previous = re.search(r"([\w.]+)[ \t]*$", before)
    word = lower(previous.group(1).rstrip(".")) if previous else ""
    if word in LETTER_WORDS or re.search(r"\d[ \t]*$", before):
        return match.group(0)
    if re.match(r"\.\s*[a-z]", after) or (re.match(r"\.\s+[A-Z]", after) and previous is not None
                                           and previous.group(1)[:1].isupper() and before.endswith(" ")):
        return match.group(0)
    return match.group(0) if _shouting(text, start, end) else mark


def _expands(before: str, symbol: str) -> bool:
    """Whether the words before a bracket spell its letters, so that the bracket defines an abbreviation:
    "artificial intelligence (AI)", "account takeover (ATO)", "Fighter Base Publishing, Inc. (FBP)". The letters
    must come in order from the last words, the first one starting a word (Schwartz and Hearst's test), within
    min(letters + 5, 2 * letters) words."""
    letters = re.sub(r"[^0-9a-z]", "", symbol.lower())
    if not letters:
        return False
    found = re.findall(r"\S+", before[-200:].split(_MARK)[-1])
    while found and lower(found[-1].strip(".,;:")) in LEGAL:
        found.pop()
    text = " ".join(found[-min(len(letters) + 5, 2 * len(letters)):]).lower()
    place = len(text) - 1
    for position in range(len(letters) - 1, -1, -1):
        letter = letters[position]
        while place >= 0 and (text[place] != letter or (position == 0 and place > 0 and text[place - 1].isalnum())):
            place -= 1
        if place < 0:
            return False
        place -= 1
    return True


@lru_cache(maxsize=4096)
def _as_ticker(symbol: str, long: bool) -> list[tuple[re.Pattern, str]]:
    """Where a text writes a symbol that is also a word (F, AI) as a ticker: in a list with companies already
    replaced, beside "stock" or "shares", after "shares of" or "ticker", in quotes. Each pattern comes with what
    replaces it, {} being the company's mark."""
    name = re.escape(symbol)
    free = r"(?<![\w.&'’$-])"  # Not inside a word, an amount or "F-150".
    end = r"(?![\w&'’-]|\.\w)"  # Not "F-150", "F&B" or "U.S."; the period that ends a sentence is fine.
    patterns = [(rf"({_ANY}(?:['’]s)?{_LIST}){name}{end}", r"\g<1>{}"),
                (rf"{free}{name}(?=\s+(?i:earnings\s+call\b)|\s+(?:F?Q[1-4]|F?[1-4]Q)\b)", "{}"),
                (rf"((?i:more\s+on)\s+){name}{end}", r"\g<1>{}"),
                # Law firms' headlines: "FF INVESTOR ALERT:", "Investors of Lineage- LINE".
                (rf"{free}{name}(?=\s+(?i:investors?|shareholders?|stockholders?)\s+(?i:alert|notice|deadline"
                 rf"|reminder|action|news)\b)", "{}"),
                (rf"((?<=\S)[-–—]\s*){name}(?=\s*(?:{_SPLIT}|\||$))", r"\g<1>{}"),
                # "E or FUPBY: Which Is the Better Value Stock", "FUPBY vs. E:".
                (rf"{free}{name}(?=\s+(?:vs\.?|or|versus)\s+(?:{_MARK}|[A-Z][A-Z0-9.]{{1,6}}\s*:))", "{}"),
                (rf"((?:{_ANY}|\b[A-Z][A-Z0-9.]{{1,6}})\s+(?:vs\.?|or|versus)\s+){name}(?=\s*:)", r"\g<1>{}"),
                (rf"{free}{name}(?={_LIST}{_MARK})", "{}"),
                (rf"{free}(?<!Class ){name}(?=(?:['’]s)?\s+{_STOCK}\b)", "{}"),
                (rf"((?i:shares|stock|stake|position)\s+(?:of|in)\s+|(?i:ticker|symbol)\s*:?\s*){name}{end}",
                 r"\g<1>{}"),
                (rf"([“\"„«]){name}(?=[”\"“»])", r"\g<1>{}"),
                (rf"(\s[-–—|]\s*){name}(?=\s*{_SPLIT})", r"\g<1>{}")]
    if long:  # "Buy A Home" and "A: the list" are not about Agilent.
        patterns += [(rf"((?i:buy|sell|buying|selling|own|owning|shorting|bought|sold)\s+){name}{end}",
                      r"\g<1>{}"),
                     (rf"(^|{_SPLIT}){name}(?=\s*[:–—|]\s)", r"\g<1>{}")]
    return [(re.compile(pattern), replacement) for pattern, replacement in patterns]


class _Article:
    """One article being masked: the company of each mark in its text, and the names the article itself gives."""

    def __init__(self, aliases: Aliases, published, tickers: list[str], indexes: list[int], others: dict):
        self.aliases, self.published = aliases, published
        self.own = dict(zip(tickers, indexes))  # Symbol -> index, for the scored tickers and their other symbols.
        for other, ticker in others.items():
            if ticker in self.own and other not in self.own:
                self.own[other] = self.own[ticker]
        self.count = self.total = max(indexes, default=-1) + 1  # Scored companies; all companies.
        self.others: dict[str, int] = {}  # The other tickers the article names -> index.
        self.tickers: dict[int, list[str]] = defaultdict(list)
        for ticker, index in self.own.items():
            self.tickers[index].append(ticker)
        self.wordlike = {ticker for ticker in self.own if len(ticker) == 1 or ticker in aliases.doubtful}
        self.local: dict[tuple[str, ...], tuple] = {}  # Names the article gives -> (index, how, any case, apart).
        self.blocked: set[tuple[str, ...]] = set()  # First words of two companies' names.
        self.keys: dict[tuple[str, ...], tuple] = {}  # Alias -> (index, how, rank, any case, apart).
        for ticker, index in zip(tickers, indexes):
            for key, options in aliases.names.get(ticker, {}).items():
                valid = [option for option in options
                         if published is None or option[1] is None or published >= option[1]]
                if not valid:
                    continue
                rank = min(RANK[option[2]] for option in valid)
                # Of two tickers, the one whose name gives the alias has it, not the one the news once wrote it before.
                if key not in self.keys or rank < self.keys[key][2]:
                    how = min(valid, key=lambda option: (ALWAYS, ALONE, CAPITALS).index(option[0]))[0]
                    self.keys[key] = (index, how, rank, any(option[3] for option in valid),
                                      frozenset().union(*(option[4] for option in valid)))
        self.starts = _starts(self.keys)

    def _first_word(self, key: tuple, index: int) -> None:
        """Keep the first word of a company's full name, once the article writes that name, as a name of it where
        it stands alone: "Montero Mining and Exploration Ltd." is Montero after that, "Sandfire Resources America"
        Sandfire. Not a common word (First, Global), nor a word two of the article's companies start with."""
        head = key[:1]
        if len(key[0]) <= 3 or head in self.keys or head in self.blocked or (
                key[0] in COMMON | HEADLINE | LEGAL | LINKS | GENERIC):
            return
        if head in self.local and self.local[head][0] != index:
            del self.local[head]
            self.blocked.add(head)  # Brookfield Asset Management and Brookfield Renewable: Brookfield is neither.
        else:
            self.local.setdefault(head, (index, ALONE, False, None))

    def valid(self, since) -> bool:
        return self.published is None or since is None or self.published >= since

    def other(self, ticker: str) -> int:
        """The index of a ticker the article is not scored for: its company's, if the article has it already."""
        if ticker in self.own:
            return self.own[ticker]
        if ticker not in self.others:
            same = next((index for known, index in (self.own | self.others).items()
                         if self.aliases.same_company(known, ticker, self.published)), None)
            if same is None:
                same = min(self.total, _LAST - _FIRST)
                self.total += 1
            self.others[ticker] = same
            self.tickers[same].append(ticker)
        return self.others[ticker]

    def hide(self, text: str) -> str:
        text = CODES.sub("[code]", EMAIL.sub("[email]", text))
        self.foreign(text)
        text = self.listings(text)
        text = self.names(text)
        text = _LEGAL_AFTER.sub(r"\1", text)
        text = _BRACKETS.sub(self.brackets, text)
        text = self.contexts(text)
        text = self.runs(text)
        text = self.names(self.defined(text), local=True)
        text = _IN_LIST.sub(self._listed_other, self.named_tickers(text))
        text = HANDLE.sub(self.handle, LINK.sub("[link]", text))  # After the names: Amazon.com and JD.com are names.
        text = _REPEATED.sub(_once, _LEGAL_AFTER.sub(r"\1", text))
        text = _REPEATED.sub(_once, _BRACKETS.sub(self.defining, _REPEATED_BRACKETS.sub(r"\1", text)))
        text = _THE.sub("", self.formerly(text))
        return self.render(_UNTIDY.sub(lambda match: match.group(1) or " ", text))

    def foreign(self, text: str) -> None:
        """Polygon tags some articles about a foreign company with the US ticker it shares: Boralex (TSX: BLX) with
        Bladex's BLX, Saputo (TSX: SAP) with SAP's. A scored ticker written only after a foreign market, in an
        article that names its company nowhere, is that other company's: it gets its own placeholder, and the scored
        company's never shows (relevance 0)."""
        for ticker in list(self.own):
            abroad = len(_FOREIGN(ticker).findall(text))
            if not abroad or _AT_HOME(ticker).search(text) or abroad < len(
                    re.findall(rf"(?<![\w.$-]){re.escape(ticker)}(?![\w-])", text)):
                continue  # Not only abroad: also at home, or on its own ("TECK rose").
            index = self.own[ticker]
            keys = {key: rule for key, rule in self.keys.items() if rule[0] == index and rule[2] <= RANK["seen"]}
            if not keys or any(capital for *_, capital in _find(text, words(text), keys, _starts(keys))):
                continue  # Its name is in the article, or the table has none to tell.
            del self.own[ticker]
            self.tickers[index].remove(ticker)
            self.wordlike.discard(ticker)
            self.other(ticker)

    def named_tickers(self, text: str) -> str:
        """Hide the other tickers the article has tied to a company ("Technology Select Sector SPDR ETF (XLK)")
        wherever it writes them again: "XLK has outperformed"."""
        symbols = [symbol for symbol in self.others if len(symbol) > 1 and symbol not in self.aliases.doubtful
                   and re.fullmatch(r"[A-Z][A-Z0-9.]*", symbol)]
        if not symbols:
            return text
        found, spans = words(text), []
        for place, (word, start, end) in enumerate(found):
            symbol = word[:-2] if lower(word).endswith("'s") else word
            if symbol in symbols and not (start and text[start - 1] in "#@$") and _bare_ticker(text, found, place):
                spans.append((start, start + len(symbol), self.others[symbol]))  # Not "RTX 4070" or "#XLK".
        for start, end, index in reversed(spans):
            text = text[:start] + _mark(index) + text[end:]
        return text

    def handle(self, match: re.Match) -> str:
        """A handle (@FordNews) goes; a hashtag (#gold) only if it names a company (#tesla, #TSLA)."""
        tag = match.group(0)
        if tag[0] == "@":
            return "[handle]"
        word = tag[1:]
        known = (lower(word),) in self.aliases.wide or (lower(word),) in self.keys or (
            word.upper() in self.aliases.known and word.upper() not in self.aliases.doubtful)
        return "[handle]" if known else tag

    def formerly(self, text: str) -> str:
        """One company, one placeholder: "A fka B", "A (formerly B)" make B A."""
        merged = {}
        for match in _FORMERLY.finditer(text):
            new, old = (ord(mark[1]) - _FIRST for mark in (match.group(1), match.group(3)))
            if old >= self.count and new != old:
                merged[old] = new
        if not merged:
            return text
        text = re.sub(_ANY, lambda mark: _mark(merged.get(ord(mark.group(0)[1]) - _FIRST,
                                                          ord(mark.group(0)[1]) - _FIRST)), text)
        return _FORMERLY.sub(lambda match: match.group(1) if match.group(1) == match.group(3) else match.group(0),
                             text)

    def listings(self, text: str) -> str:
        """Hide the tickers written after a market, a dollar sign or "ticker symbol": the scored ones, then any
        other. A scored ticker goes with its class of shares (TECK.B) and where it starts a product's code
        (MDNA11, ABV-1505 for ABVC)."""
        for ticker, index in sorted(self.own.items(), key=lambda pair: -len(pair[0])):
            name = re.escape(ticker).replace(r"\.", r"[.\-/]")  # BRK.B is written BRK-B and BRK/B too.
            text = re.sub(rf"(?:(?<![\w.]){MARKETS}|\$){name}(?:[.\-/][A-Z]{{1,2}})?(?![A-Za-z0-9])",
                          lambda _: _mark(index), text)
            if "." in ticker:
                text = re.sub(rf"(?<![\w.$-]){name}(?![\w-])", lambda _: _mark(index), text)
            if len(ticker) >= 3 and ticker.isalpha() and ticker not in self.wordlike:
                stems = "|".join(sorted({name, re.escape(ticker[:-1])} if len(ticker) >= 4 else {name}, key=len,
                                        reverse=True))
                text = re.sub(rf"(?<![\w.$-])(?:{stems})(-?)(?=\d{{1,5}}(?![\w]))", lambda _: _mark(index) + "-", text)
        text = NAMED_SYMBOL.sub(lambda match: match.group(1) + _mark(self.other(match.group(2))), text)
        text = SUFFIXED.sub(lambda match: _mark(self.own.get(match.group(1)) if match.group(1) in self.own
                                                else self.other(match.group(1))), text)  # SHOP.TO, 9618.HK.

        def listed(match: re.Match) -> str:
            symbols = re.split(r"\s*[,/]\s*", match.group(1))
            first = next((self.own[symbol] for symbol in symbols if symbol in self.own), None)
            marks = []
            for symbol in symbols:  # The tickers one market lists together are one company's.
                if first is None:
                    first = self.other(symbol)
                elif symbol not in self.own and symbol not in self.others:
                    self.others[symbol] = first
                    self.tickers[first].append(symbol)
                marks.append(_mark(self.own.get(symbol, first)))
            return ", ".join(marks)

        text = LISTED.sub(listed, text)
        return CASHTAG.sub(lambda match: match.group(0) if match.group(1) in CURRENCIES
                           else _mark(self.other(match.group(1))), text)

    def names(self, text: str, local: bool = False) -> str:
        """Hide the names and the tickers that stand in the text: the scored companies' aliases and tickers, and
        the wide aliases and tickers of any other company. With local, the names the article itself gave."""
        found, spans, named = words(text), [], set()
        keys, starts = (self.local, _starts(self.local)) if local else (self.keys, self.starts)
        for first, last, key, possessive, capital in _find(text, found, keys, starts, any_case=True):
            # Where it is hidden: always, in capitals, or alone; apart holds the words that part it from its
            # company (Price Target), None any capitalized word.
            index, how, any_case, apart = self.local[key] if local else (
                self.keys[key][0], self.keys[key][1], self.keys[key][3], self.keys[key][4])
            near = _near(text, found, first)
            if not (capital or any_case) or (how == CAPITALS and not all(
                    re.sub(r"['’]s$", "", word).isupper() for word, _, _ in found[first:last])) or (
                    how == ALONE and (near if apart is None else any(
                        word in apart for word in near + _beside(text, found, first)))):
                continue
            spans.append((*self._span(text, found, first, last, key, possessive), index, 0))
            named.add(key)
            if not local and self.keys[key][2] == RANK["name"] and len(key) > 1:
                self._first_word(key, index)
        for place, (word, start, end) in enumerate(found):
            possessive = lower(word).endswith("'s")
            ticker = word[:-2] if possessive else word
            base, dot, suffix = ticker.partition(".")
            if dot and base in self.own and suffix.isalpha() and suffix.isupper() and len(suffix) <= 2:
                ticker = base  # TECK.B: the ticker with its class of shares.
            glued = re.match(r"[+&][A-Za-z0-9]+", text[end:]) if text[end:end + 1] in "+&" else None
            if len(ticker) > 1 and ticker in self.own and glued is not None and not local:
                # IGE+XAO: another company's name, which holds the ticker; it gets its own placeholder.
                spans.append((start, end + glued.end(), False, self.other(text[start:end + glued.end()]), 0))
            elif len(ticker) > 1 and ticker in self.own and ticker not in self.wordlike and glued is None:
                spans.append((start, end, possessive, self.own[ticker], 0))
            elif not local and ticker in self.aliases.wide_symbols and ticker not in self.own and not (
                    text[max(0, start - 1):start] in ("(", "[") and text[end:end + 1] in (")", "]")) and (
                    _bare_ticker(text, found, place)):
                spans.append((start, end, possessive, self.other(ticker), 1))  # In brackets: brackets() tells.
        if not local:
            # The first words of a scored company's full name are its own, not another company's: TC BioPharm for
            # TC BioPharm Holdings, Voyager for Voyager Technologies once the article writes that name.
            prefixes = {}
            for key, rule in self.keys.items():
                if len(key) > 1 and rule[2] == RANK["name"]:
                    for count in range(len(key) - 1, 0, -1):
                        if count > 1 or key in named:
                            prefixes.setdefault(key[:count], rule[0])
            for first, last, key, possessive, capital in _find(text, found, self.aliases.wide,
                                                               self.aliases.wide_starts):
                if key in self.keys:
                    continue  # The scored company's own alias: hidden above, or not to hide in its articles.
                if key in prefixes:
                    if len(key) > 1 or _alone(text, found, first):
                        spans.append((*self._span(text, found, first, last, key, possessive), prefixes[key], 0))
                    continue
                owners = self.aliases.wide[key]
                owner = next((owner for owner in owners if self.valid(owner[2])), owners[0])
                if len(key) == 1 and not _standalone(text, found, first, strong=owner[1] == ALWAYS):
                    continue  # "Harrison Ford", "Lindsay Fitzgerald", "Rio de Janeiro".
                spans.append((*self._span(text, found, first, last, key, possessive), self.other(owner[0]), 1))
        pieces, position = [], 0
        for start, end, possessive, index, _ in sorted(spans, key=lambda span: (span[0], -span[1], span[4])):
            if start >= position and not (start and text[start - 1] in "$€£#@"):  # Longest first; not "$3M".
                pieces += [text[position:start], _mark(index) + ("'s" if possessive else "")]
                position = end
        return "".join(pieces) + text[position:]

    @staticmethod
    def _span(text: str, found: list, first: int, last: int, key: tuple, possessive: bool) -> tuple[int, int, bool]:
        """Where a name starts and ends in the text, and whether it is a possessive: with the period of its legal
        form ("Costco Wholesale Corp.'s", "Stride, Inc. Securities") unless a sentence starts after it, and with
        the apostrophe of "General Motors' sales"."""
        start, end = found[first][1], found[last - 1][2]
        if key[-1] in LEGAL and text[end:end + 1] == "." and not re.match(rf"\.[ \t]+{SENTENCE}\b", text[end:]):
            end += 1
        if not possessive and text[end:end + 1] in ("'", "’") and not text[end + 1:end + 2].isalnum() and (
                text[max(0, start - 1):start] not in ("'", "‘", "\"", "“")):
            possessive, end = True, end + 1
        return start, end, possessive

    def brackets(self, match: re.Match) -> str:
        """Hide the tickers of "(F)", "(GOOGL, GOOG)" and "(TSX: BTO, NYSE: BTG)". A scored one that is also a
        word stays where the words before it spell it: "artificial intelligence (AI)". A ticker the article is not
        scored for goes after its company's name ("Lindsay (LNN)"), and if it is nearly always a ticker, wherever
        the words before do not spell it: not in "account takeover (ATO)"."""
        inside, before = match.group(2), match.string[max(0, match.start() - 240):match.start()]
        follows = _AFTER_NAME.search(before)
        after = None if follows is None else ord(follows.group(1)) - _FIRST
        parts = re.split(r"([,;/])", inside)
        found = [re.fullmatch(rf"(\s*)({SYMBOL}|[A-Za-z][A-Za-z.]*)(\s*)", part) for part in parts]
        listing = _MARK in inside or any(
            part and part.group(2) in self.own and part.group(2) not in self.wordlike for part in found)
        for position, part in enumerate(found):
            if part is None:
                continue
            symbol = part.group(2)
            if symbol in self.own:
                index = self.own[symbol]
                if symbol in self.wordlike and not listing and after != index and _expands(before, symbol):
                    continue  # "artificial intelligence (AI)" defines a word.
            elif symbol in self.others and (after == self.others[symbol] or listing):
                index = self.others[symbol]
            elif symbol in self.aliases.known and (self._named(before, symbol) or (
                    symbol in self.aliases.wide_symbols and not _expands(before, symbol)) or (
                    _FUND_BEFORE.search(before) is not None and symbol not in self.aliases.doubtful
                    and len(symbol) > 1) or (
                    # "National Australia Bank (NABZY)": a ticker after a name, unless the name spells it (an
                    # acronym it defines).
                    len(symbol) >= 4 and symbol not in self.aliases.doubtful and _capitals_before(before)
                    and not _expands(before, symbol))):
                index = self.other(symbol)
            else:
                continue
            parts[position] = f"{part.group(1)}{_mark(index)}{part.group(3)}"
        return match.group(1) + "".join(parts) + match.group(3)

    def _named(self, before: str, symbol: str) -> bool:
        """Whether the words just before a bracket name the ticker's company: one of them starts a name of it."""
        heads = self.aliases.heads.get(symbol)
        if not heads:
            return False
        tail = words(before[-90:])[-MAX_WORDS:]
        return any(_capital(word) and (lower(word) in heads or lower(word).removesuffix("'s") in heads)
                   for word, _, _ in tail)

    def contexts(self, text: str) -> str:
        """Hide the scored tickers that are also words where the text writes them as tickers."""
        listed = "|".join(re.escape(ticker) for ticker in sorted(self.own, key=len, reverse=True))
        for _ in range(2):  # "T, F and Company C": F is in a list once T is replaced.
            for ticker in sorted(self.wordlike, key=len, reverse=True):
                mark, name = _mark(self.own[ticker]), re.escape(ticker)
                if len(ticker) == 1 and ticker not in "AI":  # A, I: words; other letters are rarely one alone.
                    text = _LETTER[ticker].sub(lambda match: _letter(match, mark), text)
                for pattern, replacement in _as_ticker(ticker, len(ticker) > 1):
                    text = pattern.sub(replacement.replace("{}", mark), text)
                if len(self.own) > 1:  # "T, F and GM": a list of the article's own tickers.
                    text = re.sub(rf"(?<![\w.&'’$-]){name}(?={_LIST}(?:{listed})(?![\w&'’-]|\.\w))", mark, text)
            # Another company's ticker in a list with the article's companies: "Company A, Company B and GM".
            text = _IN_LIST.sub(self._listed_other, text)
        return text

    def _listed_other(self, match: re.Match) -> str:
        symbol = match.group("symbol") or match.group("symbol2")
        if match.lastgroup == "versus":  # "MARUY vs. Company A": the OTC ticker of a foreign company.
            return (match.group("before2") or "") + _mark(self.other(match.group("symbol3") or match.group("symbol4")))
        if symbol in self.own or symbol not in self.aliases.known or symbol in self.aliases.doubtful:
            return match.group(0)
        return (match.group("before") or "") + _mark(self.other(symbol))

    def runs(self, text: str) -> str:
        """Hide the name written just before a ticker in brackets, "Apollo (APO)" or "Li Auto Inc. (NASDAQ: LI)",
        and keep it as a name of that company for the rest of the article. The tickers of one bracket are one
        company's, unless the article is scored for both."""
        changes = []
        for match in _TICKERS.finditer(text):
            indexes = [ord(mark[1]) - _FIRST for mark in re.findall(_ANY, match.group(0))]
            index = next((index for index in indexes if index < self.count), indexes[0])
            for other in set(indexes):
                if other >= self.count and other != index:  # Another listing of the same company.
                    changes.append((None, other, index))
            run = self._run(text, match.start(), index)
            if run is None:
                continue
            start, key, possessive, short = run[:4]
            # "(NYSE: EXK; TSX: EDR)" with both scored: the brackets stay, for the company the name is not.
            keep = any(other < self.count and other != index for other in indexes)
            for name in filter(None, (key, short)):
                if name not in self.keys:
                    self.local.setdefault(name, (index, ALWAYS, False, None))
            if key and len(key) > 1:
                self._first_word(key, index)
            changes.append((start, run[4], index, possessive, match, keep))
        merged = {change[1]: change[2] for change in changes if change[0] is None}
        pieces, position = [], 0
        for change in sorted((change for change in changes if change[0] is not None), key=lambda change: change[0]):
            start, end, index, possessive, match, keep = change
            if start < position:
                continue
            # The name becomes the mark and its tickers go; the brackets between them (a defined term) stay.
            between = text[end:match.start()]
            pieces += [text[position:start], _mark(index) + ("'s" if possessive else ""),
                       between if keep or between.strip(" \t,.") else ""]
            position = match.start() if keep else match.end()
        text = "".join(pieces) + text[position:]
        if merged:
            text = re.sub(_ANY, lambda mark: _mark(merged.get(ord(mark.group(0)[1]) - _FIRST,
                                                              ord(mark.group(0)[1]) - _FIRST)), text)
            self.local = {key: (merged.get(rule[0], rule[0]), *rule[1:]) for key, rule in self.local.items()}
        return text

    def _run(self, text: str, end: int, index: int):
        """The name written just before text[end], the bracket of a company's tickers: (its start, its words, whether
        it ends with 's, the short name a bracket gives it, where it ends), or None. Up to two brackets may stand
        between the name and the tickers: "Horizon Technology Finance Corporation (Horizon) (Nasdaq: HRZN)"."""
        stop, short = end, None
        for _ in range(2):
            between = re.search(r"[(\[]([^()\[\]]{1,120})[)\]][ \t,]*$", text[max(0, stop - 130):stop])
            if between is None:
                break
            stop -= len(between.group(0))
            inside = words(between.group(1))
            if 1 <= len(inside) <= 3 and all(_capital(word) for word, _, _ in inside) and _MARK not in between.group(1):
                short = tuple(lower(word) for word, _, _ in inside)
        low = max(0, stop - 200)
        pieces = [(piece.group(), piece.start(), piece.end()) for piece in _PIECE.finditer(text, low, stop)]
        if not pieces or text[pieces[-1][2]:stop].strip(" \t,."):
            return None  # Something else than a name stands before the brackets.
        first = len(pieces)
        for place in range(len(pieces) - 1, -1, -1):
            piece = pieces[place][0]
            if first < len(pieces) and not _joined(text, piece, pieces[place][2], *pieces[first][:2]):
                break
            mine = piece.startswith(_MARK) and ord(piece[1]) - _FIRST == index
            if not (mine or (not piece.startswith(_MARK) and (_capital(piece) or (
                    lower(piece) in LINKS and first < len(pieces)) or lower(piece) in LEGAL - LINKS))):
                break  # "Boralex inc.": a legal form written small is still the name's.
            first = place
        run = pieces[first:]
        while run and lower(run[0][0]) in LINKS:
            run = run[1:]
        marked = next((place for place, piece in enumerate(run) if piece[0].startswith(_MARK)), None)
        stop = pieces[-1][2]
        if lower(pieces[-1][0]) in LEGAL and text[stop:stop + 1] == ".":
            stop += 1  # The period of "Corp." is the name's.
        if marked is not None:  # "Deere & Company (DE)", Deere being replaced: the rest of the name goes with it.
            rest = [lower(piece[0]) for piece in run[marked + 1:] if not piece[0].startswith(_MARK)]
            if any(word.removesuffix("'s") not in LEGAL | LINKS | CORPORATE for word in rest):
                return None  # "Sculptor's Independent Board Members (NYSE: SCU)": not the name's.
            return run[marked][1], None, run[-1][0].endswith(("'s", "’s")), short, stop
        heads = set().union(*(self.aliases.heads.get(ticker, ()) for ticker in self.tickers[index]))
        start = next((place for place, piece in enumerate(run)
                      if lower(piece[0]) in heads or lower(piece[0]).removesuffix("'s") in heads), None)
        if start is None:  # A name the table lacks: the capitalized words after the last of a headline's.
            start = max((place + 1 for place, piece in enumerate(run)
                         if lower(piece[0]).removesuffix("'s") in _HEADLINE_ONLY), default=0)
            while start < len(run) and lower(run[start][0]) in LINKS:
                start += 1
        run = run[start:]
        security = SECURITY.search(text, run[0][1], run[-1][2]) if run else None
        if security is not None:  # "its Class A Common Stock (NYSE: X)" names no company.
            run = [piece for piece in run if piece[2] <= security.start()]
        if short is not None and not set(short) <= {lower(piece[0]) for piece in run}:
            short = None
        key = [lower(piece[0]) for piece in run]
        possessive = bool(key) and key[-1].endswith("'s") and len(key[-1]) > 2
        if possessive:
            key[-1] = key[-1][:-2]
        while len(key) > 1 and key[-1] in LEGAL:
            key.pop()
        key = key[:MAX_WORDS]  # A longer name is known by its first words.
        if not key or all(len(word) < 2 or word in LEGAL | LINKS | GENERIC for word in key) or (
                not heads & set(key) and (key[-1] in NOT_NAMES or any(word[0].isdigit() for word in key)
                                         or any(word in _MONTH_WORDS for word in key))):
            return None
        return run[0][1], tuple(key), possessive, short, stop

    def defined(self, text: str) -> str:
        """Keep the short names a press release defines after a company, ("Evergold" or the "Company"), as names of
        that company; what it calls itself in general (the "Company", "we") names nothing."""
        for match in _BRACKETS.finditer(text):
            if not re.search("[“\"„«]", match.group(2)):
                continue
            owner = _AFTER_NAME.search(text[max(0, match.start() - 240):match.start()])
            if owner is None:
                continue
            for the, term in _QUOTED.findall(match.group(2)):
                key = tuple(lower(word) for word, _, _ in words(term))
                key = key[1:] if key[:1] == ("the",) else key
                if the or not key or len(key) > MAX_WORDS or _MARK in term or SECURITY.search(term) or all(
                        word in GENERIC | LEGAL | LINKS for word in key):
                    continue
                if key not in self.keys:
                    self.local.setdefault(key, (ord(owner.group(1)) - _FIRST, ALWAYS,
                                                not any(map(_capital, term.split())), None))
        return text

    @staticmethod
    def defining(match: re.Match) -> str:
        """Drop the brackets of a defined term once they hold nothing but the company's placeholder and what a
        press release calls itself: A ("A" or the "Company") is A."""
        inside, before = match.group(2), match.string[max(0, match.start() - 240):match.start()]
        owner = _AFTER_NAME.search(before)
        if owner is None or not re.search("[“\"„«]", inside):
            return match.group(0)
        rest = _DEFINING.sub(" ", inside.replace(_MARK + owner.group(1), " "))
        return "" if all(lower(word) in GENERIC | {"société", "gesellschaft", "bolaget", "selskabet", "yhtiö"}
                         for word in rest.split()) else match.group(0)

    def render(self, text: str) -> str:
        """Write each mark as its placeholder: the scored companies first, then the others as the text names them."""
        order: dict[int, int] = {}
        for mark in re.findall(_ANY, text):
            index = ord(mark[1]) - _FIRST
            if index >= self.count:
                order.setdefault(index, self.count + len(order))
        return re.sub(_ANY, lambda mark: label(order.get(ord(mark.group(0)[1]) - _FIRST,
                                                         ord(mark.group(0)[1]) - _FIRST)), text)


def mask(title: str | None, description: str | None, tickers, aliases: Aliases | None = None,
         published: pd.Timestamp | None = None, others: dict[str, str] | None = None) -> tuple[str, str, list[str]]:
    """Return the title and the summary with their companies and their dates hidden, and the placeholder that stands
    for each ticker (tickers of one company share theirs; the other companies of the text get the placeholders
    after them). published, the article's day, leaves out the aliases of a company that got the ticker later.
    others maps other symbols of the tickers' companies to their ticker (JPM's preferred shares JPMpC, SOUN's
    warrants SOUNW): they are hidden with it."""
    aliases, tickers = aliases or Aliases(), [str(ticker) for ticker in tickers]
    if len(tickers) > 256:
        raise ValueError("mask() hides at most 256 companies in one article.")
    published = _day(published)
    indexes: list[int] = []
    for position, ticker in enumerate(tickers):
        same = next((indexes[earlier] for earlier in range(position)
                     if aliases.same_company(tickers[earlier], ticker, published)), None)
        indexes.append(max(indexes, default=-1) + 1 if same is None else same)
    # Private-use characters mark the placeholders; a trademark sign or a hard space parts a name from the next word.
    title, description = (SIGNS.sub("", _PRIVATE.sub("", text or "")).translate(SPACES).replace("''", "'")
                          for text in (title, description))
    description = DATELINE.sub("", description, count=1)
    text = _Article(aliases, published, tickers, indexes, others or {}).hide(title + _SPLIT + description)
    if text.count(_SPLIT) == 1:
        title, description = text.split(_SPLIT)
    else:  # Cannot happen: nothing above removes the split. If it does, each part is masked on its own.
        title, description = (_Article(aliases, published, tickers, indexes, others or {}).hide(part).replace(
            _SPLIT, "") for part in (title, description))
    description = INNER_DATELINE.sub("", strip_dates(description), count=2)
    return strip_dates(title), description, [label(index) for index in indexes]


def use(kind: str, length: int, articles: int, tagged: int, alone: int, lowercase: int, seen: bool,
        of_ticker: int, of_all: int, small: int | None = None, alias: str = "") -> str:
    """Where an alias is hidden: ALWAYS, ALONE (unless a word it forms another name with touches it) or NEVER. It is
    a `kind` alias of `length` words that `articles` articles contain on its own, written as a name; `tagged` of
    them are tagged with its ticker, `alone` of those with the alias standing alone. `lowercase` of the ticker's
    of_ticker articles, of of_all, contain it without capitals, as `small` articles do in all; seen tells whether
    the news contain it at all."""
    if not seen:
        return ALWAYS  # The news never contain it.
    if small is not None and lowercase >= MIN_SMALL and lowercase >= SMALL_SHARE * small:
        # Written small, it is still its ticker's (lululemon, e.l.f. Beauty): those articles count as the others.
        articles, tagged, alone, lowercase = articles + small, tagged + lowercase, alone + lowercase, 0
    if articles == 0:  # Only inside a longer alias of its ticker ("Coca" in Coca-Cola), or never written as a name.
        return ALWAYS if kind == "name" and length > 1 and lowercase == 0 else NEVER
    elsewhere = (articles - tagged) / max(of_all - of_ticker, 1)
    if kind == "ticker":  # Most of the many articles that write it use it for something else: a word (AI, IT, TX).
        # A ticker of five letters or more is a word only if the news write it small too (alert, smart): AIQUY
        # and WMMVY are tickers, however many comparisons name them without their tag.
        rare = articles - tagged < MIN_WORD or (len(alias) >= 5 and (small or 0) < MIN_WORD)
        return ALWAYS if tagged >= SHARES[kind] * articles or rare else NEVER
    if kind == "initials":
        return ALWAYS if tagged >= SHARES[kind] * articles else NEVER
    if length == 1 and kind != "name" and alias in PLACES:
        return NEVER  # Taiwan for Taiwan Semiconductor, Rio for Rio Tinto: the place, mostly.
    if tagged < CAPITAL_SHARE * (tagged + lowercase):
        return NEVER  # A common word in the ticker's own articles: gold, quantum computing.
    if length > 1:
        return ALWAYS if kind == "name" or tagged >= PHRASE_SHARE * articles else NEVER
    if tagged >= SHARES["start" if kind == "name" else kind] * articles:
        return ALWAYS
    if kind not in NAMES:
        return NEVER
    # A one-word name that most articles write without its ticker's tag: Twitter, Target, Dow, 3M, Apollo, Gold,
    # Now. Next to another capitalized word it is often something else: Dow Jones, 3M Ago, US Bancorp.
    if kind != "name" and (tagged < SHARES["name"] * articles or tagged / max(of_ticker, 1) < NAME_LIFT * elsewhere):
        return NEVER  # A first word (Apollo) needs to be its company's more than other articles'; a full name not.
    return ALONE if alone and alone >= CAPITAL_SHARE * (alone + lowercase) else NEVER


def wide(kind: str, alias: str, how: str, articles: int, tagged: int, small: int, anywhere: int,
         inside: int) -> str:
    """Where an alias is hidden in the articles that are not tagged with its ticker: ALWAYS, ALONE (where no
    capitalized word touches it) or "" for nowhere. `anywhere` articles contain it as a name, `inside` of them only
    inside a longer alias of some ticker, and `small` contain it without capitals; the rest as for use()."""
    if how != ALWAYS:
        return ""
    if kind == "ticker":
        return ALWAYS if len(alias) > 2 and articles and tagged >= WIDE_TICKER * articles else ""
    if kind not in NAMES:
        return ""
    if " " in alias:
        if small > WIDE_SMALL * max(anywhere, 1):
            return ""
        if not anywhere:
            return ALWAYS if kind == "name" else ""  # A full name the news never wrote yet.
        known = kind != "seen" or tagged >= PLAIN_SHARE * articles
        return ALWAYS if known and tagged >= max(1, WIDE_PHRASE * articles) and inside <= WIDE_INSIDE * anywhere else ""
    if not articles or small > WIDE_SMALL * anywhere or tagged < WIDE_WORD * articles or len(alias) < 3:
        return ""  # A word the news write small too (Block, Gap), or mostly about something else (Conn, Chile).
    return ALWAYS if tagged >= PLAIN_SHARE * articles else ALONE


def build_aliases(names: pd.DataFrame, articles: list) -> pd.DataFrame:
    """Return the alias table (ALIAS_COLUMNS) from Polygon's names (columns ticker, name, first_date) and the news,
    a list of (title, description, tickers) or (title, description, tickers, day). A row is a ticker's alias of one
    kind. articles counts the articles that contain the alias on its own, written as a name; tagged, those of them
    tagged with the ticker (or another ticker of its company); alone, those of the tagged where the alias stands
    alone; lowercase, the ticker's articles that contain it without capitals. use tells where mask() hides it in
    the ticker's articles, any_case whether written small too, apart beside which words not, wide where in the
    other articles, and since from when."""
    texts = [(f"{article[0] or ''}\n{article[1] or ''}",
              {str(ticker) for ticker in ([] if article[2] is None else article[2])},
              _day(article[3]) if len(article) > 3 else None) for article in articles]

    # Names the news write before a ticker in parentheses, with the first day each was written.
    seen, mentions, early = Counter(), Counter(), {}
    for text, tags, day in texts:
        for pair in set(seen_names(text, tags)):
            mentions[pair[0]] += 1
            seen[pair] += 1
            if day is not None and not early.get(pair, day) < day:
                early[pair] = day

    owners: dict[tuple[str, ...], dict[str, dict]] = defaultdict(dict)  # Alias -> {ticker: {kind: since}}.
    full = defaultdict(set)
    first_dates = names.first_date if "first_date" in names else pd.Series([pd.NaT] * len(names), index=names.index)
    # The names of the first date read were the ticker's before it: they have no start. Polygon's over-the-counter
    # names start a year after the others, so the first date is each market's.
    earliest = first_dates.groupby(names.market).transform("min") if "market" in names else pd.Series(
        [first_dates.min()] * len(names), index=names.index)
    for ticker, name, first_date, first in zip(names.ticker.astype(str), names.name, first_dates, earliest):
        day = pd.NaT if pd.isna(first_date) or first_date == first else first_date
        found = name_aliases(name)
        if not pd.isna(day):
            # Polygon gives some names late (LabCorp, Polestar) or misspells them for years (Nextra Energy): a name
            # the news wrote before the ticker in parentheses, at least twice, applies from the first time.
            written = [early[ticker, key] for key, kind in found.items()
                       if kind in ("name", "start") and seen[ticker, key] >= 2 and (ticker, key) in early]
            day = min([day, *written])
        for key, kind in found.items():
            kinds = owners[key].setdefault(ticker, {})
            kinds[kind] = day if kind not in kinds else pd.NaT if pd.isna(day) or pd.isna(kinds[kind]) else min(
                kinds[kind], day)
            if kind == "name":
                full[ticker].add(key)
    by_name = defaultdict(set)
    for ticker, keys in full.items():
        for key in keys:
            by_name[key].add(ticker)
    company = {ticker: set().union(*(by_name[key] for key in keys)) for ticker, keys in full.items()}

    # The names seen become aliases, unless they only add words before a known alias.
    for (ticker, key), count in sorted(seen.items(), key=lambda item: -item[1]):
        if count < max(MIN_SEEN, MIN_SEEN_SHARE * mentions[ticker]) or key[0] in LINKS or len(key[0]) < 2:
            continue
        ends = set().union(*(owners.get(key[start:], {}).get(ticker, {}) for start in range(1, len(key))))
        own = ends == {"word"} and count >= OWN_SEEN_SHARE * mentions[ticker]
        if (not ends or own) and ticker not in owners[key]:
            owners[key][ticker] = {"seen": pd.NaT}
    for ticker, brands in BRANDS.items():
        for brand in brands:
            owners[tuple(brand.split(" "))].setdefault(ticker, {"brand": pd.NaT})

    symbols = {ticker for ticker in names.ticker.astype(str) if len(ticker) >= 2}
    starts = _starts(owners)
    anywhere, in_capitals, inside, tagged, tagged_capitals, alone, lowercase, small_anywhere, of_ticker, enclosed = (
        Counter() for _ in range(10))  # small_anywhere: the articles that contain the alias without capitals.
    small_symbol = Counter()  # The articles that write a ticker of five letters or more as a word, small.
    for text, tags, _ in texts:
        found = words(text)
        tickers = set().union(*(company.get(tag, {tag}) for tag in tags)) if tags else set()
        of_ticker.update(tickers)
        present, places, small = [], defaultdict(list), set()
        for first, last, key, _, capital in _find(text, found, owners, starts, any_case=True):
            if capital:
                present.append((first, last, key))
                places[key].append((first, last, all(word.isupper() for word, _, _ in found[first:last])))
            else:
                small.add(key)
        for key in small:
            small_anywhere[key] += 1
            for ticker in tickers & owners[key].keys():
                lowercase[key, ticker] += 1
        for key, spots in places.items():
            anywhere[key] += 1
            capitals = any(spot[2] for spot in spots)
            in_capitals[key] += capitals
            # The tickers with a longer alias around every one of the alias's places: "Motor" in "Ford Motor".
            around, apart, within = None, set(), True
            for first, last, _ in spots:
                longer = {ticker for other_first, other_last, other in present
                          if other_first <= first and last <= other_last and other != key for ticker in owners[other]}
                around = longer if around is None else around & longer
                within = within and bool(longer - owners[key].keys())  # Inside another company's name.
                if len(key) == 1 and _alone(text, found, first):
                    apart |= {ticker for ticker in tickers & owners[key].keys() if ticker not in longer}
            enclosed[key] += within
            for ticker in (around or set()) & owners[key].keys():
                inside[key, ticker] += 1
            for ticker in tickers & owners[key].keys():
                if ticker not in (around or set()):
                    tagged[key, ticker] += 1
                    tagged_capitals[key, ticker] += capitals
                    alone[key, ticker] += ticker in apart
        small_symbol.update({word.upper() for word, _, _ in found if len(word) >= 5 and word.islower()} & symbols)
        present, written = {word for word, _, _ in found} & symbols, set()
        if present:
            written = {symbol for match in LISTED.finditer(text) for symbol in re.split(r"\s*[,/]\s*", match.group(1))}
            written |= {match.group(1) for match in CASHTAG.finditer(text)}
        for ticker in present:
            anywhere[ticker] += 1
            tagged[ticker] += ticker in tags or ticker in written

    rows, named = [], set()
    for key, tickers in owners.items():
        for ticker, kinds in tickers.items():
            for kind, since in kinds.items():
                initials = kind == "initials"  # Counted where the text writes them in capitals.
                numbers = ((in_capitals if initials else anywhere)[key] - inside[key, ticker],
                           (tagged_capitals if initials else tagged)[key, ticker], alone[key, ticker],
                           lowercase[key, ticker])
                seen_at_all = (in_capitals[key] if initials else anywhere[key] + small_anywhere[key]) > 0
                how = use(kind, len(key), *numbers, seen_at_all, of_ticker[ticker], len(texts), small_anywhere[key],
                          " ".join(key))
                small = numbers[3] >= MIN_SMALL and numbers[3] >= SMALL_SHARE * small_anywhere[key]
                if how != NEVER and kind in ("name", "start") and key == (ticker.lower(),):
                    named.add(ticker)  # UBS, MSCI, HSBC: the ticker is the name.
                rows.append((ticker, " ".join(key), kind, *numbers, how, since, how != NEVER and small and not initials,
                             wide(kind, " ".join(key), how, numbers[0], numbers[1], small_anywhere[key], anywhere[key],
                                  enclosed[key]), ""))
    for ticker in sorted(symbols):
        how = ALWAYS if ticker in named else use("ticker", 1, anywhere[ticker], tagged[ticker], 0, 0,
                                                 anywhere[ticker] > 0, of_ticker[ticker], len(texts),
                                                 small_symbol[ticker], ticker)
        spread = ALWAYS if ticker in named and len(ticker) >= 3 else wide(  # UBS, HSBC: a name in any article.
            "ticker", ticker, how, anywhere[ticker], tagged[ticker], 0, anywhere[ticker], 0)
        rows.append((ticker, ticker, "ticker", anywhere[ticker], tagged[ticker], 0, 0, how, pd.NaT, False, spread, ""))
    table = pd.DataFrame(rows, columns=ALIAS_COLUMNS).sort_values(["ticker", "alias", "kind"], ignore_index=True)
    table["since"] = pd.to_datetime(table.since)
    return _apart(table, texts, company, of_ticker)


def _apart(table: pd.DataFrame, texts: list, company: dict, of_ticker: Counter) -> pd.DataFrame:
    """Fill apart for the aliases hidden ALONE: the words beside which the alias is another name, and stays. They
    are the capitalized words that touch it in the sentences of the other articles (at least MIN_APART of them)
    nearly as often as in the ticker's own, ">jones" for Dow and ">hospitality" for Target; and the words that go
    with it where it is written small, MIN_PHRASE of those articles and PHRASE_RATE of them: "price<" and ">date"
    for Target, which a headline writes Price Target and Target Date."""
    rows = table[(table.use == ALONE) & ~table.alias.str.contains(" ")]
    tickers_of = defaultdict(set)
    for ticker, alias in zip(rows.ticker, rows.alias):
        tickers_of[alias].add(ticker)
    beside, beside_tagged, phrases, small = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), Counter()
    for text, tags, _ in texts:
        found = words(text)
        tickers = set().union(*(company.get(tag, {tag}) for tag in tags)) if tags else set()
        near, low = defaultdict(set), defaultdict(set)
        for index, (word, _, _) in enumerate(found):
            alias = lower(word)
            if alias not in tickers_of:
                continue
            if not _capital(word):
                low[alias].update(_beside(text, found, index))
            elif _sentence(found, index):  # In a headline every word has a capital: it tells nothing.
                near[alias].update(_near(text, found, index))
        for alias, touching in near.items():
            beside[alias].update(touching)
            for ticker in tickers & tickers_of[alias]:
                beside_tagged[alias, ticker].update(touching)
        for alias, touching in low.items():
            small[alias] += 1
            phrases[alias].update(touching)
    apart = []
    for ticker, alias, how in zip(table.ticker, table.alias, table.use):
        if how != ALONE or alias not in tickers_of:
            apart.append("")
            continue
        mine, others = beside_tagged.get((alias, ticker), {}), len(texts) - of_ticker[ticker]
        names = {word for word, count in beside.get(alias, {}).items()
                 if count - mine.get(word, 0) >= MIN_APART and (count - mine.get(word, 0)) / max(others, 1) * NAME_LIFT
                 >= mine.get(word, 0) / max(of_ticker[ticker], 1)}
        common = {word for word, count in phrases.get(alias, {}).items()
                  if count >= max(MIN_PHRASE, PHRASE_RATE * small[alias])}
        apart.append(" ".join(sorted(names | common)))
    table["apart"] = apart
    return table
