import unittest
import pandas as pd
from stock_recommender.anonymize import (ALIAS_COLUMNS, ALONE, ALWAYS, NEVER, Aliases, build_aliases, clean, label,
                                         mask, name_aliases, seen_names, strip_dates, use, wide, words)

A, B, C, D = "Company A", "Company B", "Company C", "Company D"


def table(*rows) -> pd.DataFrame:
    """An alias table of (ticker, alias, kind[, use[, since[, any case[, wide[, apart]]]]]) rows."""
    full = [(*row, *(ALWAYS, None, False, "", "")[len(row) - 3:]) for row in rows]
    frame = pd.DataFrame([(ticker, alias, kind, 0, 0, 0, 0, how, since, any_case, wide_, apart)
                          for ticker, alias, kind, how, since, any_case, wide_, apart in full], columns=ALIAS_COLUMNS)
    frame["since"] = pd.to_datetime(frame.since)
    return frame


class NamesTest(unittest.TestCase):
    def test_a_name_loses_its_security_and_its_legal_form(self):
        self.assertEqual(clean("Alphabet Inc. Class A Common Stock"), ("Alphabet", "Alphabet Inc"))
        self.assertEqual(clean("The Walt Disney Company"), ("Walt Disney", "Walt Disney Company"))
        self.assertEqual(clean("JPMorgan Chase & Co."), ("JPMorgan Chase", ""))
        self.assertEqual(clean("Taiwan Semiconductor Manufacturing Company Ltd.")[0],
                         "Taiwan Semiconductor Manufacturing")
        self.assertEqual(clean("Alibaba Group Holding Limited American Depositary Shares, each represents eight "
                               "Ordinary Shares")[0], "Alibaba Group Holding")
        self.assertEqual(clean("Unit Corporation")[0], "Unit")  # Not the description of a unit.
        # Over-the-counter names, as Polygon abbreviates them.
        self.assertEqual([clean(name)[0] for name in ("TENCENT HLDGS UNSP/ADR", "DANONE S/ADR", "ITV PLC UNSP/ADR",
                                                      "AAC TECHS HLDGS INC ORD", "W.P. Carey Inc. (REIT)")],
                         ["TENCENT HLDGS", "DANONE", "ITV", "AAC TECHS HLDGS", "W.P. Carey"])

    def test_the_aliases_of_a_name(self):
        ford = name_aliases("Ford Motor Company")
        self.assertEqual(ford, {("ford", "motor"): "name", ("ford", "motor", "company"): "name",
                                ("ford", "motor", "co"): "name", ("fordmotor",): "name", ("ford",): "start",
                                ("motor",): "word", ("fm",): "initials", ("fmc",): "initials"})
        self.assertEqual(name_aliases("Procter & Gamble Company")[("p&g",)], "initials")
        self.assertIn(("procter", "and", "gamble"), name_aliases("Procter & Gamble Company"))
        self.assertEqual(name_aliases("Amazon.Com Inc")[("amazon",)], "name")
        self.assertEqual(name_aliases("United States Steel Corporation")[("u.s", "steel")], "name")
        self.assertNotIn(("bank", "of"), name_aliases("Bank of America Corporation"))  # No alias ends on a link.
        self.assertNotIn(("inc",), name_aliases("Inc"))
        self.assertIn(("fubo", "tv"), name_aliases("fuboTV Inc."))  # As the news write it too.
        self.assertIn(("w", "p", "carey"), name_aliases("W.P. Carey Inc. (REIT)"))
        self.assertEqual(name_aliases("TENCENT HLDGS UNSP/ADR")[("tencent",)], "start")

    def test_words(self):
        self.assertEqual([word for word, _, _ in words("AT&T's U.S. unit, Lowe’s and d’Innate Pharma; Itaú (ITUB)")],
                         ["AT&T's", "U.S", "unit", "Lowe’s", "and", "Innate", "Pharma", "Itaú", "ITUB"])

    def test_names_the_news_write_before_a_ticker(self):
        text = ("Shares of Coke (KO) rose. U.S. Steel (NYSE: X) and Bank of America Corp. (BAC) fell, said Smith. "
                "Alphabet's (GOOGL) cloud grew; the retailer (TGT) did not. Apple (MSFT rival) too.")
        self.assertEqual(seen_names(text, ["KO", "X", "BAC", "GOOGL", "TGT", "MSFT"]),
                         [("KO", ("coke",)), ("X", ("u.s", "steel")), ("BAC", ("america",)), ("GOOGL", ("alphabet",))])
        self.assertEqual(seen_names("Coke (KO) rose.", ["PEP"]), [])  # KO is not a tag of the article.

    def test_placeholders(self):
        self.assertEqual([label(index) for index in (0, 1, 25, 26, 27)],
                         [A, B, "Company Z", "Company AA", "Company AB"])


class DatesTest(unittest.TestCase):
    def test_years_are_replaced(self):
        self.assertEqual(strip_dates("In 2023, FY2024 guidance beat Q3'22 and 1Q23; FY23 and F3Q22 too. Since '21, "
                                     "the 2020s and the '90s; founded in 1957, net zero by 2050; the 2024-25 season."),
                         "In 20XX, FY20XX guidance beat Q3 20XX and 1Q 20XX; FY 20XX and F3Q 20XX too. Since 'XX, "
                         "the 20XXs and the 'XXs; founded in 19XX, net zero by 20XX; the 20XX-XX season.")
        kept = "It paid $2023 for 2,023 shares, up 2024% to 2030.5; the Russell 2000 and 2000 employees, 0.2025 each."
        self.assertEqual(strip_dates(kept), kept)
        codes = "The H100 and H200 chips meet AEC-Q200 and Q101; Q4 results."
        self.assertEqual(strip_dates(codes), codes)
        self.assertEqual(strip_dates("Beats, Hikes'21 EPS View; Oslo, 25 May– the bank"),
                         "Beats, Hikes'XX EPS View; Oslo, [date]– the bank")
        self.assertEqual(strip_dates("Ford May Report Losses; Tesla May Deliveries; the Mar. CPI; 30 06 23; "
                                     "Semaine 23; the Fourth of July and July Fourth; FY29E; the 67th American "
                                     "Society of Hematology; from June 9 to the 11th, 2024; as of March, 31, 2023"),
                         "Ford May Report Losses; Tesla [month] Deliveries; the [month] CPI; [date]; [date]; [date] "
                         "and [date]; FY 20XXE; the American Society of Hematology; from [date] to the [date]; as of "
                         "[date]")
        self.assertEqual(strip_dates("Phase 3 MARCH study; Shares March Higher; the 1,000th patient; the 34TH ANNUAL "
                                     "Parade; May payrolls"),
                         "Phase 3 [month] study; Shares March Higher; the 1,000th patient; the ANNUAL Parade; [month] "
                         "payrolls")

    def test_days_and_months_are_replaced(self):
        for text, expected in [
                ("for the quarter ended September 30, 2023", "for the quarter ended [date]"),
                ("on Tuesday, March 14,2023 and Dec.14, 2021", "on Tuesday, [date] and [date]"),
                ("Sept. 9-11, 2024; June 26 and 27; Jan 7th-9th, 2025", "[date]; [date]; [date]"),
                ("Best Stocks to Buy for April 9th", "Best Stocks to Buy for [date]"),
                ("3 March 2025, the 31st of March and 15 Nov.", "[date], the [date] and [date]"),
                ("in March 2023 and Sept. of 2021", "in [date] and [date]"),
                ("on 3/15/2023, 10.9.2025 and 2023-03-15; on 3/15", "on [date], [date] and [date]; on [date]"),
                ("le 28 février 2023, am 14. Januar 2025, 15 de marzo de 2023", "le [date], am [date], [date]"),
                ("预注册将于9月21日开始", "预注册将于[date]开始"),
                ("in March, the December quarter and mid-May", "in [month], the [month] quarter and mid-[month]"),
                ("PAYABLE IN AUGUST; sales in May and June; by Oct.", "PAYABLE IN [month]; sales in [month] and "
                                                                    "[month]; by [month]"),
                ("at the 40th Annual Healthcare Conference", "at the Annual Healthcare Conference")]:
            self.assertEqual(strip_dates(text), expected)

    def test_what_is_not_a_date_stays(self):
        for kept in ["This May Be It: 3 Stocks That May Soar", "Costs to Mar Q4 Earnings at Mar-a-Lago",
                     "last year, next quarter, this week, Q3 and the fourth quarter", "on Monday at 8:30 a.m. ET",
                     "Phase 1/2 trial, a 60/40 portfolio, 24/7", "version v1.2.21 at 10.9.20.1",
                     "May 5% and 10 May Be Enough"]:
            self.assertEqual(strip_dates(kept), kept)

    def test_the_dateline_of_a_press_release_goes(self):
        for start in ["DUNMORE, Pa., April  21, 2023  (GLOBE NEWSWIRE) -- ",
                      "VANCOUVER, British Columbia and ORLANDO, Fla., June 24, 2024 (GLOBE NEWSWIRE) -- ",
                      "NEW YORK, April 21, 2023 /PRNewswire/ -- ", "SAN FRANCISCO--(BUSINESS WIRE)--",
                      "TORONTO, ON / ACCESSWIRE / April 21, 2023 / "]:
            self.assertEqual(mask("Results", start + "Ford said sales rose.", [])[1], "Ford said sales rose.")
        title = "A title the summary repeats, with a subtitle long enough that the dateline is not at its start " * 2
        inside = title + "-- MIAMI BEACH, Fla., Aug. 18, 2022 (GLOBE NEWSWIRE) -- "
        self.assertEqual(mask(None, inside + "Ford said.", [])[1], title + "-- Ford said.")
        for start in ["Maroussi, Greece – May 14, 2025 – ", "MAINZ, Germany, October 16, 2023 – ",
                      "ATLANTA, GA, June 28, 2023 (GLOBE NEWSWIRE) -- via NewMediaWire – "]:
            self.assertEqual(mask(None, start + "Ford said.", [])[1], "Ford said.")
        text = "Ford said (GLOBE NEWSWIRE) -- is a wire."  # Not a dateline: it stays.
        self.assertEqual(mask(None, "Sales rose in the quarter and the year, the company's best. " * 3 + text, [])[1],
                         "Sales rose in the quarter and the year, the company's best. " * 3 + text)


class MaskTest(unittest.TestCase):
    def setUp(self):
        self.aliases = Aliases(table(
            ("F", "ford motor", "name"), ("F", "ford motor co", "name"), ("F", "ford", "start"),
            ("GM", "general motors", "name"), ("GM", "gm", "initials"), ("GM", "GM", "ticker"),
            ("TGT", "target", "name", ALONE, None, False, "", "price< >date"), ("TGT", "target corp", "name"),
            ("TGT", "TGT", "ticker"),
            ("MCD", "mcdonald's", "name"), ("GOOGL", "alphabet", "name"), ("GOOG", "alphabet", "name"),
            ("GOLD", "barrick gold", "name"), ("GOLD", "gold", "name", ALONE, "2026-01-01"),
            ("RKLB", "rocket lab", "start"), ("RKLB", "rocket lab", "name", ALWAYS, "2025-07-01"),
            ("AI", "c3.ai", "name"), ("AI", "AI", "ticker", NEVER),
            ("EDR", "endeavour silver", "seen"), ("EXK", "endeavour silver", "name"),
            ("SQ", "block inc", "name", ALWAYS, "2022-01-03"), ("XYZ", "block inc", "name", ALWAYS, "2025-04-01"),
            ("SIX", "six flags entertainment corp", "name"),
            ("FUN", "six flags entertainment corp", "name", ALWAYS, "2024-07-01"), ("FUN", "cedar fair", "name"),
            ("OWL", "blue owl capital", "name", ALWAYS, "2021-07-01"), ("OWL", "blue owl capital inc", "name"),
            ("OBDC", "blue owl capital", "name", ALWAYS, "2023-07-03"), ("OBDC", "blue owl capital corp", "name"),
            ("INDB", "independent bank corp", "name"), ("IBCP", "independent bank corp", "name"),
            ("BRK.A", "berkshire hathaway", "name"), ("BRK.B", "berkshire hathaway", "name"),
            ("TSLL", "direxion shares etf trust direxion daily", "name"),
            ("TSLS", "direxion shares etf trust direxion daily", "name"),
            ("LULU", "lululemon athletica", "name", ALWAYS, None, True), ("LULU", "lululemon", "start", ALWAYS, None, True),
            # Other companies: hidden in any article where the table says so.
            ("MS", "morgan stanley", "name", ALWAYS, None, False, ALWAYS), ("MS", "MS", "ticker", NEVER),
            ("AAPL", "apple", "name", ALWAYS, None, False, ALWAYS), ("AAPL", "AAPL", "ticker", ALWAYS, None, False, ALWAYS),
            ("BC", "brunswick", "name", ALWAYS, None, False, ALONE), ("BC", "BC", "ticker", NEVER),
            ("LNN", "lindsay", "start", NEVER), ("LNN", "LNN", "ticker"), ("AMD", "advanced micro devices", "name"),
            ("AMD", "AMD", "ticker"), ("T", "at&t", "name"), ("O", "realty income", "name"),
            ("LLY", "eli lilly", "name"), ("BP", "bp", "name"), ("BP", "BP", "ticker"), ("TECK", "teck resources", "name"),
            ("MDNA", "medicenna therapeutics", "name"), ("ABVC", "abvc biopharma", "name"),
            ("BYD", "BYD", "ticker", NEVER), ("NVDA", "NVDA", "ticker", ALWAYS, None, False, ALWAYS),
            ("ATO", "ATO", "ticker", ALWAYS, None, False, ALWAYS), ("CEO", "CEO", "ticker", NEVER),
            ("SRAFF", "sandfire resources america", "name"), ("M", "macy's", "name"),
            ("BLX", "bladex", "name"), ("BLX", "BLX", "ticker"), ("ARES", "ares management", "name"),
            ("WBD", "warner bros discovery", "name", ALWAYS, None, False, ALWAYS),
            ("C", "citigroup", "name", ALONE, None, False, ALONE),
            ("RTX", "RTX", "ticker", ALWAYS, None, False, ALWAYS), ("CGI", "CGI", "ticker", ALWAYS, None, False, ALWAYS),
            ("ATKR", "atkore", "name"), ("XPLR", "xplr infrastructure", "name"),
            ("NEP", "nextera energy partners", "name", ALWAYS, None, False, ALWAYS), ("CAKE", "cheesecake factory", "name"),
            ("TJX", "TJX", "ticker"), ("XLK", "technology select sector spdr", "name"), ("XLK", "XLK", "ticker"),
            ("SHOP", "SHOP", "ticker"), ("FLHLF", "FLHLF", "ticker"), ("FH", "FH", "ticker"),
            ("RDDT", "reddit", "name"), ("SCU", "sculptor", "name"), ("VET", "vermilion energy", "name"),
            ("TCBPY", "tc biopharm hldgs", "name"), ("TCBP", "tc biopharm", "name", ALWAYS, None, False, ALWAYS),
            ("FIS", "fidelity national", "name", ALWAYS, None, False, ALWAYS),
            ("FNF", "fidelity national", "name", ALWAYS, None, False, ALWAYS)))

    def hide(self, text, tickers, **options):
        return mask(None, text, tickers, self.aliases, **options)[1]

    def test_names_and_tickers_become_placeholders(self):
        title, summary, labels = mask("Ford (F) and General Motors Lift Outlooks",
                                      "Ford Motor Co. (NYSE: F) and GM (NYSE: GM) raised guidance. Ford's CEO and "
                                      "ford.com were not asked; $F rose, $GM too.", ["F", "GM"], self.aliases)
        self.assertEqual(labels, [A, B])
        self.assertEqual(title, f"{A} and {B} Lift Outlooks")
        self.assertEqual(summary, f"{A} and {B} raised guidance. {A}'s CEO and [link] were not asked; {A} rose, "
                                  f"{B} too.")

    def test_a_ticker_that_is_also_a_word_is_hidden_only_where_written_as_a_ticker(self):
        self.assertEqual(self.hide("F is a grade. Plan F (F) won. Ford (F) and Ford's [F] rose; see (NYSE:F), $F.",
                                   ["F"]),
                         f"{A} is a grade. Plan F (F) won. {A} and {A}'s rose; see ({A}), {A}.")
        self.assertEqual(self.hide("C3.ai (AI) sells artificial intelligence (AI). AI rose; so did (AI, GM).",
                                   ["AI", "GM"]),
                         f"{A} sells artificial intelligence (AI). AI rose; so did ({A}, {B}).")
        # In a list of companies, beside "stock" or "shares", after "shares of" or "ticker", in quotes.
        self.assertEqual(self.hide("Buy T, F and GM; like T, O stock rose. Shares of F and F shares fell; ticker: F. "
                                   "The F-150 and Class F shares are not it. “F” is.", ["T", "F", "O", "GM"]),
                         f"Buy {A}, {B} and {D}; like {A}, {C} stock rose. Shares of {B} and {B} shares fell; "
                         f"ticker: {B}. The F-150 and Class F shares are not it. “{B}” is.")
        self.assertEqual(mask("Lawsuit Against Ford - F", "AI stock fell; this AI is new.", ["F", "AI"],
                              self.aliases)[:2], (f"Lawsuit Against {A} - {A}", f"{B} stock fell; this AI is new."))

    def test_a_one_letter_ticker_is_hidden_where_it_is_not_a_letter(self):
        self.assertEqual(self.hide("Click here for more on T. O's Dividend Stands Taller; help O maintain it. Plan B, "
                                   "Class C shares, Vitamin D, John F. Kennedy, E. coli, $5 M and T-Mobile; THE F "
                                   "WORD IS HERE.", ["T", "O", "B", "C", "D", "F", "E", "M"]),
                         f"Click here for more on {A}. {B}'s Dividend Stands Taller; help {B} maintain it. Plan B, "
                         f"Class C shares, Vitamin D, John F. Kennedy, E. coli, $5 M and T-Mobile; THE F WORD IS HERE.")
        self.assertEqual(self.hide("T earnings call for the period; T Q4 results; E or FUPBY: Which Is Better?",
                                   ["T", "E"]), f"{A} earnings call for the period; {A} Q4 results; {B} or {C}: "
                                                f"Which Is Better?")

    def test_a_word_like_ticker_in_brackets_unless_the_words_before_spell_it(self):
        self.assertEqual(self.hide("Is (BYD) Outperforming? artificial intelligence (AI) and Plan F (F).",
                                   ["BYD", "AI", "F"]), f"Is ({A}) Outperforming? artificial intelligence (AI) and "
                                                        f"Plan F (F).")
        # Another company's ticker that is nearly always one goes from brackets, unless the words spell it.
        self.assertEqual(self.hide("the chipmaker (NVDA) and account takeover (ATO) fraud", ["F"]),
                         f"the chipmaker ({B}) and account takeover (ATO) fraud")

    def test_another_companys_ticker_in_a_list_with_the_articles_companies(self):
        self.assertEqual(self.hide("Ford, GM and AAPL rose; so did ZZZZ and Ford, unlike CEO and Ford.", ["F"]),
                         f"{A}, {B} and {C} rose; so did ZZZZ and {A}, unlike CEO and {A}.")

    def test_a_ticker_is_hidden_wherever_it_stands(self):
        # With its class of shares, at the start of a product's code, and without any row in the table.
        self.assertEqual(self.hide("TECK.B and TECK rose (TSX: TECK.A); MDNA11 and ABV-1505 too. TELNY, an ADR.",
                                   ["TECK", "MDNA", "ABVC", "TELNY"]),
                         f"{A} and {A} rose ({A}); {B}-11 and {C}-1505 too. {D}, an ADR.")
        self.assertEqual(self.hide("BRK-B, BRK/A and BRK.B rose.", ["BRK.A", "BRK.B"]), f"{A}, {A} and {A} rose.")

    def test_brackets_of_tickers(self):
        self.assertEqual(self.hide("General Motors (NYSE: GM, TSX: GMM) and Ford Motor Co. (F; XETRA: FMC1) rose.",
                                   ["GM", "F"]), f"{A} and {B} rose.")  # Its other listings are the company too.
        self.assertEqual(self.hide("Ford's (F) sales, Ford (“Ford”) (F) (NYSE: F), (see GM) and (GM).", ["F", "GM"]),
                         f"{A}'s sales, {A}, (see {B}) and ({B}).")
        self.assertEqual(self.hide("Ford Motor's Ford (NYSE: F) unit and Ford's GM stake.", ["F", "GM"]),
                         f"{A} unit and {A}'s {B} stake.")
        self.assertEqual(self.hide("Ford (NYSE - F), Ford Motor (CUSIP: 345370860; ISIN: US3453708600) and "
                                   "GM (Euronext Paris : GMP ; Nasdaq : GM) fell.", ["F", "GM"]),
                         f"{A}, {A} ([code]; [code]) and {B} fell.")

    def test_of_two_tickers_the_one_named_so_has_the_alias(self):
        text = "Endeavour Silver Corp. (NYSE: EXK; TSX: EDR) released results."  # EDR is another company's ticker.
        self.assertEqual(self.hide(text, ["EDR", "EXK"]), f"{B} ({B}; {A}) released results.")

    def test_an_alias_needs_capitals(self):
        self.assertEqual(self.hide("We ford the river. FORD and Ford rose; gm of flour, Gm, GM.", ["F", "GM"]),
                         f"We ford the river. {A} and {A} rose; gm of flour, Gm, {B}.")
        # Unless its own articles write it small.
        self.assertEqual(self.hide("lululemon athletica inc. (LULU) fell; lululemon too.", ["LULU"]),
                         f"{A} fell; {A} too.")

    def test_a_common_word_is_a_name_unless_it_touches_a_word_it_forms_another_name_with(self):
        text = ("Target (TGT) got a new Price Target and a Target Date. Target said TGT's sales rose; Target CEO too. "
                "Does Target Have A Plan?")
        self.assertEqual(self.hide(text, ["TGT"]),
                         f"{A} got a new Price Target and a Target Date. {A} said {A}'s sales rose; {A} CEO too. "
                         f"Does {A} Have A Plan?")

    def test_a_name_with_its_own_possessive(self):
        self.assertEqual(self.hide("McDonald's sales rose. McDonald's's? Ford's too.", ["MCD", "F"]),
                         f"{A} sales rose. {A}'s? {B}'s too.")
        self.assertEqual(self.hide("Target Corp.'s sales and Ford Motor Co. The end.", ["TGT", "F"]),
                         f"{A}'s sales and {B}. The end.")
        self.assertEqual(self.hide("General Motors' (GM) revenue and General Motors’ plan.", ["GM"]),
                         f"{A}'s revenue and {A}'s plan.")

    def test_what_follows_a_name_goes_with_it(self):
        self.assertEqual(mask("Ford Motor Co. Investors Have Opportunity", "Eli Lilly and Company (LLY) and BP p.l.c. "
                              "(BP) rose; Ford, inc (F) and Ford Motor Co., Ltd. II fell. Ford® too.", ["F", "LLY", "BP"],
                              self.aliases)[:2],
                         (f"{A} Investors Have Opportunity", f"{B} and {C} rose; {A} and {A} fell. {A} too."))

    def test_tickers_of_one_company_share_a_placeholder(self):
        title, summary, labels = mask("Alphabet (GOOGL, GOOG) vs Ford", None, ["GOOGL", "F", "GOOG"], self.aliases)
        self.assertEqual((title, labels), (f"{A} vs {B}", [A, B, A]))
        same = lambda first, second, day=None: self.aliases.same_company(first, second, day and pd.Timestamp(day))
        self.assertTrue(same("BRK.A", "BRK.B"))  # Two classes of shares.
        self.assertTrue(same("SQ", "XYZ", "2025-06-01"))  # A new ticker: both are Block, Inc. then.
        self.assertFalse(same("SQ", "XYZ", "2024-06-01"))
        self.assertFalse(same("FUN", "SIX", "2023-11-02"))  # Cedar Fair and Six Flags, before they merged.
        self.assertTrue(same("FUN", "SIX", "2024-08-01"))
        self.assertFalse(same("OWL", "OBDC", "2024-01-05"))  # Blue Owl Capital Inc. and Corporation.
        self.assertFalse(same("INDB", "IBCP"))  # Two banks of one name, always.
        self.assertFalse(same("TSLL", "TSLS"))  # A name cut short: a bull fund and a bear fund.
        merger = mask("Cedar Fair and Six Flags agree to merge", None, ["FUN", "SIX"], self.aliases,
                      pd.Timestamp("2023-11-02", tz="UTC"))
        self.assertEqual((merger[0], merger[2]), (f"{A} and Six Flags agree to merge", [A, B]))

    def test_the_other_securities_of_a_company_are_hidden_with_it(self):
        text = "Ford (NYSE: F; FpB) and FpC rose; GMW did not."
        self.assertEqual(self.hide(text, ["F"]), f"{A} ({A}; FpB) and FpC rose; GMW did not.")
        self.assertEqual(self.hide(text, ["F"], others={"FpB": "F", "FpC": "F", "GMW": "GM"}),
                         f"{A} and {A} rose; GMW did not.")  # GM is not scored here.
        with self.assertRaises(ValueError):
            mask(None, None, [f"T{index}" for index in range(257)])
        self.assertEqual(self.hide("Ford  rose", ["F"]), f"{A} rose")

    def test_an_alias_starts_with_the_name_it_comes_from(self):
        day = pd.Timestamp("2023-05-01", tz="UTC")
        text = "Gold rose. Barrick Gold said Gold demand grew."
        self.assertEqual(self.hide(text, ["GOLD"], published=day), f"Gold rose. {A} said Gold demand grew.")
        late = pd.Timestamp("2026-02-01", tz="UTC")  # GOLD is Gold.com's ticker now.
        self.assertEqual(self.hide(text, ["GOLD"], published=late), f"{A} rose. {A} said {A} demand grew.")
        self.assertEqual(self.hide("Rocket Lab rose.", ["RKLB"], published=day), f"{A} rose.")  # Its first words.

    def test_the_article_gives_the_names_the_table_lacks(self):
        # The capitalized words before a scored ticker in brackets, and the short names a press release defines.
        title, summary, _ = mask(
            "Grand Peak Capital Corp. Update: Why Bonanza (BCEI) Rose",
            "Grand Peak Capital Corp. (\"Grand Peak\" or the \"Company\") (CSE: GPK, FSE: 6GP1, OTC: GPKUF) announces "
            "a deal. Grand Peak’s CEO and Bonanza's agree; Peak demand and its Class A Common Stock (OTC: GPKUF) rose.",
            ["GPKUF", "BCEI"], self.aliases)
        self.assertEqual(title, f"{A} Update: Why {B} Rose")
        self.assertEqual(summary, f"{A} announces a deal. {A}'s CEO and {B}'s agree; Peak demand and its Class A "
                                  f"Common Stock ({A}) rose.")
        self.assertEqual(self.hide("Montero Mining and Exploration Ltd. (TSX-V: MON) („Montero“ oder „MMX“) drilled. "
                                   "Montero and MMX; First Quantum Minerals (FM) and First Quarter results.",
                                   ["MON", "FM"]),
                         f"{A} drilled. {A} and {A}; {B} and First Quarter results.")
        # The first word of such a name, where it stands alone.
        self.assertEqual(self.hide("Evergold Resources Corp. (TSX-V: EVER) rose. Evergold said Evergold Lake is big.",
                                   ["EVER"]), f"{A} rose. {A} said Evergold Lake is big.")
        # A name the table knows but does not use (Lindsay, NEVER above) is one once the article ties it to its ticker.
        self.assertEqual(self.hide("Lindsay (LNN) rose. Lindsay said so.", ["LNN"]), f"{A} rose. {A} said so.")

    def test_the_first_word_of_a_full_name_the_article_writes(self):
        self.assertEqual(self.hide("Sandfire Resources America Inc. said Sandfire would grow; Sandfire BC Holdings "
                                   "lent. Stocks like Macy''s (M) rose.", ["SRAFF", "M"]),
                         f"{A} said {A} would grow; Sandfire BC Holdings lent. Stocks like {B} rose.")

    def test_a_foreign_company_with_the_scored_ticker_is_another_company(self):
        # Polygon tags Boralex (TSX: BLX) with Bladex's BLX: Bladex's placeholder never shows, so it scores 0.
        self.assertEqual(self.hide("Boralex inc. (“Boralex” or the “Company”) (TSX: BLX) announces results. "
                                   "Boralex grew.", ["BLX"]), f"{B} announces results. {B} grew.")
        self.assertEqual(self.hide("Bladex (NYSE: BLX) and its unit (TSX: BLX) grew.", ["BLX"]), f"{A} and its unit "
                                                                                              f"({A}) grew.")

    def test_names_in_headlines_products_and_people(self):
        self.assertEqual(self.hide("Citigroup Maintains Neutral Rating for Ford. Warner Bros. Discovery rose.", ["F"]),
                         f"{B} Maintains Neutral Rating for {A}. {C} rose.")
        # A strong name (most articles that write it are its company's) goes but after a first name; a weaker
        # one where a capitalized word touches it, outside a headline.
        self.assertEqual(self.hide("Forget Apple, Apple Watch sales rose; Harrison Ford acts.", ["GM"]),
                         f"Forget {B}, {B} Watch sales rose; Harrison Ford acts.")
        self.assertEqual(self.hide("New Brunswick said Brunswick and Brunswick Corp. rose; Brunswick Boat sales too.",
                                   ["GM"]), f"New Brunswick said {B} and {B} rose; Brunswick Boat sales too.")
        self.assertEqual(self.hide("the GeForce RTX 4070, a CGI-animated show; RTX and CGI fell.", ["F"]),
                         f"the GeForce RTX 4070, a CGI-animated show; {B} and {C} fell.")

    def test_what_is_written_around_a_name_stays_out_of_it(self):
        self.assertEqual(self.hide("Peering Into Ares Management's Recent Short Interest - Ares Management "
                                   "(NYSE:ARES)", ["ARES"]), f"Peering Into {A}'s Recent Short Interest - {A}")
        self.assertEqual(mask(None, "Contact the firm before April 23 Deadline (NYSE:ATKR)", ["ATKR"],
                              self.aliases)[1], f"Contact the firm before [date] Deadline ({A})")
        self.assertEqual(self.hide("XPLR Infrastructure, LP fka NextEra Energy Partners, LP Investors", ["XPLR"]),
                         f"{A} Investors")  # A former name is the same company.
        self.assertEqual(self.hide("The Cheesecake Factory Incorporated (NASDAQ: CAKE) fell; TJX Companies rose.",
                                   ["CAKE", "TJX"]), f"{A} fell; {B} rose.")
        self.assertEqual(self.hide("Filament Health Corp. (OTCQB: FLHLF)(NEO: FH) rose", ["F"]), f"{B} rose")
        self.assertEqual(self.hide("Technology Select Sector SPDR ETF (XLK) beat Ford; XLK has outperformed.",
                                   ["F"]), f"{B} beat {A}; {B} has outperformed.")

    def test_headline_words_are_not_a_name(self):
        self.assertEqual(self.hide("Contact BFA Law if You Suffered Losses (NYSE:RDDT)", ["RDDT"]),
                         f"Contact BFA Law if You Suffered Losses ({A})")
        self.assertEqual(self.hide("Sculptor’s Independent Board Members (NYSE: SCU) and CIO", ["SCU"]),
                         f"{A}'s Independent Board Members ({A}) and CIO")
        self.assertEqual(self.hide("AEGIS Doubles Down on Vermillion Energy (VET), Should You Invest?", ["VET"]),
                         f"AEGIS Doubles Down on {A}, Should You Invest?")  # Misspelled: the article tells.

    def test_a_scored_companys_short_name_is_its_own(self):
        self.assertEqual(self.hide("TC BioPharm Hldgs said; technology from TC BioPharm grew.", ["TCBPY"]),
                         f"{A} said; technology from {A} grew.")
        self.assertIn(("morgan", "stanley"), self.aliases.wide)  # Hidden in any article.
        self.assertEqual(name_aliases("AKER SOLTNS ASA UNSP/ADR").get(("aker", "solutions")), "name")
        self.assertEqual(name_aliases("NATL AUSTRALIA BK").get(("national", "australia", "bank")), "name")

    def test_a_foreign_company_compared_by_its_otc_ticker(self):
        self.assertEqual(mask("MARUY vs. Ford: Which Stock Should Value Investors Buy Now?", "Ford or UOVEY: Which Is "
                              "Better? GM and Ford rose.", ["F"], self.aliases)[:2],
                         (f"{B} vs. {A}: Which Stock Should Value Investors Buy Now?",
                          f"{A} or {C}: Which Is Better? {D} and {A} rose."))

    def test_tags_and_codes(self):
        self.assertEqual(self.hide("#gold and #AAPL; @FordNews. ID code DK0061027356, Luxshare (002475.SZ) and "
                                   "SHOP.TO.", ["F"]),
                         f"#gold and [handle]; [handle]. ID code [code], Luxshare ([code]) and {B}.")

    def test_other_companies_are_hidden_too(self):
        title, summary, labels = mask(
            "Ford, Li Auto Fall", "Shares of Ford Motor (NYSE: F), Baidu Inc. (NASDAQ: BIDU), JD.Com Inc. (NASDAQ: JD) "
            "and Li Auto Inc. (NASDAQ: LI) fell. Li Auto lost 5%; Baidu less. Morgan Stanley and Apple Watch; AAPL "
            "and $LI; under the ticker symbol “BTTX”. Lindsay's (LNN) results beat macular degeneration (AMD) and "
            "artificial intelligence (AI). New Brunswick and Brunswick said MS is a disease.", ["F"], self.aliases)
        self.assertEqual(labels, [A])
        self.assertEqual(title, f"{A}, {B} Fall")  # The others follow the scored companies, as the text names them.
        self.assertEqual(summary, f"Shares of {A}, {C}, {D} and {B} fell. {B} lost 5%; {C} less. Company E and "
                                  f"Company F Watch; Company F and {B}; under the ticker symbol “Company G”. "
                                  f"Company H's results beat macular degeneration (AMD) and artificial intelligence "
                                  f"(AI). New Brunswick and Company I said MS is a disease.")

    def test_addresses_and_codes_go(self):
        self.assertEqual(self.hide("See www.ford.com, https://ir.ford.com/news?id=1, investors.ford.com or "
                                   "ir@ford.com; @FordNews #ford. Ford.com sells, Zacks.com rates.", ["F"]),
                         "See [link], [link], [link] or [email]; [handle] [handle]. [link] sells, [link] rates.")

    def test_without_a_table_tickers_and_the_names_beside_them_are_hidden(self):
        self.assertEqual(mask("Ford (NYSE: F) and GM (GM) in 2023", "F and GM rose; Ford too.", ["F", "GM"]),
                         (f"{A} and {B} in 20XX", f"{A} and {B} rose; {A} too.", [A, B]))
        self.assertEqual(mask(None, None, []), ("", "", []))
        with self.assertRaises(ValueError):  # A table built before mask() hid other companies.
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                table(("F", "ford", "start")).drop(columns=["wide"]).to_parquet(f"{directory}/old.parquet")
                Aliases.read(f"{directory}/old.parquet")


class BuildTest(unittest.TestCase):
    def test_where_an_alias_is_used(self):
        of = dict(seen=True, of_ticker=100, of_all=100_000)
        self.assertEqual(use("start", 1, 0, 0, 0, 0, False, 100, 100_000), ALWAYS)  # The news never contain it.
        self.assertEqual(use("start", 1, 0, 0, 0, 0, **of), NEVER)  # Coca: only ever inside Coca-Cola.
        self.assertEqual(use("name", 3, 0, 0, 0, 0, **of), ALWAYS)  # A full name, only inside a longer one.
        self.assertEqual(use("start", 1, 160, 137, 100, 0, **of), ALWAYS)  # Ford.
        self.assertEqual(use("start", 1, 1000, 5, 4, 0, **of), NEVER)  # American, for American Airlines.
        self.assertEqual(use("start", 1, 900, 60, 50, 0, **of), ALONE)  # Apollo: analysts' firm in other articles.
        self.assertEqual(use("start", 1, 900, 60, 20, 40, **of), NEVER)  # Energy: written small in its own articles.
        self.assertEqual(use("word", 1, 127, 26, 20, 3, **of), NEVER)  # Solar, for First Solar.
        self.assertEqual(use("ticker", 1, 31, 15, 0, 0, **of), NEVER)  # ALL, for Allstate.
        self.assertEqual(use("ticker", 1, 393, 391, 0, 0, **of), ALWAYS)  # NEE.
        self.assertEqual(use("name", 1, 102_604, 564, 502, 0, **of), ALONE)  # Nasdaq: its name, however rare.
        self.assertEqual(use("start", 2, 304, 79, 0, 0, **of), ALWAYS)  # Goldman Sachs: analysts everywhere.
        self.assertEqual(use("start", 2, 4000, 8, 0, 0, **of), NEVER)  # United States, for U.S. Steel.
        self.assertEqual(use("name", 2, 347, 65, 0, 0, **of), ALWAYS)  # Morgan Stanley: a full name.
        self.assertEqual(use("name", 2, 19, 4, 0, 8, **of, small=900), NEVER)  # Quantum Computing, written small.
        self.assertEqual(use("name", 2, 3, 3, 0, 30, **of, small=31), ALWAYS)  # e.l.f. Beauty: small, but its own.
        self.assertEqual(use("name", 1, 900, 800, 700, 0, **of), ALWAYS)  # Nvidia: nearly always with its tag.
        self.assertEqual(use("name", 1, 757, 100, 86, 0, **of), ALONE)  # Twitter: named in many other articles.
        self.assertEqual(use("name", 1, 765, 80, 70, 10, **of), ALONE)  # Target.
        self.assertEqual(use("name", 1, 385, 31, 8, 7, **of), NEVER)  # Gold, for Barrick Gold.
        self.assertEqual(use("name", 1, 3000, 6, 6, 40, **of), NEVER)  # Now, for NOW Inc.: small in its articles.
        self.assertEqual(use("name", 1, 108, 17, 5, 0, **of), ALONE)  # Williams, mostly in longer names.

    def test_where_an_alias_is_hidden_in_other_articles(self):
        self.assertEqual(wide("name", "morgan stanley", ALWAYS, 4285, 733, 0, 4300, 10), ALWAYS)
        self.assertEqual(wide("name", "bancorp inc", ALWAYS, 1947, 48, 0, 1947, 1800), "")  # The end of other names.
        self.assertEqual(wide("seen", "nasdaq global market", ALWAYS, 440, 66, 0, 440, 0), "")  # Not from its name.
        self.assertEqual(wide("name", "kailera therapeutics", ALWAYS, 0, 0, 0, 0, 0), ALWAYS)  # Never written yet.
        self.assertEqual(wide("name", "apple", ALWAYS, 7643, 6973, 40, 7700, 0), ALWAYS)
        self.assertEqual(wide("start", "brunswick", ALWAYS, 291, 131, 0, 291, 0), ALONE)  # New Brunswick.
        self.assertEqual(wide("name", "block", ALWAYS, 1598, 919, 900, 1598, 0), "")  # Also written small: a word.
        self.assertEqual(wide("seen", "conn", ALWAYS, 1005, 105, 0, 1005, 0), "")  # Conn., mostly.
        self.assertEqual(wide("name", "target", ALONE, 9176, 1406, 0, 9176, 0), "")
        self.assertEqual(wide("ticker", "NVDA", ALWAYS, 5000, 4900, 0, 5000, 0), ALWAYS)
        self.assertEqual(wide("ticker", "ON", ALWAYS, 1000, 600, 0, 1000, 0), "")  # Often the word, in capitals.

    def test_the_news_confirm_the_aliases(self):
        names = pd.DataFrame({"ticker": ["F", "AAL", "KO", "KOO", "NEE", "NEE", "TGT", "LULU", "DANOY", "UBS"],
                              "first_date": pd.to_datetime(["2021-01-01"] * 3 + ["2023-01-02", "2021-01-01",
                                                           "2025-10-01", "2021-01-01", "2021-01-01", "2022-01-03",
                                                           "2021-01-01"]),
                              "market": ["stocks"] * 8 + ["otc", "stocks"],
                              "name": ["Ford Motor Company", "American Airlines Group Inc.", "Coca-Cola Company",
                                       "Koo Holdings Inc", "Nextra Energy, Inc.", "NextEra Energy, Inc.",
                                       "Target Corporation", "Lululemon Athletica Inc.", "DANONE S/ADR",
                                       "UBS Group AG"]})
        day = pd.Timestamp("2022-03-01", tz="UTC")
        news = ([("Ford (F) rises", "Ford Motor said sales rose. Ford gained.", ["F"], day)] * 4
                + [("American Airlines (AAL) falls", "American Airlines Group cut flights for American travelers.",
                    ["AAL"], day)] * 4
                + [("American consumers spend", "American shoppers and Ford dealers.", ["XRT"], day)] * 9
                + [("Coke (KO) pops", "Coke (KO) is Coca-Cola.", ["KO"], day)] * 3
                + [("NextEra Energy (NEE) gains", "NextEra Energy raised its dividend.", ["NEE"], day)] * 3
                + [("Target (TGT) Gets A Price Target", "Target Stores did well; Target said so.", ["TGT"], day)] * 2
                + [("Target (TGT) Rises", "Target Stores did well; Target said so.", ["TGT"], day)] * 4
                + [("Analysts Lift A Price Target", "The price target is higher.", ["WMT"], day)] * 12
                + [("Markets close mixed", "American stocks were little changed.", ["SPY"], day)] * 40
                + [("lululemon (LULU) jumps", "Shares of lululemon athletica rose.", ["LULU"], day)] * 4
                + [("Danone buys a rival", "Danone said the deal closed.", ["DANOY"], day)] * 3
                + [("UBS profit rises", "UBS said its profit rose.", ["UBS"], day)] * 3
                + [("Broker cuts a rating", "A UBS analyst cut the rating.", ["WMT"], day)] * 8)
        aliases = build_aliases(names, news)
        self.assertEqual(list(aliases.columns), ALIAS_COLUMNS)
        row = lambda ticker, alias: aliases[(aliases.ticker == ticker) & (aliases.alias == alias)].iloc[0]
        ford = row("F", "ford")
        # Ford is in its 4 articles and in 9 of the 95 others: its name, where no other name holds it.
        self.assertEqual((ford.kind, ford.articles, ford.tagged, ford.use), ("start", 13, 4, ALONE))
        self.assertEqual((row("F", "ford motor").use, row("F", "ford motor").wide), (ALWAYS, ALWAYS))
        american = row("AAL", "american")
        self.assertEqual((american.articles, american.tagged, american.use, american.wide), (53, 4, NEVER, ""))
        self.assertEqual(row("AAL", "american airlines").use, ALWAYS)
        self.assertEqual((row("KO", "coke").kind, row("KO", "coke").use), ("seen", ALWAYS))
        self.assertEqual(row("KO", "coca cola").use, ALWAYS)  # Coca-Cola and Coca Cola.
        self.assertTrue(pd.isna(row("F", "ford").since))  # The ticker's name since before the first date read.
        self.assertEqual(row("KOO", "koo holdings").since, pd.Timestamp("2023-01-02"))
        # A name Polygon gives late applies from the day the news wrote it before the ticker.
        self.assertEqual(row("NEE", "nextera energy").since, pd.Timestamp("2022-03-01"))
        self.assertTrue(pd.isna(row("DANOY", "danone").since))  # The first date of its market.
        self.assertEqual((row("DANOY", "danone").use, row("DANOY", "danone").wide), (ALWAYS, ALWAYS))
        # Beside "Price", Target is another name: the other articles write the two together.
        target = row("TGT", "target")
        self.assertEqual((target.use, target.apart), (ALONE, "price<"))
        self.assertEqual((row("LULU", "lululemon").use, bool(row("LULU", "lululemon").any_case)), (ALWAYS, True))
        # UBS is in more articles of other tickers than of its own, but it is its company's name: hidden as one.
        self.assertEqual((row("UBS", "ubs").use, row("UBS", "UBS").use, row("UBS", "UBS").tagged), (ALONE, ALWAYS, 3))
        masked = mask("Coca-Cola (KO) and Ford", "Coke rose; American Airlines fell.", ["KO", "AAL"], Aliases(aliases))
        self.assertEqual(masked[:2], (f"{A} and Ford", f"{A} rose; {B} fell."))
        masked = mask("NextEra Energy gains; Danone too", "Target Stores has a Price Target. lululemon rose.",
                      ["NEE", "TGT", "LULU"], Aliases(aliases), day)
        self.assertEqual(masked[:2], (f"{A} gains; {D} too", f"{B} Stores has a Price Target. {C} rose."))


if __name__ == "__main__":
    unittest.main()
