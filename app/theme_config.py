"""
Configuration for grouping tickers into Themes, Sectors, and Factors.
Used by Theme_Analysis.py for rotation strategies.
"""

# Merged mapping including emojis from ThematicDashboard.py
THEME_MAPPING = {
    # ── Sectors (GICS/Industry) ──
    "Technology": ["QQQ", "IUIT.L", "ESIT.L"],
    "Financials": ["IUFS.L", "ESIF.L", "BNKS.L"],
    "Healthcare": ["IHCU.L", "ESIH.L", "HEAL.L", "BTEC.L"],
    "Energy": ["IUES.L", "ESIE.L", "WENS.L", "INRG.L", "XLE"],
    "Industrials": ["IUIS.L", "ESIN.L", "DFND.L"],
    "Consumer Discretionary": ["IUCD.L", "XLY", "ESIC.L", "PLAY.L", "EBIG.L"],
    "Consumer Staples": ["ICSU.L", "XLP", "IUCS.L"],
    "Utilities": ["IUSU.L", "XDLU.L"],
    "Materials": ["IUMS.L", "PICK"],
    "Real Estate": ["XRES.L", "DPYG.L", "IASP.SW", "IUKP.L", "IUSP.L", "IWDP"],
    "Comms": ["XLC"],
    # ── Thematic (Growth/Innovation) ──
    "🤖 AI & Big Data": [
        "AIAG.L",
        "XAIX.L",
        "ROBG.L",
        "RBTX.L",
        "BOTZ",
        "AIAG",
        "XAIX",
    ],
    "🔒 Cybersecurity": ["ISPY.L", "LOCK.L", "CIBR", "HACK", "ISPY", "LOCK"],
    "💾 Hardware & Semis": ["SMH.L", "SOXX", "SMH"],
    "☁️ Cloud Computing": ["FSKY.L", "CLOU", "SKYY", "WCLD", "DGIT.L", "FSKY"],
    "🦾 Robotics & Automation": ["ROBG", "RBTX"],
    "🌿 Clean Energy": ["INRG.L", "ICLN", "PBW", "TAN", "INRG"],
    "🚗 Autonomous & EV": ["ECAR.L", "BATG.L", "LITG.L", "DRIV", "ECAR"],
    "🔌 Battery & EV": ["BATG", "CHRG", "LITG"],
    "💳 FinTech & Digital": ["FING.L", "BCHN.L", "DAGB.L", "FINX", "FING", "DGIT"],
    "⛓️ Blockchain & Crypto": ["BCHN", "DAGB"],
    "📡 IoT": ["SNSG.L", "SNSG"],
    "⚛️ Quantum Computing": ["QNTG", "WQTM.L", "WQTM"],
    "🎮 Gaming & eSports": ["ESGB.L", "ESGB", "PLAY"],
    "🚀 Space & Innovation": ["JEDI.L", "ARKX", "JEDI", "DFND", "NATO"],
    "🛡️ Defense": ["ITA", "DFND.L", "NATO.L", "PPA"],
    "💊 Healthcare & Biotech": ["HEAL", "BTEC"],
    "🧬 Nanotech / Health": ["DOCG.L", "DOCG"],
    "👴 Aging Population": ["AGED.L", "AGED"],
    "🏙️ Smart Cities": ["IQCY.L", "IQCY"],
    "🛢️ Traditional Energy": ["WENS"],
    "💧 Water": ["IH2O.L", "PHO", "CGW", "IH2O"],
    "🔋 Hydrogen": ["HTWG.L", "HTWG", "HDRO"],
    "☢️ Uranium & Nuclear": ["URNG.L", "NUCG.L", "URA", "URNG", "NUCG"],
    "📦 E-commerce & Logistics": ["ECOG.L", "EBIG", "ECOG"],
    "💎 Luxury": ["LUXG.L", "LUXG"],
    "🌾 Agribusiness": ["SPAG.L", "KROG.L", "MOO", "VEGI", "SPAG", "KROG"],
    "🏗️ Infrastructure": ["INFR.L", "IGF", "PAVE", "INFR"],
    "Marijuana": ["MJ", "YOLO"],
    # ── Regions ──
    "US Large Cap": ["CSP1.L", "SPY", "IVV", "VOO"],
    "US Small Cap": ["IWM", "IJR", "VB", "SMGB.L"],
    "UK": ["VUKG.L", "VMIG.L", "ISF.L", "VUKE.L"],
    "Europe": ["IMEA.L", "FEZ", "EUMD.L", "DJSC.L", "IEUR"],
    "Japan": ["EWJ", "IJPH.L", "SCJ", "DXJ"],
    "China": ["FXI", "MCHI", "KWEB", "CQQQ"],
    "Emerging Markets": ["EEM", "VWO", "IEMG", "EXCS.L", "EEMS"],
    "India": ["INDA", "EPI"],
    "Brazil": ["EWZ"],
    "Asia Pacific": ["VDPG.L", "AAXJ"],
    "Latin America": ["ILF"],
    # ── Factors ──
    "Value": ["IWVL.L", "IUVL.L", "IEVL.L", "VTV", "IUSV"],
    "Growth": ["R1GB.L", "VUG", "IUSG"],
    "Quality": ["IWQU.L", "IUQA.L", "QUAL"],
    "Momentum": ["IWMO.L", "IUMF.L", "MTUM"],
    "Min Volatility": ["MVOL.L", "USMV", "EFAV", "LOWV.L"],
    "Dividends": ["USDV.L", "UKDV.L", "WQDS.L", "IDVY.L", "VIG", "VYM"],
    "Size": ["IEFS.L", "SIZE"],
    # ── Macro ──
    "Gold": ["GLD", "SGLD.L", "IAU"],
    "⛏️ Gold Miners": ["GDX", "GDXJ"],
    "Commodities": ["GSG", "DBC", "PDBC"],
    "Bonds - Gov": ["IEF", "TLT", "SHY", "GOVT", "IB01.L", "GLTL.L", "IBGL.AS"],
    "Bonds - Corp": ["HYG", "LQD", "JNK"],
    "Bonds - Inf Linked": ["TIP", "ITPG.L", "INXG.L", "LTPZ.L", "STIP"],
    "Bonds - EM": ["EMB", "LEMB"],
    "Volatility": ["^VIX", "VIXY"],
    "Currency": ["UUP", "FXE", "FXY", "FXB"],
}

# Reverse mapping for easy lookup
TICKER_TO_THEME = {}
for theme, tickers in THEME_MAPPING.items():
    for t in tickers:
        # Some tickers might be in multiple themes?
        # For now, last one wins or we can make it a list.
        # Let's make it a list to support multi-theme assets.
        if t not in TICKER_TO_THEME:
            TICKER_TO_THEME[t] = []
        TICKER_TO_THEME[t].append(theme)
