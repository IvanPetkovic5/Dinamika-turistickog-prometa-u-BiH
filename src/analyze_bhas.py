import glob
import os
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfplumber

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

PDF_DIR = os.path.join(BASE_DIR, "bhas_pdfs")
OUT_DIR = os.path.join(BASE_DIR, "analysis")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
DATASET_CSV = os.path.join(OUT_DIR, "dataset_zemlje.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "dataset_ukupno.csv")
EDA_TXT = os.path.join(OUT_DIR, "eda_izvjestaj.txt")

os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.size"] = 11

BROJ_POLJE_RE = re.compile(r"^-?\d[\d.,]*$|^-$")


def _je_broj_polje(polje):
    return bool(BROJ_POLJE_RE.match(polje.strip()))


def ba_broj(tekst):
    if tekst is None:
        return None
    t = str(tekst).strip()
    t = t.replace("\u202f", " ").replace("\xa0", " ")
    t = t.replace(" ", "").replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def izvuci_godinu_mjesec(putanja_pdf):
    ime = os.path.basename(putanja_pdf)
    m = re.search(r"(20\d{2})[_\-](\d{1,2})", ime)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


ALIASI_ZEMALJA = {
    "makedonija b.j.r.": "Sjeverna Makedonija",
    "sjeverna makedonija": "Sjeverna Makedonija",
    "švajcarska (uključujući i lihtenštajn)": "Švicarska",
    "švicarska (uključujući i lihtenštajn)": "Švicarska",
    "švajcarska": "Švicarska",
    "švicarska": "Švicarska",
    "sjedinjene američke države": "SAD",
    "sad": "SAD",
    "holandija": "Nizozemska/Holandija",
    "nizozemska": "Nizozemska/Holandija",
    "ujedinjeni arapski emirati": "UAE",
    "uae": "UAE",
    "ujedinjeno kraljevstvo": "Ujedinjeno Kraljevstvo",
    "velika britanija": "Ujedinjeno Kraljevstvo",
}

NIJE_ZEMLJA = {
    "ukupno",
    "domaći turisti",
    "domaci turisti",
    "strani turisti",
    "ukupno strani turisti",
    "sve zemlje",
    "ostale zemlje",
    "ostale evropske zemlje",
    "ostale afričke zemlje",
    "ostale sjeverno-američke zemlje",
    "ostale sjevernoameričke zemlje",
    "ostale zemlje južne i srednje amerike",
    "ostale azijske zemlje",
    "ostale zemlje okeanije",
}


def _normalizuj_tekst(tekst):
    if not tekst:
        return ""
    tekst = tekst.strip().lower()
    nfkd = unicodedata.normalize("NFD", tekst)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def normalizuj_zemlju(naziv):
    n = naziv.strip()
    kljuc = n.lower()
    kljuc = re.sub(r"\s*\d\)\s*$", "", kljuc).strip()
    return ALIASI_ZEMALJA.get(kljuc, n.strip())


def redovi_tabele_po_koordinatama(putanja_pdf):
    svi_redovi = []
    with pdfplumber.open(putanja_pdf) as pdf:
        for page in pdf.pages:
            words = page.extract_words(
                x_tolerance=2, y_tolerance=2, use_text_flow=True
            )
            if not words:
                continue

            words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))

            linije = []
            trenutna_linija = []

            for w in words_sorted:
                if not trenutna_linija:
                    trenutna_linija.append(w)
                else:
                    avg_top = sum(x["top"] for x in trenutna_linija) / len(
                        trenutna_linija
                    )
                    if abs(w["top"] - avg_top) < 7.0:
                        trenutna_linija.append(w)
                    else:
                        linije.append(trenutna_linija)
                        trenutna_linija = [w]
            if trenutna_linija:
                linije.append(trenutna_linija)

            for linija_rijeci in linije:
                linija_rijeci = sorted(linija_rijeci, key=lambda w: w["x0"])
                polja = []
                trenutno_polje = linija_rijeci[0]["text"]

                for prev, cur in zip(linija_rijeci, linija_rijeci[1:]):
                    gap = cur["x0"] - prev["x1"]
                    zadnji_tekst = (
                        trenutno_polje.strip().split()[-1]
                        if trenutno_polje.strip()
                        else ""
                    )
                    c_text = cur["text"].strip()

                    is_num_last = _je_broj_polje(zadnji_tekst)
                    is_num_cur = _je_broj_polje(c_text)

                    if is_num_last and is_num_cur:
                        je_hiljadarka = bool(re.match(r"^\d{3}[\d.,]*$", c_text))
                        if je_hiljadarka and gap < 10.0:
                            trenutno_polje += c_text
                        else:
                            polja.append(trenutno_polje)
                            trenutno_polje = c_text
                    elif not is_num_last and not is_num_cur:
                        if gap < 45.0:
                            trenutno_polje += " " + c_text
                        else:
                            polja.append(trenutno_polje)
                            trenutno_polje = c_text
                    else:
                        if gap < 8.0:
                            trenutno_polje += " " + c_text
                        else:
                            polja.append(trenutno_polje)
                            trenutno_polje = c_text

                polja.append(trenutno_polje)
                svi_redovi.append(polja)
    return svi_redovi


def parsiraj_red_zemlje_iz_polja(polja):
    if len(polja) < 6:
        return None

    prvo = polja[0].strip()
    if not prvo or prvo[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZČĆŠĐŽ":
        return None
    if prvo.upper() == prvo:
        return None

    ostatak = polja[1:]
    if not all(_je_broj_polje(p) for p in ostatak):
        return None

    naziv = re.sub(r"\d\)\s*$", "", prvo).strip()
    return naziv, ostatak


def parsiraj_pdf(putanja_pdf):
    godina, mjesec = izvuci_godinu_mjesec(putanja_pdf)
    dolasci_domaci = nocenja_domaci = dolasci_strani = nocenja_strani = None

    svi_redovi = redovi_tabele_po_koordinatama(putanja_pdf)

    for polja in svi_redovi:
        if not polja:
            continue

        etiketa_norm = _normalizuj_tekst(polja[0])
        brojevi = [p for p in polja[1:] if _je_broj_polje(p)]

        if len(brojevi) < 5:
            continue

        if "domac" in etiketa_norm and dolasci_domaci is None:
            dolasci_domaci = ba_broj(brojevi[1])
            nocenja_domaci = ba_broj(brojevi[4])
        elif "stran" in etiketa_norm and dolasci_strani is None:
            if "udio" not in etiketa_norm and "struktura" not in etiketa_norm:
                dolasci_strani = ba_broj(brojevi[1])
                nocenja_strani = ba_broj(brojevi[4])

        if dolasci_domaci is not None and dolasci_strani is not None:
            break

    dolasci_ukupno = (
        (dolasci_domaci + dolasci_strani)
        if dolasci_domaci is not None and dolasci_strani is not None
        else None
    )
    nocenja_ukupno = (
        (nocenja_domaci + nocenja_strani)
        if nocenja_domaci is not None and nocenja_strani is not None
        else None
    )
    prosjecan_boravak = (
        round(nocenja_ukupno / dolasci_ukupno, 2)
        if dolasci_ukupno and nocenja_ukupno
        else None
    )

    zbirni_red = {
        "fajl": os.path.basename(putanja_pdf),
        "godina": godina,
        "mjesec_kraja": mjesec,
        "dolasci_ukupno": dolasci_ukupno,
        "dolasci_domaci": dolasci_domaci,
        "dolasci_strani": dolasci_strani,
        "nocenja_ukupno": nocenja_ukupno,
        "nocenja_domaci": nocenja_domaci,
        "nocenja_strani": nocenja_strani,
        "prosjecan_boravak": prosjecan_boravak,
        "udio_stranih_%": (
            round(nocenja_strani / nocenja_ukupno * 100, 2)
            if nocenja_strani and nocenja_ukupno
            else None
        ),
    }

    redovi_zemalja = []
    nije_zemlja_norm = {_normalizuj_tekst(z) for z in NIJE_ZEMLJA}

    for polja in svi_redovi:
        rez = parsiraj_red_zemlje_iz_polja(polja)
        if rez is None:
            continue

        naziv_sirovi, brojevi = rez
        naziv_norm = _normalizuj_tekst(naziv_sirovi)

        if (
            naziv_norm in nije_zemlja_norm
            or "ukupno" in naziv_norm
            or "domac" in naziv_norm
            or "stran" in naziv_norm
        ):
            continue

        naziv = normalizuj_zemlju(naziv_sirovi)

        if len(brojevi) >= 8:
            dolasci_tekuca = ba_broj(brojevi[1])
            nocenja_tekuca = ba_broj(brojevi[4])
            struktura_pct = ba_broj(brojevi[6])
            prosj_boravak = ba_broj(brojevi[7])
        elif len(brojevi) >= 5:
            dolasci_tekuca = ba_broj(brojevi[1])
            nocenja_tekuca = ba_broj(brojevi[4])
            struktura_pct = None
            prosj_boravak = None
        else:
            continue

        if nocenja_tekuca is None:
            continue

        redovi_zemalja.append({
            "fajl": os.path.basename(putanja_pdf),
            "godina": godina,
            "mjesec_kraja": mjesec,
            "zemlja": naziv,
            "dolasci": dolasci_tekuca,
            "nocenja": nocenja_tekuca,
            "struktura_%": struktura_pct,
            "prosjecan_boravak": prosj_boravak,
        })

    return zbirni_red, redovi_zemalja


def main():
    putanje = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not putanje:
        raise SystemExit(f"Nema PDF fajlova u '{PDF_DIR}/'.")

    zbirni_redovi = []
    svi_redovi_zemalja = []

    print(f"Parsiram {len(putanje)} PDF fajlova...\n")
    for p in putanje:
        try:
            zbir, zemlje = parsiraj_pdf(p)
            status = "OK" if zbir["nocenja_ukupno"] else "UPOZORENJE"
            print(
                f"  {status}: {zbir['fajl']} -> godina={zbir['godina']},"
                f" mjesec={zbir['mjesec_kraja']}, zemalja={len(zemlje)}"
            )
            zbirni_redovi.append(zbir)
            svi_redovi_zemalja.extend(zemlje)
        except Exception as e:
            print(f"  GREŠKA: {p}: {e}")

    df_ukupno = (
        pd.DataFrame(zbirni_redovi)
        .sort_values(["godina", "mjesec_kraja"])
        .reset_index(drop=True)
    )
    df_zemlje = pd.DataFrame(svi_redovi_zemalja)

    df_ukupno.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    df_zemlje.to_csv(DATASET_CSV, index=False, encoding="utf-8-sig")

    eda_linije = [
        "EDA IZVJEŠTAJ - Analiza Turističkog Prometa BiH",
        "=" * 60,
        f"Broj obrađenih PDF izvještaja: {len(putanje)}",
        (
            "Vremenski raspon:"
            f" {df_ukupno['godina'].min():.0f}-{df_ukupno['godina'].max():.0f}"
        ),
        (
            "Dataset 'po zemljama':"
            f" {len(df_zemlje)} redova x {df_zemlje.shape[1]} kolona"
        ),
        (
            "Broj jedinstvenih zemalja porijekla:"
            f" {df_zemlje['zemlja'].nunique()}"
        ),
        "",
        "Opisna statistika (noćenja po zemlji):",
        df_zemlje["nocenja"]
        .describe()
        .to_string(float_format=lambda x: f"{x:,.0f}"),
        "",
        "Top 10 zemalja po ukupnim noćenjima:",
        str(
            df_zemlje.groupby("zemlja")["nocenja"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        ),
    ]

    with open(EDA_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(eda_linije))

    df_godisnji = df_ukupno[df_ukupno["mjesec_kraja"] == 12].copy()
    if df_godisnji.empty:
        df_godisnji = df_ukupno.loc[
            df_ukupno.groupby("godina")["mjesec_kraja"].idxmax()
        ].copy()

    god_agg = (
        df_godisnji.groupby("godina")[["nocenja_domaci", "nocenja_strani"]]
        .max()
        .dropna()
    )

    if not god_agg.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = np.arange(len(god_agg.index))
        sirina = 0.35

        bar1 = ax.bar(
            x - sirina / 2,
            god_agg["nocenja_domaci"] / 1e6,
            sirina,
            label="Domaći turisti",
            color="#1f77b4",
            alpha=0.85,
        )
        bar2 = ax.bar(
            x + sirina / 2,
            god_agg["nocenja_strani"] / 1e6,
            sirina,
            label="Strani turisti",
            color="#ff7f0e",
            alpha=0.85,
        )

        ax.set_title(
            "Slika 1: Godišnji trend noćenja turista u BiH (januar - decembar)",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("Godina")
        ax.set_ylabel("Broj noćenja (u milionima)")
        ax.set_xticks(x)
        ax.set_xticklabels(god_agg.index.astype(int))
        ax.legend(frameon=True, facecolor="white", edgecolor="none")

        for bar in bar1:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.05,
                f"{yval:.2f}M",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        for bar in bar2:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.05,
                f"{yval:.2f}M",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        fig.tight_layout()
        fig.savefig(
            os.path.join(PLOTS_DIR, "slika1_trend_nocenja.png"), dpi=300
        )
        plt.close(fig)

    if not df_zemlje.empty:
        top10_zemlje = (
            df_zemlje.groupby("zemlja")["nocenja"]
            .sum()
            .nlargest(10)
            .sort_values(ascending=True)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(
            top10_zemlje.index,
            top10_zemlje.values / 1e3,
            color="#2b5c8f",
            alpha=0.85,
        )
        ax.set_title(
            "Slika 2: Top 10 zemalja po ukupnom broju noćenja (u hiljadama)",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("Broj noćenja (u hiljadama)")
        ax.set_ylabel("Zemlja")

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + (width * 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{width:,.1f}k",
                va="center",
                fontsize=9,
            )

        fig.tight_layout()
        fig.savefig(
            os.path.join(PLOTS_DIR, "slika2_top10_zemlje.png"), dpi=300
        )
        plt.close(fig)

    if not df_zemlje.empty:
        ukupna_nocenja_zemalja = df_zemlje.groupby("zemlja")["nocenja"].sum()
        top5 = ukupna_nocenja_zemalja.nlargest(5)
        ostalo = pd.Series(
            {"Ostale zemlje": ukupna_nocenja_zemalja.sum() - top5.sum()}
        )
        podaci_pie = pd.concat([top5, ostalo])

        fig, ax = plt.subplots(figsize=(8, 8))
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#7f7f7f",
        ]

        wedges, texts, autotexts = ax.pie(
            podaci_pie,
            labels=podaci_pie.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            pctdistance=0.75,
        )

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_weight("bold")

        ax.set_title(
            "Slika 3: Struktura noćenja stranih turista (Top 5 vs Ostale)",
            fontsize=13,
            pad=12,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(PLOTS_DIR, "slika3_struktura_top_zemalja.png"), dpi=300
        )
        plt.close(fig)

    print(
        "Generisani CSV fajlovi, EDA izvještaj i 3 grafikona u 'analysis/plots/'."
    )


if __name__ == "__main__":
    main()