import os
import re
import glob

import pandas as pd
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

PDF_DIR = os.path.join(BASE_DIR, "bhas_pdfs")
OUT_DIR = os.path.join(BASE_DIR, "analysis")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
SUMMARY_CSV = os.path.join(OUT_DIR, "dataset_ukupno.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

import pdfplumber

BROJ_RE = re.compile(r"^-?\d[\d.,]*$|^-$")

def je_broj(t):
    return bool(BROJ_RE.match(t.strip()))

def ba_broj(t):
    if t is None:
        return None
    t = str(t).strip().replace(" ", "").replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None

def izvuci_godinu_mjesec(putanja):
    ime = os.path.basename(putanja)
    m = re.search(r"(20\d{2})[_\-](\d{1,2})", ime)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def redovi_iz_pdfa(putanja):
    redovi = []
    with pdfplumber.open(putanja) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for linija in text.split("\n"):
                polja = linija.split()
                redovi.append(polja)
    return redovi

def parsiraj_pdf(putanja):
    godina, mjesec = izvuci_godinu_mjesec(putanja)
    dolasci_domaci = nocenja_domaci = dolasci_strani = nocenja_strani = None

    for polja in redovi_iz_pdfa(putanja):
        if not polja:
            continue
        etiketa = polja[0].lower()
        brojevi = [p for p in polja[1:] if je_broj(p)]
        if len(brojevi) < 2:
            continue

        if "domac" in etiketa and dolasci_domaci is None:
            dolasci_domaci = ba_broj(brojevi[0])
            nocenja_domaci = ba_broj(brojevi[-1])
        elif "stran" in etiketa and dolasci_strani is None:
            dolasci_strani = ba_broj(brojevi[0])
            nocenja_strani = ba_broj(brojevi[-1])

    dolasci_ukupno = None
    nocenja_ukupno = None
    if dolasci_domaci is not None and dolasci_strani is not None:
        dolasci_ukupno = dolasci_domaci + dolasci_strani
        nocenja_ukupno = nocenja_domaci + nocenja_strani

    return {
        "fajl": os.path.basename(putanja),
        "godina": godina,
        "mjesec_kraja": mjesec,
        "dolasci_ukupno": dolasci_ukupno,
        "dolasci_domaci": dolasci_domaci,
        "dolasci_strani": dolasci_strani,
        "nocenja_ukupno": nocenja_ukupno,
        "nocenja_domaci": nocenja_domaci,
        "nocenja_strani": nocenja_strani,
    }

def main():
    putanje = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not putanje:
        raise SystemExit(f"Nema PDF fajlova u '{PDF_DIR}/'.")

    redovi = []
    for p in putanje:
        try:
            redovi.append(parsiraj_pdf(p))
            print(p, "OK")
        except Exception as e:
            print(p, "GRESKA", e)

    df = pd.DataFrame(redovi).sort_values(["godina", "mjesec_kraja"]).reset_index(drop=True)
    df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    df_god = df[df["mjesec_kraja"] == 12]
    if df_god.empty:
        df_god = df.loc[df.groupby("godina")["mjesec_kraja"].idxmax()]

    agg = df_god.groupby("godina")[["nocenja_domaci", "nocenja_strani"]].max().dropna()
    if not agg.empty:
        agg.plot(kind="bar")
        plt.title("Nocenja turista po godinama")
        plt.ylabel("Broj nocenja")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "trend_nocenja.png"))
        plt.close()

    print("Zavrseno.")

if __name__ == "__main__":
    main()