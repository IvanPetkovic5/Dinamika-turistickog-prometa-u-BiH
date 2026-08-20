import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import pdfplumber

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

PDF_DIR = os.path.join(BASE_DIR, "bhas_pdfs")
OUT_DIR = os.path.join(BASE_DIR, "analysis")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
SUMMARY_CSV = os.path.join(OUT_DIR, "dataset_ukupno.csv")
ZEMLJE_CSV = os.path.join(OUT_DIR, "dataset_zemlje.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

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
  list_zemalja = []

  for polja in redovi_iz_pdfa(putanja):
    if not polja:
      continue

    brojevi = [p for p in polja if je_broj(p)]
    tekst_dijelovi = [p for p in polja if not je_broj(p)]

    puni_naziv = " ".join(tekst_dijelovi).strip()
    etiketa = puni_naziv.lower()

    if len(brojevi) >= 2:
      if "domac" in etiketa and dolasci_domaci is None:
        dolasci_domaci = ba_broj(brojevi[0])
        nocenja_domaci = ba_broj(brojevi[-1])
      elif "stran" in etiketa and dolasci_strani is None:
        dolasci_strani = ba_broj(brojevi[0])
        nocenja_strani = ba_broj(brojevi[-1])

    if (
        len(brojevi) >= 2
        and puni_naziv
        and not any(
            k in etiketa for k in ["ukupno", "domaci", "strani", "zemlja", "udio"]
        )
    ):

      dolasci = ba_broj(brojevi[0])
      nocenja = ba_broj(brojevi[-1])

      prosjek_dani = None
      if dolasci and nocenja and dolasci > 0:
        prosjek_dani = round(nocenja / dolasci, 2)

      list_zemalja.append({
          "godina": godina,
          "mjesec": mjesec,
          "zemlja": puni_naziv,
          "dolasci": dolasci,
          "nocenja": nocenja,
          "prosjecan_boravak": prosjek_dani,
      })

  dolasci_ukupno = None
  nocenja_ukupno = None
  if dolasci_domaci is not None and dolasci_strani is not None:
    dolasci_ukupno = dolasci_domaci + dolasci_strani
    nocenja_ukupno = nocenja_domaci + nocenja_strani

  zbirno = {
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

  return zbirno, list_zemalja


def main():
  putanje = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
  if not putanje:
    raise SystemExit(f"Nema PDF fajlova u '{PDF_DIR}/'.")

  redovi_zbirno = []
  redovi_zemlje = []

  for p in putanje:
    try:
      zbirno, zemlje = parsiraj_pdf(p)
      redovi_zbirno.append(zbirno)
      redovi_zemlje.extend(zemlje)
      print(f"[OK] {os.path.basename(p)} - pronađeno zemalja: {len(zemlje)}")
    except Exception as e:
      print(f"[GRESKA] {os.path.basename(p)}:", e)

  df_zbirno = pd.DataFrame(redovi_zbirno)
  df_zemlje = pd.DataFrame(redovi_zemlje)

  df_zbirno.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
  df_zemlje.to_csv(ZEMLJE_CSV, index=False, encoding="utf-8-sig")

  if not df_zemlje.empty:
    top_zemlje = df_zemlje.groupby("zemlja")["nocenja"].sum().nlargest(5)

    plt.figure(figsize=(8, 8))
    plt.pie(top_zemlje, labels=top_zemlje.index, autopct="%1.1f%%")
    plt.title("Top 5 zemalja po broju noćenja")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top_zemlje_pie.png"))
    plt.close()

  print("\nProces završen. Sačuvani CSV fajlovi.")


if __name__ == "__main__":
  main()