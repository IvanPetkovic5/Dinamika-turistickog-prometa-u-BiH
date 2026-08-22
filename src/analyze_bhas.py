import os
import re
import glob
import unicodedata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pdfplumber
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11


def razmakni_labele(ax, tacke_x, tacke_y, labele, fontsize=7,
                     offset_pocetni=14, iteracije=400, korak=0.9,
                     padding=3.0, min_pomak_za_liniju=6.0):
    """Force-based raspoređivanje labela u piksel-koordinatama sa optimizovanim brojem iteracija."""
    fig = ax.figure
    fig.canvas.draw()

    trans = ax.transData
    trans_inv = ax.transData.inverted()

    n = len(labele)
    if n == 0:
        return []

    tacke_px = np.array([trans.transform((x, y)) for x, y in zip(tacke_x, tacke_y)])

    renderer = fig.canvas.get_renderer()
    sirine, visine = [], []
    for lab in labele:
        t = ax.text(0, 0, lab, fontsize=fontsize, alpha=0)
        bbox = t.get_window_extent(renderer=renderer)
        sirine.append(bbox.width + padding)
        visine.append(bbox.height + padding)
        t.remove()
    sirine = np.array(sirine)
    visine = np.array(visine)

    poluprecnik_tacke = 7.0
    bbox_axes = ax.get_window_extent()
    margina = 4.0

    zlatni_ugao = np.pi * (3 - np.sqrt(5))
    label_px = tacke_px.copy().astype(float)
    for i in range(n):
        ugao = i * zlatni_ugao
        label_px[i, 0] += offset_pocetni * np.cos(ugao)
        label_px[i, 1] += offset_pocetni * np.sin(ugao)

    for it in range(iteracije):
        pomjeraji = np.zeros_like(label_px)
        hladjenje = max(0.25, 1.0 - it / iteracije)

        for i in range(n):
            for j in range(i + 1, n):
                dx = label_px[i, 0] - label_px[j, 0]
                dy = label_px[i, 1] - label_px[j, 1]
                preklop_x = (sirine[i] + sirine[j]) / 2 - abs(dx)
                preklop_y = (visine[i] + visine[j]) / 2 - abs(dy)
                if preklop_x > 0 and preklop_y > 0:
                    if preklop_x < preklop_y:
                        sila = preklop_x * korak * hladjenje
                        smjer = 1 if dx >= 0 else -1
                        pomjeraji[i, 0] += smjer * sila / 2
                        pomjeraji[j, 0] -= smjer * sila / 2
                    else:
                        sila = preklop_y * korak * hladjenje
                        smjer = 1 if dy >= 0 else -1
                        pomjeraji[i, 1] += smjer * sila / 2
                        pomjeraji[j, 1] -= smjer * sila / 2

            for j in range(n):
                if j == i:
                    continue
                dx = label_px[i, 0] - tacke_px[j, 0]
                dy = label_px[i, 1] - tacke_px[j, 1]
                dist = np.hypot(dx, dy)
                min_dist = poluprecnik_tacke + max(sirine[i], visine[i]) / 2
                if dist < min_dist and dist > 1e-6:
                    sila = (min_dist - dist) * korak * 0.6 * hladjenje
                    pomjeraji[i, 0] += (dx / dist) * sila
                    pomjeraji[i, 1] += (dy / dist) * sila

        for i in range(n):
            nazad = (tacke_px[i] - label_px[i]) * 0.01
            pomjeraji[i] += nazad

        label_px += pomjeraji

        for i in range(n):
            pola_w, pola_h = sirine[i] / 2, visine[i] / 2
            label_px[i, 0] = np.clip(label_px[i, 0],
                                      bbox_axes.x0 + pola_w + margina,
                                      bbox_axes.x1 - pola_w - margina)
            label_px[i, 1] = np.clip(label_px[i, 1],
                                      bbox_axes.y0 + pola_h + margina,
                                      bbox_axes.y1 - pola_h - margina)

        if np.abs(pomjeraji).max() < 0.02:
            break

    finalne_pozicije = []
    for i, lab in enumerate(labele):
        lx, ly = trans_inv.transform(label_px[i])
        finalne_pozicije.append((lx, ly))

        pomak_px = np.hypot(*(label_px[i] - tacke_px[i]))
        arrowprops = None
        if pomak_px > min_pomak_za_liniju:
            arrowprops = dict(arrowstyle="-", color="#999999", lw=0.6, alpha=0.7,
                               shrinkA=2, shrinkB=4)

        ax.annotate(
            lab,
            xy=(tacke_x[i], tacke_y[i]), xycoords="data",
            xytext=(lx, ly), textcoords="data",
            fontsize=fontsize, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
            arrowprops=arrowprops,
            zorder=5,
        )

    return finalne_pozicije


BROJ_POLJE_RE = re.compile(r"^-?\d[\d.,]*$|^-$")

def _je_broj_polje(polje):
    return bool(BROJ_POLJE_RE.match(polje.strip()))

def ba_broj(tekst):
    if tekst is None:
        return None
    t = str(tekst).strip()
    t = t.replace("\u202f", " ").replace("\xa0", " ")
    t = t.replace(" ", "")
    t = t.replace(".", "")
    t = t.replace(",", ".")
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
    "ukupno", "domaći turisti", "domaci turisti", "strani turisti",
    "ukupno strani turisti", "sve zemlje", "ostale zemlje",
    "ostale evropske zemlje", "ostale afričke zemlje", 
    "ostale sjeverno-američke zemlje", "ostale sjevernoameričke zemlje",
    "ostale zemlje južne i srednje amerike", "ostale azijske zemlje",
    "ostale zemlje okeanije"
}

def normalizuj_zemlju(naziv):
    n = naziv.strip()
    kljuc = n.lower()
    kljuc = re.sub(r"\s*\d\)\s*$", "", kljuc).strip()
    return ALIASI_ZEMALJA.get(kljuc, n.strip())

def _normalizuj_tekst(tekst):
    if not tekst:
        return ""
    tekst = tekst.strip().lower()
    nfkd = unicodedata.normalize("NFD", tekst)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def redovi_tabele_po_koordinatama(putanja_pdf):
    svi_redovi = []
    with pdfplumber.open(putanja_pdf) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=2, use_text_flow=True)
            if not words:
                continue
            
            words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
            
            linije = []
            trenutna_linija = []
            
            for w in words_sorted:
                if not trenutna_linija:
                    trenutna_linija.append(w)
                else:
                    avg_top = sum(x["top"] for x in trenutna_linija) / len(trenutna_linija)
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
                    
                    zadnji_tekst = trenutno_polje.strip().split()[-1] if trenutno_polje.strip() else ""
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
    svi_redovi = redovi_tabele_po_koordinatama(putanja_pdf)
    
    dolasci_domaci = nocenja_domaci = dolasci_strani = nocenja_strani = None
    
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

    dolasci_ukupno = (dolasci_domaci + dolasci_strani) if (dolasci_domaci is not None and dolasci_strani is not None) else None
    nocenja_ukupno = (nocenja_domaci + nocenja_strani) if (nocenja_domaci is not None and nocenja_strani is not None) else None
    
    prosjecan_boravak = round(nocenja_ukupno / dolasci_ukupno, 2) if (dolasci_ukupno and dolasci_ukupno > 0 and nocenja_ukupno is not None) else None
    udio_stranih_pct = round((nocenja_strani / nocenja_ukupno) * 100, 2) if (nocenja_strani is not None and nocenja_ukupno and nocenja_ukupno > 0) else None

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
        "udio_stranih_%": udio_stranih_pct,
    }

    redovi_zemalja = []
    nije_zemlja_norm = {_normalizuj_tekst(z) for z in NIJE_ZEMLJA}
    
    for polja in svi_redovi:
        rez = parsiraj_red_zemlje_iz_polja(polja)
        if rez is None:
            continue

        naziv_sirovi, brojevi = rez
        naziv_norm = _normalizuj_tekst(naziv_sirovi)

        if naziv_norm in nije_zemlja_norm or "ukupno" in naziv_norm or "domac" in naziv_norm or "stran" in naziv_norm:
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
            print(f"  {status}: {zbir['fajl']} -> godina={zbir['godina']}, mjesec={zbir['mjesec_kraja']}, zemalja={len(zemlje)}")
            zbirni_redovi.append(zbir)
            svi_redovi_zemalja.extend(zemlje)
        except Exception as e:
            print(f"  GREŠKA: {p}: {e}")

    df_ukupno = pd.DataFrame(zbirni_redovi).sort_values(["godina", "mjesec_kraja"]).reset_index(drop=True)
    df_zemlje = pd.DataFrame(svi_redovi_zemalja)

    df_ukupno = df_ukupno.drop_duplicates().reset_index(drop=True)
    df_zemlje = df_zemlje.drop_duplicates().reset_index(drop=True)

    df_ukupno.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    df_zemlje.to_csv(DATASET_CSV, index=False, encoding="utf-8-sig")

    eda_linije = [
        "EDA IZVJEŠTAJ - za sekciju 'Dataset & metodologija'",
        "=" * 60,
        f"Broj obrađenih PDF izvještaja: {len(putanje)}",
        f"Vremenski raspon: {df_ukupno['godina'].min():.0f}-{df_ukupno['godina'].max():.0f}",
        f"Dataset 'po zemljama': {len(df_zemlje)} redova x {df_zemlje.shape[1]} kolona",
        f"Broj jedinstvenih zemalja porijekla: {df_zemlje['zemlja'].nunique()}",
        "",
        "PROVJERI INTEGRITETA PODATAKA:",
        f"Duplirani redovi u datasetu zemalja: {df_zemlje.duplicated().sum()}",
        "Broj nedostajućih vrijednosti po kolonama (df_zemlje):",
        df_zemlje.isnull().sum().to_string(),
        "",
        "PRVIH 5 REDOVA (df_zemlje.head()):",
        df_zemlje.head().to_string(),
        "",
        "OPISNA STATISTIKA (df_zemlje.describe()):",
        df_zemlje.describe().to_string(float_format=lambda x: f"{x:,.2f}"),
        "",
        "TOP 10 ZEMALJA PO UKUPNIM NOĆENJIMA:",
        str(df_zemlje.groupby("zemlja")["nocenja"].sum().sort_values(ascending=False).head(10))
    ]

    with open(EDA_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(eda_linije))

    df_godisnji = df_ukupno[df_ukupno["mjesec_kraja"] == 12].copy()
    if df_godisnji.empty:
        df_godisnji = df_ukupno.loc[df_ukupno.groupby("godina")["mjesec_kraja"].idxmax()].copy()

    god_agg = df_godisnji.groupby("godina")[["nocenja_domaci", "nocenja_strani"]].max().dropna()
    
    if not god_agg.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = np.arange(len(god_agg.index))
        sirina = 0.35

        bar1 = ax.bar(x - sirina/2, god_agg["nocenja_domaci"] / 1e6, sirina, label="Domaći turisti", color="#1f77b4", alpha=0.85)
        bar2 = ax.bar(x + sirina/2, god_agg["nocenja_strani"] / 1e6, sirina, label="Strani turisti", color="#ff7f0e", alpha=0.85)

        ax.set_title("Slika 1: Godišnji trend noćenja turista u BiH (januar - decembar)", fontsize=13, pad=12)
        ax.set_xlabel("Godina")
        ax.set_ylabel("Broj noćenja (u milionima)")
        ax.set_xticks(x)
        ax.set_xticklabels(god_agg.index.astype(int))
        ax.legend(frameon=True, facecolor="white", edgecolor="none")
        
        for bar in bar1:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}M", ha='center', va='bottom', fontsize=8.5)
        for bar in bar2:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}M", ha='center', va='bottom', fontsize=8.5)

        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "slika1_trend_nocenja.png"), dpi=300)
        plt.close(fig)

    if not df_godisnji.empty:
        najnovija_godina = df_godisnji["godina"].max()
        df_klaster = df_zemlje[(df_zemlje["godina"] == najnovija_godina) & (df_zemlje["mjesec_kraja"] == 12)].copy()

        if df_klaster.empty:
            df_klaster = df_zemlje[df_zemlje["godina"] == df_zemlje["godina"].max()].copy()

        df_klaster = df_klaster.dropna(subset=["nocenja", "struktura_%", "prosjecan_boravak"])
        df_klaster = df_klaster[df_klaster["struktura_%"] >= 0.05].copy()

        if len(df_klaster) >= 6:
            df_klaster["log_struktura"] = np.log10(df_klaster["struktura_%"])
            features = df_klaster[["log_struktura", "prosjecan_boravak"]].copy()
            scaler = StandardScaler()
            X = scaler.fit_transform(features)

            k = 3
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            df_klaster["klaster"] = km.fit_predict(X)

            opis = (df_klaster.groupby("klaster")[["struktura_%", "prosjecan_boravak"]]
                    .mean().sort_values("struktura_%"))
            redoslijed = opis.index.tolist()
            imena_klastera = {
                redoslijed[0]: "Mala evropska tržišta",
                redoslijed[1]: "Veliki/regionalni izvori",
                redoslijed[2]: "Daleka, duga putovanja",
            }
            if opis.loc[redoslijed[1], "prosjecan_boravak"] > opis.loc[redoslijed[2], "prosjecan_boravak"]:
                imena_klastera[redoslijed[1]], imena_klastera[redoslijed[2]] = (
                    imena_klastera[redoslijed[2]], imena_klastera[redoslijed[1]]
                )

            fig, ax = plt.subplots(figsize=(14, 8.5))
            boje = ["#2ca02c", "#d62728", "#9467bd"]

            nocenja_min = df_klaster["nocenja"].min()
            nocenja_max = df_klaster["nocenja"].max()

            if nocenja_max > nocenja_min:
                def velicina(n):
                    return 40 + 330 * (
                        np.sqrt(n - nocenja_min) /
                        np.sqrt(nocenja_max - nocenja_min)
                    )
            else:
                def velicina(n):
                    return 120

            for i, kl in enumerate(redoslijed):
                sub = df_klaster[df_klaster["klaster"] == kl]

                ax.scatter(
                    sub["struktura_%"],
                    sub["prosjecan_boravak"],
                    s=velicina(sub["nocenja"]),
                    color=boje[i],
                    label=imena_klastera[kl],
                    edgecolor="white",
                    linewidth=1.2,
                    alpha=0.78,
                    zorder=3
                )

            ax.set_xscale("log")
            ax.set_xlim(0.04, 20)
            ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}%"))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())

            ax.axvline(1, color="#777777", linestyle="--", linewidth=0.8, alpha=0.45, zorder=1)
            ax.axhline(2, color="#777777", linestyle="--", linewidth=0.8, alpha=0.45, zorder=1)

            ax.set_title(
                f"Slika 2: Segmentacija emitivnih tržišta turista ({najnovija_godina:.0f}.)",
                fontsize=16,
                fontweight="bold",
                pad=16
            )
            ax.set_xlabel("Udio u noćenjima stranih turista (%) — logaritamska skala", fontsize=12, labelpad=10)
            ax.set_ylabel("Prosječan broj noćenja po dolasku", fontsize=12, labelpad=10)

            ax.grid(True, which="major", axis="both", alpha=0.18, linewidth=0.8)
            ax.grid(True, which="minor", axis="x", alpha=0.08, linewidth=0.5)

            legenda = ax.legend(
                title="Klasteri (KMeans, k=3)",
                frameon=True,
                facecolor="white",
                edgecolor="#dddddd",
                framealpha=0.95,
                loc="upper left",
                fontsize=10,
                title_fontsize=10.5
            )
            legenda.get_frame().set_linewidth(0.8)

            razmakni_labele(
                ax,
                df_klaster["struktura_%"].values,
                df_klaster["prosjecan_boravak"].values,
                df_klaster["zemlja"].values,
                fontsize=8,
                korak=0.85,
                iteracije=400,
                offset_pocetni=13,
                min_pomak_za_liniju=7.0
            )

            ax.text(
                0.985, 0.025,
                "Veličina tačke ∝ broj noćenja",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9, style="italic", color="#555555"
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.margins(x=0.02, y=0.08)

            fig.tight_layout()
            fig.savefig(
                os.path.join(PLOTS_DIR, "slika2_klasteri_zemalja.png"),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig)

            df_klaster.to_csv(os.path.join(OUT_DIR, "klasteri_zemalja.csv"), index=False, encoding="utf-8-sig")

    print("\n--- ZAVRŠENO USPJEŠNO ---")

if __name__ == "__main__":
    main()