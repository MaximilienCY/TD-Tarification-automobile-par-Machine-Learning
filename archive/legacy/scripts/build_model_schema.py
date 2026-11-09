"""Build presentation diagrams from verified main-model metadata.

Outputs: editable Excalidraw scene, four vector SVGs and a four-page PDF.
This script does not fit a model or modify any prediction.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "schemas"
PDF_OUT = ROOT / "output" / "pdf"
W, H = 1920, 1080
C = {
    "bg": "#f6f7f4", "white": "#ffffff", "ink": "#16354b",
    "muted": "#506777", "line": "#b9c6ca", "soft": "#e8edef",
    "freq": "#007b79", "freq_bg": "#e7f3f0",
    "sev": "#b45c36", "sev_bg": "#fbefe7",
    "dark": "#16354b", "light_ink": "#d8e4e8",
}
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("Schema", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("SchemaBold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))


def width(text, size, bold=False):
    return pdfmetrics.stringWidth(text, "SchemaBold" if bold else "Schema", size)


class Page:
    def __init__(self, number, slug, title, subtitle, section):
        self.number, self.slug = number, slug
        self.ops = []
        self.rect(0, 0, W, H, C["bg"], stroke=None)
        self.text(64, 36, "TARIFICATION AUTOMOBILE  /  MODÈLE PRINCIPAL", 19, C["muted"], bold=True)
        self.text(1856, 36, f"{number:02d} / 04", 19, C["muted"], align="right")
        self.text(64, 78, title, 44, bold=True)
        self.text(64, 141, subtitle, 24, C["muted"])
        self.line([(64, 187), (1856, 187)], C["line"], 1)
        self.text(64, 1038, section, 16, C["muted"])
        self.text(1856, 1038, "Source : modèle et diagnostics du dépôt · 20 octobre 2026", 16, C["muted"], align="right")

    def text(self, x, y, text, size=25, color=None, bold=False, align="left", leading=1.36, max_width=None, group=None):
        color = color or C["ink"]
        if max_width:
            lines = []
            for para in text.split("\n"):
                current = ""
                for word in para.split():
                    proposed = f"{current} {word}".strip()
                    if current and width(proposed, size, bold) > max_width:
                        lines.append(current)
                        current = word
                    else:
                        current = proposed
                lines.append(current)
            text = "\n".join(lines)
        tw = max((width(t, size, bold) for t in text.split("\n")), default=0)
        tx = x - (tw if align == "right" else tw / 2 if align == "center" else 0)
        th = len(text.split("\n")) * size * leading
        assert tx >= -1 and tx + tw <= W + 1, (self.slug, text, tx, tw)
        assert y + th < H + 1, (self.slug, text, y, th)
        self.ops.append(dict(type="text", x=tx, y=y, text=text, size=size,
                             color=color, bold=bold, leading=leading,
                             width=tw, height=th, group=group))

    def rect(self, x, y, w, h, fill, stroke=None, radius=16, sw=1.4, group=None):
        self.ops.append(dict(type="rect", x=x, y=y, w=w, h=h, fill=fill,
                             stroke=stroke, radius=radius, sw=sw, group=group))

    def line(self, points, color=None, sw=2.8, dash=False, arrow=False):
        self.ops.append(dict(type="line", points=points, color=color or C["muted"],
                             sw=sw, dash=dash, arrow=arrow))

    def arrow(self, points, color=None, dash=False):
        self.line(points, color=color, sw=3, dash=dash, arrow=True)

    def card(self, x, y, w, h, label, fill=None, color=None, group=None):
        group = group or f"card-{len(self.ops)}"
        self.rect(x, y, w, h, fill or C["white"], C["line"], group=group)
        self.text(x + 24, y + 21, label, 18, color or C["muted"], bold=True, group=group)
        return group

    def svg(self):
        items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                 f'<title>{html.escape(self.slug)}</title>',
                 '<style>text {font-family: "DejaVu Sans", Arial, sans-serif;}</style>']
        for op in self.ops:
            if op["type"] == "rect":
                items.append(f'<rect x="{op["x"]}" y="{op["y"]}" width="{op["w"]}" height="{op["h"]}" rx="{op["radius"]}" fill="{op["fill"]}" stroke="{op["stroke"] or "none"}" stroke-width="{op["sw"]}"/>')
            elif op["type"] == "text":
                for i, line in enumerate(op["text"].split("\n")):
                    y = op["y"] + op["size"] * .93 + i * op["size"] * op["leading"]
                    items.append(f'<text x="{op["x"]}" y="{y}" font-size="{op["size"]}" font-weight="{700 if op["bold"] else 400}" fill="{op["color"]}">{html.escape(line)}</text>')
            else:
                pts = " ".join(f"{x},{y}" for x, y in op["points"])
                dash = ' stroke-dasharray="9 7"' if op["dash"] else ""
                items.append(f'<polyline points="{pts}" fill="none" stroke="{op["color"]}" stroke-width="{op["sw"]}" stroke-linejoin="round"{dash}/>')
                if op["arrow"]:
                    items.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in arrow_head(op["points"]))}" fill="{op["color"]}"/>')
        items.append("</svg>")
        return "\n".join(items)

    def draw_pdf(self, doc):
        doc.saveState()
        doc.scale(.5, .5)
        for op in self.ops:
            if op["type"] == "rect":
                doc.setFillColor(HexColor(op["fill"]))
                if op["stroke"]:
                    doc.setStrokeColor(HexColor(op["stroke"]))
                    doc.setLineWidth(op["sw"])
                doc.roundRect(op["x"], H-op["y"]-op["h"], op["w"], op["h"], op["radius"],
                              stroke=int(bool(op["stroke"])), fill=1)
            elif op["type"] == "text":
                doc.setFillColor(HexColor(op["color"]))
                doc.setFont("SchemaBold" if op["bold"] else "Schema", op["size"])
                for i, line in enumerate(op["text"].split("\n")):
                    doc.drawString(op["x"], H-op["y"]-op["size"]*.93-i*op["size"]*op["leading"], line)
            else:
                doc.setStrokeColor(HexColor(op["color"]))
                doc.setLineWidth(op["sw"])
                doc.setDash([9, 7] if op["dash"] else [])
                path = doc.beginPath()
                path.moveTo(op["points"][0][0], H-op["points"][0][1])
                for x, y in op["points"][1:]:
                    path.lineTo(x, H-y)
                doc.drawPath(path)
                doc.setDash([])
                if op["arrow"]:
                    pts = arrow_head(op["points"])
                    path = doc.beginPath()
                    path.moveTo(pts[0][0], H-pts[0][1])
                    for x, y in pts[1:]:
                        path.lineTo(x, H-y)
                    path.close()
                    doc.setFillColor(HexColor(op["color"]))
                    doc.drawPath(path, stroke=0, fill=1)
        doc.restoreState()
        doc.showPage()


def arrow_head(points):
    a, b = points[-2:]
    theta = math.atan2(b[1]-a[1], b[0]-a[0])
    back = (b[0]-13*math.cos(theta), b[1]-13*math.sin(theta))
    return [b, (back[0]+6*math.sin(theta), back[1]-6*math.cos(theta)),
            (back[0]-6*math.sin(theta), back[1]+6*math.cos(theta))]


def overview():
    p = Page(1, "01_vue_ensemble", "Du contrat à la prime pure", "Deux modèles complémentaires : risque de sinistre × coût moyen si sinistre.", "VUE D’ENSEMBLE · Données → apprentissage → prédictions → soumission")
    p.card(64, 222, 450, 500, "01  DONNÉES ET PRÉPARATION")
    p.text(88, 276, "50 000 contrats train", 30, bold=True)
    p.text(88, 326, "45 785 clients\n2 917 sinistres : 5,834 %", 25)
    p.line([(88, 417), (490, 417)], C["line"], 1)
    p.text(88, 439, "Comprendre et préparer", 25, bold=True)
    p.text(88, 484, "Cibles, distributions, corrélations\nValeurs manquantes et anomalies\nEncodages adaptés aux modèles", 23)
    p.text(88, 605, "Identifiants, sexes et cibles\nexclus des prédicteurs.", 22, C["muted"])
    p.text(88, 679, "Traitements appris sur le train autorisé.", 18, C["muted"])

    p.card(620, 222, 600, 205, "02A  FRÉQUENCE · TOUS LES CONTRATS", C["freq_bg"], C["freq"])
    p.text(644, 277, "Quelle probabilité de sinistre ?", 29, bold=True)
    p.text(644, 325, "Logistique + correction par courbes\nSortie : p(X), comprise entre 0 et 1", 25)
    p.text(644, 394, "Variables étendues, dont le département.", 20, C["freq"])

    p.card(620, 500, 600, 222, "02B  GRAVITÉ · SINISTRES POSITIFS", C["sev_bg"], C["sev"])
    p.text(644, 555, "Quel coût moyen si sinistre ?", 29, bold=True)
    p.text(644, 605, "37,5 % Gamma + 37,5 % Gamma corrigé XGB\n+ 25 % Gamma corrigé Ridge", 21)
    p.text(644, 679, "Sortie : s(X), montant conditionnel en euros.", 22, C["sev"])

    p.arrow([(514, 472), (564, 472), (564, 324), (620, 324)], C["freq"])
    p.arrow([(564, 472), (564, 612), (620, 612)], C["sev"])

    p.card(1336, 222, 520, 126, "03  TEST : NOUVEAUX PROSPECTS")
    p.text(1360, 272, "50 000 lignes, sans cibles connues", 25)
    p.text(1360, 311, "Paramètres figés après l’apprentissage.", 20, C["muted"])
    p.arrow([(1596, 348), (1596, 404)], C["muted"], dash=True)
    p.rect(1336, 404, 520, 242, C["dark"], radius=18)
    p.text(1360, 430, "04  COMBINAISON POUR CHAQUE PROSPECT", 17, C["light_ink"], bold=True)
    p.text(1596, 478, "prime(X) = p(X) × s(X)", 34, C["white"], bold=True, align="center")
    p.text(1360, 542, "Coût attendu de la période observée\nLa probabilité reste continue.\nAucun seuil oui/non avant multiplication.", 23, C["light_ink"])
    p.arrow([(1220, 325), (1280, 325), (1280, 470), (1336, 470)], C["freq"])
    p.arrow([(1220, 612), (1280, 612), (1280, 580), (1336, 580)], C["sev"])
    p.text(1243, 286, "p", 26, C["freq"], bold=True)
    p.text(1243, 640, "s", 26, C["sev"], bold=True)

    p.card(64, 802, 1156, 202, "VALIDATION DES DEUX MODÈLES AVANT LE RÉAPPRENTISSAGE FINAL")
    p.text(88, 851, "3 plis séparés par client → prédictions sur les clients exclus", 27, bold=True)
    p.text(88, 897, "Recette fixe : RMSE de prime 541,916 € · calibration · erreurs par segment", 25)
    p.text(88, 947, "Le test CSV n’est pas la validation : ses coûts ne sont pas connus localement.", 23, C["muted"])

    p.card(1336, 752, 520, 252, "05  SOUMISSION")
    p.text(1360, 806, "submission.csv", 32, bold=True)
    p.text(1360, 861, "Colonnes : index, pred\nIndex : 50 000 à 99 999\npred = prime prédite en euros", 25)
    p.text(1360, 968, "La RMSE est un score global, pas une colonne.", 19, C["muted"])
    p.arrow([(1596, 646), (1596, 752)], C["muted"])
    return p


def frequency():
    p = Page(2, "02_detail_frequence", "Fréquence : prédire une probabilité", "Apprentissage sur les 50 000 contrats · cible : 1 si sinistre, 0 sinon.", "FRÉQUENCE · Régression logistique et correction de logit")
    p.card(64, 224, 452, 563, "ENTRÉES ET TRANSFORMATIONS")
    p.text(88, 277, "Caractéristiques du contrat,\ndu véhicule et des conducteurs", 23, bold=True)
    p.text(88, 367, "Numériques", 22, C["freq"], bold=True)
    p.text(88, 405, "Médianes apprises sur le train\nlog1p de certaines variables\nStandardisation", 23)
    p.text(88, 520, "Catégorielles", 22, C["freq"], bold=True)
    p.text(88, 558, "One-hot, modalités rares regroupées\nMarque / modèle / département :\nencodage cible lissé hors client", 22)
    p.text(88, 687, "Plus de 3 variables au total.\nNi identifiants, ni sexes, ni cibles\nparmi les prédicteurs.", 22, C["muted"])

    p.card(610, 224, 536, 211, "01  BASE LOGISTIQUE", C["freq_bg"], C["freq"])
    p.text(634, 278, "p₀ = sigmoid(b + β × X préparé)", 26, bold=True)
    p.text(634, 328, "Pénalisation L2 : C = 0,1\nApprentissage par log-loss\nSans rééquilibrage des classes", 23)
    p.arrow([(516, 395), (560, 395), (560, 329), (610, 329)], C["freq"])

    p.card(610, 510, 536, 277, "02  CORRECTION δ(X)", C["freq_bg"], C["freq"])
    p.text(634, 562, "Courbes souples d’âge du véhicule", 25, bold=True)
    p.text(634, 608, "Splines : 4 nœuds, degré 2\nContrat + usage + âge × contrat\nInteractions multipliées par 0,5\nLog-loss + pénalisation L2 : α = 100", 23)
    p.text(634, 750, "Apprise à partir de p₀ calculées hors client.", 20, C["freq"])
    p.arrow([(878, 435), (878, 510)], C["freq"])

    p.rect(1242, 265, 614, 258, C["freq"], radius=18)
    p.text(1266, 293, "03  PROBABILITÉ FINALE", 18, C["white"], bold=True)
    p.text(1549, 350, "p = sigmoid(logit(p₀) + δ)", 32, C["white"], bold=True, align="center")
    p.text(1266, 411, "La correction augmente ou réduit le risque\nselon l’âge, le contrat et l’usage.\nLa sortie reste une probabilité entre 0 et 1.", 23, C["white"])
    p.arrow([(1146, 650), (1196, 650), (1196, 448), (1242, 448)], C["freq"])
    p.arrow([(1146, 329), (1242, 329)], C["freq"])

    p.card(1242, 577, 614, 210, "ÉVITER LA FUITE D’INFORMATION")
    p.text(1266, 629, "Un même client reste dans le même pli.", 25, bold=True)
    p.text(1266, 675, "Encodages : 5 plis internes par client.\nCorrection : probabilités issues de 3 sous-plis.\nImputations et encodages appris sans les exclus.", 22)

    p.rect(64, 855, 1792, 149, C["white"], C["line"])
    p.text(88, 876, "DIAGNOSTIC SUR CLIENTS EXCLUS", 18, C["muted"], bold=True)
    p.text(88, 919, "AUC : 0,6533", 29, C["freq"], bold=True)
    p.text(470, 919, "Log-loss : 0,21393", 29, C["freq"], bold=True)
    p.text(1020, 919, "Brier : 0,05407", 29, C["freq"], bold=True)
    p.text(88, 968, "Calibration vérifiée. La matrice de confusion illustre un seuil ; elle ne pilote pas la prime.", 21, C["muted"])
    return p


def severity():
    p = Page(3, "03_detail_gravite", "Gravité : Gamma et deux corrections", "Apprentissage sur 2 917 sinistres positifs · cible : montant du sinistre en euros.", "GRAVITÉ · Une moyenne conditionnelle pour chaque prospect")
    p.card(64, 224, 452, 548, "DONNÉES D’APPRENTISSAGE")
    p.text(88, 275, "Sinistres positifs uniquement", 25, bold=True)
    p.text(88, 325, "Coût moyen : 1 771,84 €\nMédiane : 1 236,00 €\nMaximum : 21 826,96 €", 23)
    p.text(88, 441, "Pas de suppression des gros sinistres.", 21, C["muted"])
    p.line([(88, 494), (492, 494)], C["line"], 1)
    p.text(88, 517, "Pour XGBoost et Ridge", 23, C["sev"], bold=True)
    p.text(88, 561, "Gamma et probabilité auxiliaires\ncalculés hors client.\nPoids ∝ p_aux × Gamma_aux²", 23)
    p.text(88, 684, "La probabilité auxiliaire sert aux poids ;\nelle diffère de la fréquence finale.", 20, C["muted"])
    p.text(88, 741, "ε = 10⁻⁸ : plancher numérique positif.", 18, C["muted"])

    p.card(580, 224, 388, 328, "01  GAMMA · ANCRE", C["sev_bg"], C["sev"])
    p.text(604, 276, "g = exp(b + β × X)", 28, bold=True)
    p.text(604, 331, "Contrat\nÂge du véhicule\nUtilisation", 24)
    p.text(604, 447, "Lien log · α = 0,1\nDéviance Gamma + L2\nMontant toujours positif", 23)

    p.card(1012, 224, 400, 328, "02  GAMMA CORRIGÉ XGBOOST", C["sev_bg"], C["sev"])
    p.text(1036, 276, "sX = max(g × ratio, ε)", 25, bold=True)
    p.text(1036, 328, "Mêmes 3 variables que Gamma\nCible : coût / Gamma_aux", 21)
    p.text(1036, 410, "240 arbres · profondeur 2\nLearning rate : 0,025\nL2 : 100 · L1 : 1\nPoids min. enfant : 100", 22)

    p.card(1456, 224, 400, 328, "03  GAMMA CORRIGÉ RIDGE", C["sev_bg"], C["sev"])
    p.text(1480, 276, "sR = g × max(1 + c, ε)", 25, bold=True)
    p.text(1480, 327, "Cible de c : coût / Gamma_aux − 1\nRidge : α = 10 · splines degré 2", 20)
    p.text(1480, 411, "Variables étendues :\nconducteurs, technique/prix,\nmarque/modèle, département…\nFin de vente exclue.", 22)
    p.arrow([(516, 382), (580, 382)], C["sev"])

    p.rect(580, 635, 1276, 137, C["sev"], radius=18)
    p.text(604, 655, "04  GRAVITÉ RETENUE : MOYENNE PONDÉRÉE DES MONTANTS", 18, C["white"], bold=True)
    p.text(1218, 700, "s(X) = 0,375 × g(X) + 0,375 × sX(X) + 0,25 × sR(X)", 30, C["white"], bold=True, align="center")
    for x in (774, 1212, 1656):
        p.arrow([(x, 552), (x, 635)], C["sev"])

    p.card(64, 840, 452, 164, "DIAGNOSTIC SUR SINISTRÉS EXCLUS")
    p.text(88, 891, "RMSE gravité : 1 452,48 €", 26, C["sev"], bold=True)
    p.text(88, 941, "Cible conditionnelle, différente de la prime.", 19, C["muted"])
    p.card(580, 840, 1276, 164, "EXEMPLE PÉDAGOGIQUE FICTIF")
    p.text(604, 889, "g = 2 000 € · sX = 2 400 € · sR = 2 600 € → s = 2 300 €", 26, bold=True)
    p.text(604, 942, "Avec p = 8 % : prime = 0,08 × 2 300 = 184 €. La gravité est estimée pour tous les prospects.", 21)
    return p


def validation():
    p = Page(4, "04_validation_et_limites", "Valider, interpréter et connaître les limites", "Le modèle retenu est une recette figée ; les diagnostics ne constituent pas un nouveau test indépendant.", "VALIDATION · Clients exclus, erreurs de prime et limites d’usage")
    p.card(64, 224, 530, 221, "01  SÉPARER PAR CLIENT")
    p.text(88, 277, "3 plis de validation", 28, bold=True)
    p.text(88, 330, "Environ 2/3 pour apprendre, 1/3 exclu.\nChaque client appartient à un seul pli.\nOn recommence pour couvrir tout le train.", 23)
    p.card(670, 224, 560, 221, "02  APPRENDRE SANS LES CLIENTS EXCLUS")
    p.text(694, 277, "Préparation + fréquence + gravité", 26, bold=True)
    p.text(694, 330, "Toutes les étapes sont ajustées sur le train.\nLes auxiliaires sont calculés hors client.\nOn applique ensuite la recette aux exclus.", 22)
    p.card(1306, 224, 550, 221, "03  ÉVALUER PUIS FIGER")
    p.text(1330, 277, "Comparer prime et coût réel", 27, bold=True)
    p.text(1330, 330, "Calibration, résidus, segments et variables.\nPuis réapprentissage sur tout le train.\nTest sans cibles : aucune RMSE locale de test.", 21)
    p.arrow([(594, 334), (670, 334)])
    p.arrow([(1230, 334), (1306, 334)])

    p.rect(64, 511, 1166, 204, C["dark"], radius=18)
    p.text(88, 536, "MÉTRIQUE PRINCIPALE : RMSE DE LA PRIME", 18, C["light_ink"], bold=True)
    p.text(88, 579, "RMSE = √ moyenne[(prime prédite − coût réel)²]", 30, C["white"], bold=True)
    p.text(88, 638, "541,916 €", 36, C["white"], bold=True)
    p.text(380, 642, "contre 541,971 € avant la correction de fréquence", 24, C["light_ink"])
    p.text(88, 678, "Coût réel = 0 sans sinistre, montant observé sinon. Gain local faible : 0,055 €.", 20, C["light_ink"])

    p.card(1306, 511, 550, 204, "SCORE PUBLIC KAGGLE COMMUNIQUÉ")
    p.text(1330, 564, "1 365,657", 38, bold=True)
    p.text(1330, 622, "Reçu utilisateur du 12 octobre 2026.\nPopulation différente du diagnostic local.\nScore privé inconnu ; non revérifié sur Kaggle.", 21, C["muted"])

    p.card(64, 784, 530, 220, "FORCES")
    p.text(88, 836, "Décomposition métier interprétable\nPlis et encodages séparés par client\nRégularisation et sorties positives\n50 000 primes reproduites et contrôlées", 23)
    p.card(670, 784, 560, 220, "FAIBLESSES")
    p.text(694, 836, "Seulement 2 917 sinistres pour apprendre\nGros coûts : forte influence sur la RMSE\nMAE légèrement plus élevée après correction\nDonnées déjà explorées, gain très faible", 22)
    p.card(1306, 784, 550, 220, "PROCHAINES ÉTAPES")
    p.text(1330, 836, "Nouvelle cohorte de validation\nClarifier dates, unités et exposition\nSuivre calibration et dérives par segment\nExercice prêt ; tarif commercial à approfondir", 21)
    return p


def make_excalidraw(pages):
    elements = []
    serial = 0

    def base(kind, x, y, w, h, page, **extra):
        nonlocal serial
        serial += 1
        return dict(id=f"schema-{serial:04d}", type=kind, x=x, y=y,
                    width=w, height=h, angle=0, strokeColor=C["ink"],
                    backgroundColor="transparent", fillStyle="solid", strokeWidth=1,
                    strokeStyle="solid", roughness=0, opacity=100, groupIds=[],
                    frameId=f"page-{page.number}", roundness=None,
                    seed=serial * 117, version=1, versionNonce=serial * 13,
                    isDeleted=False, boundElements=None, updated=1792447200000,
                    created=1792447200000, link=None, locked=False, **extra)

    for page in pages:
        oy = (page.number-1)*(H+140)
        frame = base("frame", 0, oy, W, H, page)
        frame.update(id=f"page-{page.number}", frameId=None, name=page.slug,
                     strokeColor=C["line"])
        elements.append(frame)
        for op in page.ops:
            if op["type"] == "rect":
                item = base("rectangle", op["x"], oy+op["y"], op["w"], op["h"], page)
                item.update(backgroundColor=op["fill"], strokeColor=op["stroke"] or op["fill"],
                            strokeWidth=op["sw"], roundness={"type": 3} if op["radius"] else None)
            elif op["type"] == "text":
                # Helvetica/Arial is native to Excalidraw; use generous width so
                # font substitution does not wrap an otherwise valid line.
                item = base("text", op["x"], oy+op["y"], op["width"]+3, op["height"], page)
                item.update(text=op["text"], originalText=op["text"],
                            strokeColor=op["color"], fontSize=op["size"], fontFamily=2,
                            textAlign="left", verticalAlign="top", containerId=None,
                            autoResize=True, lineHeight=op["leading"])
            else:
                xs, ys = zip(*op["points"])
                x, y = min(xs), min(ys)
                item = base("arrow" if op["arrow"] else "line", x, oy+y,
                            max(xs)-x, max(ys)-y, page)
                item.update(points=[[px-x, py-y] for px, py in op["points"]],
                            strokeColor=op["color"], strokeWidth=op["sw"],
                            strokeStyle="dashed" if op["dash"] else "solid",
                            startBinding=None, endBinding=None,
                            startArrowhead=None, endArrowhead="arrow" if op["arrow"] else None,
                            elbowed=False, lastCommittedPoint=None)
            if op.get("group"):
                item["groupIds"] = [f"p{page.number}-{op['group']}"]
            elements.append(item)
    return dict(type="excalidraw", version=2, source="https://excalidraw.com",
                elements=elements, files={},
                appState={"viewBackgroundColor": "#ffffff", "gridSize": None,
                          "scrollX": 45, "scrollY": 30, "zoom": {"value": .65},
                          "frameRendering": {"enabled": True, "clip": True, "name": True, "outline": True}})


def main():
    meta = json.loads((ROOT / "reports/final_model.json").read_text())
    recipe = meta["recipe"]
    assert recipe["name"] == "frequency_curves_1365_657"
    assert recipe["severity"]["gamma"]["weight"] == .375
    assert recipe["severity"]["xgboost_ratio"]["weight"] == .375
    assert recipe["severity"]["ridge_ratio"]["weight"] == .25
    assert meta["fitting_rows"] == 50000 and meta["positive_claims"] == 2917
    assert recipe["severity"]["xgboost_ratio"]["trees"] == 240
    OUT.mkdir(parents=True, exist_ok=True)
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    pages = [overview(), frequency(), severity(), validation()]
    for page in pages:
        (OUT / f"{page.slug}.svg").write_text(page.svg(), encoding="utf-8")
    pdf_file = PDF_OUT / "schema_modele_principal.pdf"
    doc = canvas.Canvas(str(pdf_file), pagesize=(W/2, H/2), pageCompression=1)
    doc.setTitle("Tarification automobile - Schéma du modèle principal")
    doc.setAuthor("Projet de tarification automobile")
    doc.setSubject("Fréquence, gravité, combinaison et validation de la recette frequency_curves_1365_657")
    for page in pages:
        page.draw_pdf(doc)
    doc.save()
    scene = make_excalidraw(pages)
    (OUT / "schema_modele_principal.excalidraw").write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "recipe": recipe["name"], "source_model_sha256": meta["model_sha256"],
        "pages": [page.slug for page in pages], "excalidraw_elements": len(scene["elements"]),
        "metadata_sha256": hashlib.sha256((ROOT / "reports/final_model.json").read_bytes()).hexdigest(),
        "note": "Schémas explicatifs uniquement ; aucun modèle réajusté, aucune prédiction modifiée.",
    }
    (OUT / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"pdf": str(pdf_file), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
