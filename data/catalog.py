"""
SKU Catalog — 45 real French products organized by warehouse zone.
Same catalog as Project 2 (Route Optimizer) and Project 1 (KPI Dashboard).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class SKU:
    sku_id: str
    name: str
    zone: str  # FROZEN | FRESH | AMBIENT | HEAVY
    base_demand: float  # units/day baseline
    unit_price: float  # € per unit (sale)
    unit_cost: float  # € per unit (purchase)
    margin: float  # € per unit
    supplier: str
    weight_kg: float
    category: str

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# FROZEN — 10 SKUs (zone -22°C)
# ---------------------------------------------------------------------------
FROZEN: List[SKU] = [
    SKU("FRZ-001", "Findus Épinards 1kg",          "FROZEN", 18.0, 3.95, 2.40, 1.55, "Findus France",     1.00, "Légumes"),
    SKU("FRZ-002", "Picard Frites 1kg",            "FROZEN", 24.0, 4.50, 2.80, 1.70, "Picard Surgelés",   1.00, "Frites"),
    SKU("FRZ-003", "Marie Lasagnes Bolognaise",    "FROZEN", 19.0, 5.20, 3.10, 2.10, "Marie SAS",         0.60, "Plats préparés"),
    SKU("FRZ-004", "Ben&Jerry's Cookie Dough",     "FROZEN", 15.0, 6.90, 3.80, 3.10, "Unilever",          0.46, "Glaces"),
    SKU("FRZ-005", "Bonduelle Maïs Surgelé 750g",  "FROZEN", 17.0, 3.20, 1.90, 1.30, "Bonduelle",         0.75, "Légumes"),
    SKU("FRZ-006", "Charal Steaks Hachés 10×100g", "FROZEN", 22.0, 9.90, 6.20, 3.70, "Charal",            1.00, "Viande"),
    SKU("FRZ-007", "Iglo Cabillaud Pané 450g",     "FROZEN", 16.0, 5.80, 3.50, 2.30, "Iglo France",       0.45, "Poisson"),
    SKU("FRZ-008", "Bonne Maman Tarte Pommes",     "FROZEN", 14.0, 5.50, 3.20, 2.30, "Bonne Maman",       0.55, "Desserts"),
    SKU("FRZ-009", "Thiriet Profiteroles 6 pcs",   "FROZEN", 12.0, 7.20, 4.20, 3.00, "Thiriet",           0.40, "Desserts"),
    SKU("FRZ-010", "Crème Brûlée Pierre Martinet", "FROZEN", 13.0, 4.80, 2.80, 2.00, "Pierre Martinet",   0.45, "Desserts"),
]

# ---------------------------------------------------------------------------
# FRESH — 10 SKUs (zone +2..+8°C)
# ---------------------------------------------------------------------------
FRESH: List[SKU] = [
    SKU("FRS-001", "Danone Activia Nature 4×125g", "FRESH", 34.0, 2.45, 1.45, 1.00, "Danone",          0.50, "Yaourts"),
    SKU("FRS-002", "Président Beurre Doux 250g",   "FRESH", 28.0, 2.80, 1.70, 1.10, "Lactalis",        0.25, "Beurre"),
    SKU("FRS-003", "Fleury Michon Jambon 4 tr.",   "FRESH", 26.0, 3.30, 2.00, 1.30, "Fleury Michon",   0.16, "Charcuterie"),
    SKU("FRS-004", "Elle&Vire Crème Liquide 33cl", "FRESH", 21.0, 1.95, 1.15, 0.80, "Elle & Vire",     0.33, "Crème"),
    SKU("FRS-005", "Yoplait Nature 16×125g",       "FRESH", 20.0, 3.90, 2.40, 1.50, "Yoplait",         2.00, "Yaourts"),
    SKU("FRS-006", "Leerdammer Tranches 200g",     "FRESH", 19.0, 3.60, 2.20, 1.40, "Bel Group",       0.20, "Fromage"),
    SKU("FRS-007", "Lactel Lait Demi-Écrémé 1L",   "FRESH", 42.0, 1.05, 0.65, 0.40, "Lactalis",        1.00, "Lait"),
    SKU("FRS-008", "Herta Lardons Fumés 200g",     "FRESH", 23.0, 2.85, 1.70, 1.15, "Herta (Nestlé)",  0.20, "Charcuterie"),
    SKU("FRS-009", "Paysan Breton Beurre 250g",    "FRESH", 25.0, 3.10, 1.90, 1.20, "Sill Paysan",     0.25, "Beurre"),
    SKU("FRS-010", "Chavroux Bûche Chèvre 150g",   "FRESH", 17.0, 2.95, 1.80, 1.15, "Savencia",        0.15, "Fromage"),
]

# ---------------------------------------------------------------------------
# AMBIENT — 15 SKUs (room temperature)
# ---------------------------------------------------------------------------
AMBIENT: List[SKU] = [
    SKU("AMB-001", "Panzani Pâtes Penne 500g",     "AMBIENT", 30.0, 1.45, 0.85, 0.60, "Panzani",         0.50, "Pâtes"),
    SKU("AMB-002", "Barilla Spaghetti n°5 500g",   "AMBIENT", 28.0, 1.55, 0.90, 0.65, "Barilla",         0.50, "Pâtes"),
    SKU("AMB-003", "Kellogg's Corn Flakes 500g",   "AMBIENT", 18.0, 3.20, 1.95, 1.25, "Kellogg's",       0.50, "Céréales"),
    SKU("AMB-004", "Nescafé Classic 200g",         "AMBIENT", 24.0, 7.90, 4.80, 3.10, "Nestlé",          0.20, "Café"),
    SKU("AMB-005", "Heinz Ketchup 570g",           "AMBIENT", 20.0, 2.95, 1.80, 1.15, "Heinz",           0.57, "Sauces"),
    SKU("AMB-006", "Nutella 750g",                 "AMBIENT", 45.0, 5.20, 3.10, 2.10, "Ferrero",         0.75, "Pâte à tartiner"),
    SKU("AMB-007", "Evian 6×1.5L",                 "AMBIENT", 38.0, 4.80, 2.90, 1.90, "Danone Waters",   9.00, "Eaux"),
    SKU("AMB-008", "Coca-Cola 6×33cl",             "AMBIENT", 52.0, 4.20, 2.50, 1.70, "Coca-Cola EP",    2.20, "Sodas"),
    SKU("AMB-009", "Pringles Original 175g",       "AMBIENT", 31.0, 2.80, 1.70, 1.10, "Kellogg's",       0.18, "Apéritif"),
    SKU("AMB-010", "Haribo Tagada 300g",           "AMBIENT", 29.0, 2.20, 1.30, 0.90, "Haribo",          0.30, "Confiserie"),
    SKU("AMB-011", "William Saurin Cassoulet 840g","AMBIENT", 16.0, 3.40, 2.05, 1.35, "William Saurin",  0.84, "Plats cuisinés"),
    SKU("AMB-012", "Amora Moutarde Fine 215g",     "AMBIENT", 19.0, 1.65, 1.00, 0.65, "Unilever",        0.22, "Condiments"),
    SKU("AMB-013", "Fleury Michon Rillettes 220g", "AMBIENT", 15.0, 2.60, 1.55, 1.05, "Fleury Michon",   0.22, "Charcuterie"),
    SKU("AMB-014", "Ricard 70cl",                  "AMBIENT", 12.0, 19.90,12.00, 7.90, "Pernod Ricard",  0.70, "Alcool"),
    SKU("AMB-015", "Orangina 1.5L",                "AMBIENT", 33.0, 1.95, 1.15, 0.80, "Suntory",         1.50, "Sodas"),
]

# ---------------------------------------------------------------------------
# HEAVY — 10 SKUs (lourd/encombrant)
# ---------------------------------------------------------------------------
HEAVY: List[SKU] = [
    SKU("HVY-001", "Evian 6×1.5L Pack",            "HEAVY", 26.0, 4.80, 2.90, 1.90, "Danone Waters",   9.00, "Eaux"),
    SKU("HVY-002", "Ariel Lessive Liquide 3L",     "HEAVY", 22.0, 14.90, 9.20, 5.70, "Procter & Gamble",3.00, "Lessive"),
    SKU("HVY-003", "Huile Lesieur Tournesol 5L",   "HEAVY", 17.0, 12.90, 8.10, 4.80, "Lesieur",         5.00, "Huile"),
    SKU("HVY-004", "Coca-Cola 24×33cl",            "HEAVY", 19.0, 14.80, 9.00, 5.80, "Coca-Cola EP",    8.80, "Sodas"),
    SKU("HVY-005", "Persil Lessive Poudre 4kg",    "HEAVY", 14.0, 13.50, 8.20, 5.30, "Henkel",          4.00, "Lessive"),
    SKU("HVY-006", "Sopalin Essuie-Tout 12 rlx",   "HEAVY", 23.0, 9.90, 6.00, 3.90, "Essity",          1.80, "Papier"),
    SKU("HVY-007", "Lotus WC 24 rouleaux",         "HEAVY", 25.0, 12.90, 7.80, 5.10, "Essity",          3.20, "Papier"),
    SKU("HVY-008", "Ricard 6×1L Pack",             "HEAVY", 11.0, 119.00,72.00,47.00, "Pernod Ricard",  6.00, "Alcool"),
    SKU("HVY-009", "Heineken 24×25cl",             "HEAVY", 21.0, 18.90,11.50, 7.40, "Heineken",        7.20, "Bière"),
    SKU("HVY-010", "Skip Lessive Capsules ×40",    "HEAVY", 16.0, 11.50, 7.00, 4.50, "Unilever",        1.20, "Lessive"),
]

ALL_SKUS: List[SKU] = FROZEN + FRESH + AMBIENT + HEAVY

ZONE_COLORS = {
    "FROZEN":  "#4FC3F7",
    "FRESH":   "#66BB6A",
    "AMBIENT": "#FFB347",
    "HEAVY":   "#BF7FFF",
}


def get_sku(sku_id: str) -> SKU | None:
    for s in ALL_SKUS:
        if s.sku_id == sku_id:
            return s
    return None


def by_zone(zone: str) -> List[SKU]:
    if zone == "ALL":
        return ALL_SKUS
    return [s for s in ALL_SKUS if s.zone == zone]


def catalog_df():
    import pandas as pd
    return pd.DataFrame([s.to_dict() for s in ALL_SKUS])
