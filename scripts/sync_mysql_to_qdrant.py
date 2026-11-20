import datetime
import html
import json
import os
import re
import sys
from typing import Any

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set UTF-8 encoding for standard output and error (e.g. for Khmer text)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pymysql
import pymysql.cursors
from qdrant_client.models import PointStruct

from app.core.config import settings
from app.providers.factory import get_datasource_provider
from app.services.vector_service import vector_service
from app.services.embedder import embedder

BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 16
ALLOWED_DOCUMENT_SOURCES = {
    "fa_cw_happening_now",
    "fa_cw_campaign",
    "fa_spa",
    "fa_eat_drink",
    "fa_tourpackages",
    "fa_guides",
    "fa_tourguide",
    "fa_hotels",
    "fa_private_villa",
    "fa_cw_things_to_do",
    "fa_cw_destinations",
    "fa_cw_practical_tips",
    "fa_car_services",
    "fa_cars",
    "fa_transportation",
}


def _existing_document_payloads() -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    offset = None
    while True:
        points, offset = vector_service.client.scroll(
            collection_name=vector_service.collection_name,
            offset=offset,
            limit=256,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            existing[str(point.id)] = payload
        if offset is None:
            return existing


# =====================================================================
# Database Helpers & Graceful Connection
# =====================================================================
def get_db_connection() -> pymysql.Connection:
    """Connect through the configured datasource provider."""
    return get_datasource_provider().connect()


BASE_UPLOAD_URL = settings.MEDIA_BASE_URL.rstrip("/")


def slugify(text: Any) -> str:
    """Generate a clean URL-friendly slug from text."""
    if not text:
        return ""
    s = re.sub(r"[^\w\s-]", "", str(text).lower()).strip()
    s = re.sub(r"[\s_-]+", "-", s)
    return s.strip("-")


def resolve_thumbnail(image_val: Any, images_val: Any = None) -> str:
    """Extract and build full thumbnail URL from image or images field."""
    path = ""
    if image_val and str(image_val).strip() and str(image_val).strip() != "-":
        path = str(image_val).strip()
    elif images_val and str(images_val).strip():
        parts = [p.strip() for p in str(images_val).split(",") if p.strip()]
        if parts:
            path = parts[0]

    if not path:
        return ""

    if "," in path:
        path = path.split(",")[0].strip()

    if path.startswith("http://") or path.startswith("https://"):
        return path

    if not path.startswith("/"):
        path = "/" + path

    return f"{BASE_UPLOAD_URL}{path}" if BASE_UPLOAD_URL else path


def get_table_columns(cursor: pymysql.cursors.DictCursor, table_name: str) -> set[str]:
    """Safely discover column names present on a table."""
    try:
        cursor.execute(f"SHOW COLUMNS FROM `{table_name}`;")
        return {row["Field"] for row in cursor.fetchall()}
    except Exception:
        return set()


def build_active_where(columns: set[str], prefix: str = "") -> str:
    """Build a safe WHERE clause filtering active and non-deleted records."""
    p = f"{prefix}." if prefix else ""
    clauses = []
    if "status" in columns:
        clauses.append(f"{p}status IN ('1', 'normal', 'public')")
    if "deletetime" in columns:
        clauses.append(f"({p}deletetime IS NULL OR {p}deletetime = 0)")
    return (" WHERE " + " AND ".join(clauses)) if clauses else ""


# =====================================================================
# Text Cleaning & Extraction Utilities
# =====================================================================
def clean_html(text: Any) -> str:
    """Strip HTML tags, decode entities, and normalize whitespace."""
    if not text:
        return ""
    s = str(text)
    # Replace block level breaks with spaces
    s = re.sub(r"<(br|p|div|li)[^>]*>", " ", s, flags=re.IGNORECASE)
    # Remove remaining HTML tags
    s = re.sub(r"<[^>]+>", " ", s)
    # Unescape HTML entities
    s = html.unescape(s)
    # Collapse multiple whitespace characters
    s = re.sub(r"\s+", " ", s).strip()
    return s


def format_time_val(val: Any) -> str:
    """Format MySQL timedelta/time/string values safely."""
    if val is None:
        return ""
    if isinstance(val, datetime.timedelta):
        total_seconds = int(val.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours:02d}:{minutes:02d}"
    return str(val).strip()


def format_price_json(val: Any) -> str:
    """Format tour package pricejson field into readable text."""
    if not val:
        return ""
    if isinstance(val, dict):
        return ", ".join(f"{k}: ${v}" for k, v in val.items() if v)
    if isinstance(val, str):
        try:
            data = json.loads(val)
            if isinstance(data, dict):
                return ", ".join(f"{k}: ${v}" for k, v in data.items() if v)
            elif isinstance(data, list):
                return ", ".join(str(x) for x in data if x)
        except Exception:
            return clean_html(val)
    return str(val)


def parse_sections(raw_val: Any) -> list[str]:
    """Parse JSON or structured content containing headings and descriptions."""
    if not raw_val:
        return []
    items = []
    if isinstance(raw_val, str):
        try:
            parsed = json.loads(raw_val)
            if isinstance(parsed, list):
                items = parsed
        except Exception:
            cleaned = clean_html(raw_val)
            return [cleaned] if cleaned else []
    elif isinstance(raw_val, list):
        items = raw_val

    results = []
    for item in items:
        if isinstance(item, dict):
            heading = clean_html(item.get("heading", ""))
            content = clean_html(item.get("content", ""))
            if heading and content:
                results.append(f"{heading}: {content}")
            elif content:
                results.append(content)
            elif heading:
                results.append(heading)
        elif isinstance(item, str):
            c = clean_html(item)
            if c:
                results.append(c)
    return results


def resolve_categories(cat_ids_raw: Any, categories_map: dict[str, str]) -> list[str]:
    """Resolve comma-separated category IDs into category names."""
    if not cat_ids_raw:
        return []
    ids = [c.strip() for c in str(cat_ids_raw).split(",") if c.strip()]
    return [categories_map[cid] for cid in ids if cid in categories_map]


# =====================================================================
# Dedicated Sync Functions for Each Entity / Table
# =====================================================================

# ---------------------------------------------------------------------
# 1. fa_cw_happening_now (Events & Festivals) -> doc ID: happening_{id}
# ---------------------------------------------------------------------
def sync_happening_now(
    cur: pymysql.cursors.DictCursor,
    existing_tables: set[str],
    resolve_destination: Any,
) -> list[tuple[str, str, dict]]:
    """Synchronize fa_cw_happening_now (Events & Festivals)."""
    docs = []
    if "fa_cw_happening_now" not in existing_tables:
        return docs

    cols = get_table_columns(cur, "fa_cw_happening_now")
    where = build_active_where(cols)
    cur.execute(f"SELECT * FROM fa_cw_happening_now{where};")
    events = cur.fetchall()

    for r in events:
        eid = r["id"]
        title = clean_html(r.get("title") or f"Event #{eid}")
        title_km = clean_html(r.get("title_km") or "")
        loc_id = r.get("location_id")
        dest_en, dest_km = resolve_destination(loc_id)

        lines = [f"Event / Festival / Happening Now: {title}"]
        if title_km:
            lines[0] += f" ({title_km})"

        loc_str = f"Location / Destination: {dest_en}"
        if dest_km:
            loc_str += f" ({dest_km})"
        lines.append(loc_str)

        venue = clean_html(r.get("location_map") or "")
        if venue:
            lines.append(f"Venue / Address: {venue}")

        if r.get("is_free") == 1:
            lines.append("Admission: Free Admission")

        overview = clean_html(r.get("overview") or "")
        overview_km = clean_html(r.get("overview_km") or "")
        if overview:
            lines.append(f"Overview: {overview}")
        if overview_km:
            lines.append(f"ទិដ្ឋភាពទូទៅ: {overview_km}")

        content = clean_html(r.get("content") or "")
        content_km = clean_html(r.get("content_km") or "")
        if content and content != overview:
            lines.append(f"Details & Schedule: {content}")
        if content_km and content_km != overview_km:
            lines.append(f"ព័ត៌មានលម្អិត និងកាលវិភាគ: {content_km}")

        tc = clean_html(r.get("term_condition") or "")
        tc_km = clean_html(r.get("term_condition_km") or "")
        if tc:
            lines.append(f"Terms & Conditions: {tc}")
        if tc_km:
            lines.append(f"លក្ខខណ្ឌ: {tc_km}")

        text = "\n".join(lines)
        meta = {
            "source": "fa_cw_happening_now",
            "item_id": int(eid),
            "entity_type": "happening",
            "id": int(eid),
            "name": str(title),
            "destination": str(dest_en),
        }
        docs.append((f"happening_{eid}", text, meta))

    print(f"-> fa_cw_happening_now: Processed {len(docs)} active records.")
    return docs


# ---------------------------------------------------------------------
# 2. fa_cw_campaign (Promotions & Offers) -> doc ID: campaign_{id}
# ---------------------------------------------------------------------
def sync_campaigns(
    cur: pymysql.cursors.DictCursor,
    existing_tables: set[str],
) -> list[tuple[str, str, dict]]:
    """Synchronize fa_cw_campaign (Promotions & Special Offers)."""
    docs = []
    if "fa_cw_campaign" not in existing_tables:
        return docs

    # Pre-cache campaign items & discounts
    camp_items_map: dict[int, list[dict]] = {}
    if "fa_cw_campaign_item" in existing_tables:
        cur.execute("""
            SELECT ci.id, ci.campaign_id, ci.service_type, ci.discount, rd.discount as room_discount
            FROM fa_cw_campaign_item ci
            LEFT JOIN fa_cw_campaign_room_discount rd ON rd.campaign_item_id = ci.id;
        """)
        for item in cur.fetchall():
            camp_items_map.setdefault(item["campaign_id"], []).append(item)

    # Pre-cache active promo codes
    promocodes: list[dict] = []
    if "fa_promocode" in existing_tables:
        p_cols = get_table_columns(cur, "fa_promocode")
        p_where = build_active_where(p_cols)
        cur.execute(f"SELECT * FROM fa_promocode{p_where};")
        promocodes = cur.fetchall()

    cols = get_table_columns(cur, "fa_cw_campaign")
    where = build_active_where(cols)
    cur.execute(f"SELECT * FROM fa_cw_campaign{where};")
    campaigns = cur.fetchall()

    for c in campaigns:
        cid = c["id"]
        name = clean_html(c.get("name") or f"Campaign #{cid}")

        lines = [f"Promotion & Special Offer: {name}"]
        sd = c.get("start_date")
        ed = c.get("end_date")
        if sd and ed:
            try:
                sd_str = sd.strftime("%B %d, %Y") if hasattr(sd, "strftime") else str(sd)
                ed_str = ed.strftime("%B %d, %Y") if hasattr(ed, "strftime") else str(ed)
                lines.append(f"Validity Period: {sd_str} to {ed_str}")
            except Exception:
                lines.append(f"Validity: {sd} - {ed}")

        lines.append("Campaign Status: Active Promotion")

        items = camp_items_map.get(cid, [])
        services = set()
        max_discount = 0.0
        for itm in items:
            st = (itm.get("service_type") or "").replace("wtd_", "").replace("cw_", "").replace("_", " ").title()
            if st:
                services.add(st)
            disc = itm.get("discount") or itm.get("room_discount")
            if disc and float(disc) > max_discount:
                max_discount = float(disc)

        if max_discount > 0:
            lines.append(f"Special Discount: Up to {max_discount:.0f}% off")
        if services:
            lines.append(f"Applicable Categories: {', '.join(sorted(services))}")

        # Link promo codes if any
        matching_codes = [p["code"] for p in promocodes if p.get("code")]
        if matching_codes:
            lines.append(f"Promo Code(s): {', '.join(matching_codes)}")

        lines.append(
            f"Details: Book during the {name} promotional campaign to enjoy special seasonal discounts "
            f"across participating hotels, private villas, activities, and tours in Cambodia."
        )

        text = "\n".join(lines)
        meta = {
            "source": "fa_cw_campaign",
            "item_id": int(cid),
            "entity_type": "campaign",
            "id": int(cid),
            "name": str(name),
            "destination": "Cambodia",
        }
        docs.append((f"campaign_{cid}", text, meta))

    print(f"-> fa_cw_campaign: Processed {len(docs)} active records.")
    return docs


# ---------------------------------------------------------------------
# 3. fa_spa (Spas & Wellness) -> doc ID: spa_{id}
# ---------------------------------------------------------------------
def sync_spas(
    cur: pymysql.cursors.DictCursor,
    existing_tables: set[str],
    resolve_destination: Any,
    categories_map: dict[str, str],
) -> list[tuple[str, str, dict]]:
    """Synchronize fa_spa records with spa treatments, facilities, and policies."""
    docs = []
    if "fa_spa" not in existing_tables:
        return docs

    treatments_map: dict[str, list[dict]] = {}
    if "fa_spa_menuitems" in existing_tables:
        cur.execute(
            "SELECT * FROM fa_spa_menuitems "
            "WHERE status IN ('public', '1', 'normal') "
            "ORDER BY id ASC;"
        )
        for m in cur.fetchall():
            sid = str(m.get("spa_id") or "").strip()
            if not sid:
                continue
            treatments_map.setdefault(sid, []).append({
                "title": clean_html(m.get("title") or ""),
                "price": float(m["price"]) if m.get("price") and float(m["price"]) > 0 else None,
                "short_description": clean_html(m.get("short_description") or ""),
            })

    cols = get_table_columns(cur, "fa_spa")
    where = build_active_where(cols)
    cur.execute(f"SELECT * FROM fa_spa{where};")
    for r in cur.fetchall():
        sid = r["id"]
        title = clean_html(r.get("title") or f"Spa #{sid}")
        title_km = clean_html(r.get("title_km") or "")
        loc_id = r.get("locations_id") or r.get("location_id")
        dest_en, dest_km = resolve_destination(loc_id)

        lines = [f"Spa & Wellness Center: {title}"]
        if title_km:
            lines[0] += f" ({title_km})"

        loc_str = f"Address / Location: {dest_en}"
        if dest_km:
            loc_str += f" ({dest_km})"
        lines.append(loc_str)

        cats = resolve_categories(r.get("categories_id"), categories_map)
        if cats:
            lines.append(f"Category / Facilities: {', '.join(cats)}")

        ot = format_time_val(r.get("open_time"))
        ct = format_time_val(r.get("closed_time"))
        if ot or ct:
            lines.append(f"Opening Hours: Open {ot} - Closed {ct}")

        short_desc = clean_html(r.get("short_description") or "")
        short_desc_km = clean_html(r.get("short_description_km") or "")
        if short_desc:
            lines.append(f"Overview: {short_desc}")
        if short_desc_km:
            lines.append(f"ទិដ្ឋភាពទូទៅ: {short_desc_km}")

        desc = clean_html(r.get("description") or "")
        desc_km = clean_html(r.get("description_km") or "")
        if desc and desc != short_desc:
            lines.append(f"Details & Policies: {desc}")
        if desc_km and desc_km != short_desc_km:
            lines.append(f"ព័ត៌មានលម្អិត និងគោលការណ៍: {desc_km}")

        items = treatments_map.get(str(sid), [])
        if items:
            prices = [item["price"] for item in items if item.get("price")]
            if prices:
                lines.append(f"Starting Price: ${min(prices):.2f}")
            lines.append("Available Treatments & Massages:")
            for item in items[:12]:
                i_line = f"- {item['title']}"
                if item.get("price"):
                    i_line += f": ${item['price']:.2f}"
                if item.get("short_description"):
                    i_line += f" ({item['short_description']})"
                lines.append(i_line)

        text = "\n".join(lines)
        meta = {
            "source": "fa_spa",
            "item_id": int(sid),
            "entity_type": "spa",
            "id": int(sid),
            "name": str(title),
            "destination": str(dest_en),
        }
        docs.append((f"spa_{sid}", text, meta))

    print(f"-> fa_spa: Processed {len(docs)} active records.")
    return docs


# ---------------------------------------------------------------------
# 4. fa_eat_drink (Dining & Restaurants) -> doc ID: dining_{id}
# ---------------------------------------------------------------------
def sync_dining(
    cur: pymysql.cursors.DictCursor,
    existing_tables: set[str],
    resolve_destination: Any,
    categories_map: dict[str, str],
) -> list[tuple[str, str, dict]]:
    """Synchronize fa_eat_drink records with cuisine, pricing, hours, and menu dishes."""
    docs = []
    if "fa_eat_drink" not in existing_tables:
        return docs

    # Pre-cache menu items by eat_drink_id
    menu_map: dict[int, list[dict]] = {}
    if "fa_eat_drink_menuitems" in existing_tables:
        cur.execute(
            "SELECT * FROM fa_eat_drink_menuitems "
            "WHERE status IN ('public', '1', 'normal') "
            "ORDER BY weigh DESC, id ASC;"
        )
        for m in cur.fetchall():
            ed_id = m.get("eat_drink_id")
            if not ed_id:
                continue
            menu_map.setdefault(ed_id, []).append({
                "title": clean_html(m.get("title") or ""),
                "price": float(m["price"]) if m.get("price") and float(m["price"]) > 0 else None,
                "short_description": clean_html(m.get("short_description") or ""),
            })

    cols = get_table_columns(cur, "fa_eat_drink")
    where = build_active_where(cols)
    cur.execute(f"SELECT * FROM fa_eat_drink{where};")
    restaurants = cur.fetchall()

    for r in restaurants:
        rid = r["id"]
        title = clean_html(r.get("title") or f"Restaurant #{rid}")
        title_km = clean_html(r.get("title_km") or "")
        loc_id = r.get("locations_id") or r.get("location_id")
        dest_en, dest_km = resolve_destination(loc_id)

        lines = [f"Dining & Restaurant: {title}"]
        if title_km:
            lines[0] += f" ({title_km})"

        loc_str = f"Address / Location: {dest_en}"
        if dest_km:
            loc_str += f" ({dest_km})"
        lines.append(loc_str)

        cats = resolve_categories(r.get("categories_id"), categories_map)
        if cats:
            lines.append(f"Cuisine & Category: {', '.join(cats)}")

        ot = format_time_val(r.get("open_time"))
        ct = format_time_val(r.get("closed_time"))
        if ot or ct:
            lines.append(f"Opening Hours: Open {ot} - Closed {ct}")

        price = r.get("price")
        if price and float(price) > 0:
            lines.append(f"Price Range / Starting Price: ${float(price):.2f}")
        booking_fee = r.get("booking_table_price")
        if booking_fee and float(booking_fee) > 0:
            lines.append(f"Table Reservation Fee: ${float(booking_fee):.2f}")

        short_desc = clean_html(r.get("short_description") or "")
        short_desc_km = clean_html(r.get("short_description_km") or "")
        if short_desc:
            lines.append(f"Overview: {short_desc}")
        if short_desc_km:
            lines.append(f"ទិដ្ឋភាពទូទៅ: {short_desc_km}")

        desc = clean_html(r.get("description") or "")
        desc_km = clean_html(r.get("description_km") or "")
        if desc and desc != short_desc:
            lines.append(f"Details: {desc}")
        if desc_km and desc_km != short_desc_km:
            lines.append(f"ព័ត៌មានលម្អិត: {desc_km}")

        dishes = menu_map.get(rid, [])
        if dishes:
            lines.append("Signature Dishes & Menu Highlights:")
            for dish in dishes[:10]:
                d_line = f"- {dish['title']}"
                if dish.get("price"):
                    d_line += f": ${dish['price']:.2f}"
                if dish.get("short_description"):
                    d_line += f" ({dish['short_description']})"
                lines.append(d_line)

        text = "\n".join(lines)
        meta = {
            "source": "fa_eat_drink",
            "item_id": int(rid),
            "entity_type": "dining",
            "id": int(rid),
            "name": str(title),
            "destination": str(dest_en),
        }
        docs.append((f"dining_{rid}", text, meta))

    print(f"-> fa_eat_drink: Processed {len(docs)} active records.")
    return docs


# ---------------------------------------------------------------------
# 5. fa_tourpackages (Tour Packages) -> doc ID: tour_{id}
# ---------------------------------------------------------------------
def sync_tourpackages(
    cur: pymysql.cursors.DictCursor,
    existing_tables: set[str],
    resolve_destination: Any,
    categories_map: dict[str, str],
) -> list[tuple[str, str, dict]]:
    """Synchronize fa_tourpackages records with itinerary, inclusions, and pricing."""
    docs = []
    if "fa_tourpackages" not in existing_tables:
        return docs

    cols = get_table_columns(cur, "fa_tourpackages")
    where = build_active_where(cols)
    cur.execute(f"SELECT * FROM fa_tourpackages{where};")
    for r in cur.fetchall():
        tid = r["id"]
        title = clean_html(r.get("title") or f"Tour Package #{tid}")
        title_km = clean_html(r.get("title_km") or "")
        loc_id = r.get("locations_id") or r.get("location_id")
        dest_en, dest_km = resolve_destination(loc_id)

        lines = [f"Tour Package & Excursion: {title}"]
        if title_km:
            lines[0] += f" ({title_km})"

        loc_str = f"Location / Destination: {dest_en}"
        if dest_km:
            loc_str += f" ({dest_km})"
        lines.append(loc_str)

        cats = resolve_categories(r.get("categories_id"), categories_map)
        if cats:
            lines.append(f"Tour Category: {', '.join(cats)}")

        dur = r.get("duration")
        if dur and str(dur) != "0":
            lines.append(f"Duration: {dur} day(s)")
        gs = r.get("groupsize")
        if gs and int(gs) > 0:
            lines.append(f"Group Size: Up to {gs} people")
        shift = clean_html(r.get("time_shift") or "")
        if shift:
            lines.append(f"Schedule / Shift: {shift}")

        pricing = format_price_json(r.get("pricejson"))
        if pricing:
            lines.append(f"Pricing: {pricing}")

        short_desc = clean_html(r.get("short_description") or "")
        short_desc_km = clean_html(r.get("short_description_km") or "")
        if short_desc:
            lines.append(f"Overview: {short_desc}")
        if short_desc_km:
            lines.append(f"ទិដ្ឋភាពទូទៅ: {short_desc_km}")

        hl = clean_html(r.get("highlight") or "")
        if hl:
            lines.append(f"Highlights: {hl}")

        inc = clean_html(r.get("include_content") or "")
        if inc:
            lines.append(f"Inclusions: {inc}")

        desc = clean_html(r.get("description") or "")
        desc_km = clean_html(r.get("description_km") or "")
        if desc and desc != short_desc:
            lines.append(f"Itinerary & Details: {desc}")
        if desc_km and desc_km != short_desc_km:
            lines.append(f"ព័ត៌មានលម្អិត និងកាលវិភាគ: {desc_km}")

        text = "\n".join(lines)
        meta = {
            "source": "fa_tourpackages",
            "item_id": int(tid),
            "entity_type": "tour_package",
            "id": int(tid),
            "name": str(title),
            "destination": str(dest_en),
        }
        docs.append((f"tour_{tid}", text, meta))

    print(f"-> fa_tourpackages: Processed {len(docs)} active records.")
    return docs


# ---------------------------------------------------------------------
# 6. fa_guide / fa_guides / fa_tourguide -> doc ID: guide_{id}
# ---------------------------------------------------------------------
def sync_guides(
    cur: pymysql.cursors.DictCursor,
    existing_tables: set[str],
    resolve_destination: Any,
) -> list[tuple[str, str, dict]]:
    """Synchronize tour guides (fa_guides, fa_guide_packages, and fa_tourguide services)."""
    docs = []

    # A. Guides from fa_guides & fa_guide_packages
    if "fa_guides" in existing_tables:
        pkg_by_guide: dict[int, list[dict]] = {}
        if "fa_guide_packages" in existing_tables:
            gp_cols = get_table_columns(cur, "fa_guide_packages")
            gp_where = build_active_where(gp_cols)
            cur.execute(f"SELECT * FROM fa_guide_packages{gp_where};")
            for p in cur.fetchall():
                pkg_by_guide.setdefault(p["guide_id"], []).append(p)

        g_cols = get_table_columns(cur, "fa_guides")
        g_where = build_active_where(g_cols)
        cur.execute(f"SELECT * FROM fa_guides{g_where};")
        for g in cur.fetchall():
            gid = g["id"]
            slug = g.get("slug") or ""
            title = slug.replace("-", " ").title() if slug else f"Tour Guide #{gid}"

            # Infer primary spoken languages from slug
            lang = "English"
            if "vietnamese" in slug.lower():
                lang = "Vietnamese"
            elif "chinese" in slug.lower():
                lang = "Chinese"
            elif "french" in slug.lower():
                lang = "French"

            lines = [f"Licensed Tour Guide: {title}"]
            lines.append(f"Spoken Languages: {lang}, Khmer")
            lines.append("Location / Destination: Siem Reap, Angkor Wat, Cambodia")

            shifts = g.get("time_shift")
            if shifts:
                try:
                    s_dict = json.loads(shifts)
                    lines.append(f"Available Time Shifts: {', '.join(s_dict.values())}")
                except Exception:
                    pass

            short_desc = clean_html(g.get("short_desc") or "")
            if short_desc:
                lines.append(f"Bio / Overview: {short_desc}")

            desc = clean_html(g.get("description") or "")
            if desc and desc != short_desc:
                lines.append(f"Specialties & Details: {desc}")

            pkgs = pkg_by_guide.get(gid, [])
            if pkgs:
                lines.append("Guiding Rates & Packages:")
                for p in pkgs:
                    p_name = clean_html(p.get("title") or "Tour")
                    pr = p.get("price")
                    disc = p.get("discount_percent")
                    p_str = f"- {p_name}: ${float(pr):.2f}"
                    if disc and float(disc) > 0:
                        p_str += f" ({float(disc):.0f}% discount available)"
                    lines.append(p_str)

            text = "\n".join(lines)
            meta = {
                "source": "fa_guides",
                "item_id": int(gid),
                "entity_type": "guide",
                "id": int(gid),
                "name": str(title),
                "destination": "Siem Reap",
            }
            docs.append((f"guide_{gid}", text, meta))

    # B. Tour Guide Services & Itineraries from fa_tourguide
    if "fa_tourguide" in existing_tables:
        tg_cols = get_table_columns(cur, "fa_tourguide")
        tg_where = build_active_where(tg_cols)
        cur.execute(f"SELECT * FROM fa_tourguide{tg_where};")
        for tg in cur.fetchall():
            tgid = tg["id"]
            title = clean_html(tg.get("title") or f"Tour Guide Service #{tgid}")
            loc_id = tg.get("location_id")
            dest_en, dest_km = resolve_destination(loc_id)

            lines = [f"Tour Guide Excursion & Service: {title}"]
            lines.append(f"Location / Destination: {dest_en}")
            lines.append("Spoken Languages: English (Professional English-Speaking Guide), Khmer")

            pj = tg.get("price_json")
            if pj:
                try:
                    p_data = json.loads(pj)
                    lines.append(f"Guiding Rates: {', '.join(f'${v} per {k}' for k, v in p_data.items())}")
                except Exception:
                    lines.append(f"Guiding Rates: {pj}")

            sc = clean_html(tg.get("short_content") or "")
            if sc:
                lines.append(f"Bio & Overview: {sc}")

            inc = clean_html(tg.get("include") or "")
            if inc:
                lines.append(f"Inclusions: {inc}")

            exc = clean_html(tg.get("exclude") or "")
            if exc:
                lines.append(f"Exclusions: {exc}")

            itin = clean_html(tg.get("itinerary") or "")
            if itin and itin != sc:
                lines.append(f"Itinerary & Temple Specialties: {itin}")

            text = "\n".join(lines)
            meta = {
                "source": "fa_tourguide",
                "item_id": int(tgid),
                "entity_type": "guide",
                "id": int(tgid),
                "name": str(title),
                "destination": str(dest_en),
            }
            # Keep guide_{id} prefix unique across guide records
            docs.append((f"guide_tg_{tgid}", text, meta))

    print(f"-> fa_guides / fa_tourguide: Processed {len(docs)} active records.")
    return docs


# ---------------------------------------------------------------------
# Existing Tables (Hotels, Villas, Activities, Destinations, Tips, Transit)
# ---------------------------------------------------------------------
def sync_hotels(
    cur: pymysql.cursors.DictCursor,
    existing_tables: set[str],
    resolve_destination: Any,
    hotel_rooms_map: dict[int, list[dict]],
    hotel_campaign_discounts: dict[int, float] = None,
) -> list[tuple[str, str, dict]]:
    """Synchronize fa_hotels records with room configurations and stay card metadata."""
    docs = []
    if "fa_hotels" not in existing_tables:
        return docs

    h_cols = get_table_columns(cur, "fa_hotels")
    h_where = build_active_where(h_cols)
    cur.execute(f"SELECT * FROM fa_hotels{h_where};")
    hotels = cur.fetchall()

    for h in hotels:
        hid = h["id"]
        title = clean_html(h.get("title") or h.get("name") or f"Hotel #{hid}")
        title_km = clean_html(h.get("title_km") or h.get("name_km") or "")
        loc_id = h.get("locations_id") or h.get("location_id")
        dest_en, dest_km = resolve_destination(loc_id)

        raw_slug = str(h.get("slug") or "").strip()
        slug = raw_slug or slugify(title) or f"hotel-{hid}"
        thumbnail = resolve_thumbnail(h.get("image"), h.get("images"))

        rooms = hotel_rooms_map.get(hid, [])
        price = h.get("price")
        starting_price = 0.0
        if price and float(price) > 0:
            starting_price = float(price)
        elif rooms:
            r_prices = [rm["price"] for rm in rooms if rm.get("price") and float(rm["price"]) > 0]
            if r_prices:
                starting_price = float(min(r_prices))

        discount_percent = (hotel_campaign_discounts or {}).get(hid, 0.0)
        if discount_percent > 0 and starting_price > 0:
            discount_price = round(starting_price * (1.0 - (discount_percent / 100.0)), 2)
        else:
            discount_price = 0.0

        category_type = "hotels"
        card_url = settings.CARD_URL_TEMPLATE.format(
            base_url=settings.MEDIA_BASE_URL.rstrip("/"),
            category_type="hotels",
            slug=slug,
            id=hid,
        )

        lines = [f"Hotel: {title}"]
        if title_km:
            lines[0] += f" ({title_km})"

        loc_str = f"Destination / Location: {dest_en}"
        if dest_km:
            loc_str += f" ({dest_km})"
        lines.append(loc_str)

        if starting_price > 0:
            lines.append(f"Starting Price: ${starting_price:.2f} per night.")

        short_desc = clean_html(h.get("short_content") or h.get("intro") or "")
        short_desc_km = clean_html(h.get("short_content_km") or "")
        if short_desc:
            lines.append(f"Overview: {short_desc}")
        if short_desc_km:
            lines.append(f"ទិដ្ឋភាពទូទៅ: {short_desc_km}")

        content = clean_html(h.get("content") or h.get("description") or "")
        content_km = clean_html(h.get("content_km") or "")
        if content and content != short_desc:
            lines.append(f"Details: {content}")
        if content_km and content_km != short_desc_km:
            lines.append(f"ព័ត៌មានលម្អិត: {content_km}")

        extra_info = clean_html(h.get("extra_info") or "")
        if extra_info:
            lines.append(f"Amenities & Features: {extra_info}")

        policy = clean_html(h.get("policy") or "")
        if policy:
            lines.append(f"Hotel Policy: {policy}")

        if rooms:
            lines.append("Available Room Types:")
            for rm in rooms:
                rm_line = f"- {rm['name']}"
                if rm["name_km"]:
                    rm_line += f" ({rm['name_km']})"
                parts = []
                if rm.get("max_adults") or rm.get("max_kids"):
                    cap = []
                    if rm.get("max_adults"):
                        cap.append(f"{rm['max_adults']} adults")
                    if rm.get("max_kids"):
                        cap.append(f"{rm['max_kids']} children")
                    parts.append("Capacity: " + ", ".join(cap))
                if rm.get("price") and float(rm["price"]) > 0:
                    parts.append(f"Price: ${float(rm['price']):.2f}")
                if rm.get("description"):
                    parts.append(rm["description"])
                if parts:
                    rm_line += ": " + "; ".join(parts)
                lines.append(rm_line)

        text = "\n".join(lines)
        meta = {
            "source": "fa_hotels",
            "item_id": int(hid),
            "entity_type": "hotel",
            "category_type": category_type,
            "id": int(hid),
            "title": str(title),
            "name": str(title),
            "slug": str(slug),
            "thumbnail": str(thumbnail),
            "price": float(starting_price),
            "discount_price": float(discount_price),
            "url": str(card_url),
            "destination": str(dest_en),
        }
        docs.append((f"hotel_{hid}", text, meta))

    print(f"-> fa_hotels: Processed {len(docs)} active records.")
    return docs


def sync_villas(
    cur: pymysql.cursors.DictCursor,
    existing_tables: set[str],
    resolve_destination: Any,
    villa_rooms_map: dict[int, list[dict]],
    villa_campaign_discounts: dict[int, float] = None,
) -> list[tuple[str, str, dict]]:
    """Synchronize fa_private_villa records and stay card metadata."""
    docs = []
    if "fa_private_villa" not in existing_tables:
        return docs

    v_cols = get_table_columns(cur, "fa_private_villa")
    v_where = build_active_where(v_cols)
    cur.execute(f"SELECT * FROM fa_private_villa{v_where};")
    villas = cur.fetchall()

    for v in villas:
        vid = v["id"]
        title = clean_html(v.get("title") or v.get("name") or f"Villa #{vid}")
        title_km = clean_html(v.get("title_km") or v.get("name_km") or "")
        loc_id = v.get("locations_id") or v.get("location_id")
        dest_en, dest_km = resolve_destination(loc_id)

        raw_slug = str(v.get("slug") or "").strip()
        slug = raw_slug or slugify(title) or f"villa-{vid}"
        thumbnail = resolve_thumbnail(v.get("image"), v.get("images"))

        rooms = villa_rooms_map.get(vid, [])
        starting_price = 0.0
        if rooms:
            r_prices = [rm["price"] for rm in rooms if rm.get("price") and float(rm["price"]) > 0]
            if r_prices:
                starting_price = float(min(r_prices))
        if starting_price == 0.0 and v.get("villa_fee") and float(v.get("villa_fee")) > 0:
            starting_price = float(v.get("villa_fee"))

        discount_percent = (villa_campaign_discounts or {}).get(vid, 0.0)
        if discount_percent > 0 and starting_price > 0:
            discount_price = round(starting_price * (1.0 - (discount_percent / 100.0)), 2)
        else:
            discount_price = 0.0

        category_type = "private-villas"
        card_url = settings.CARD_URL_TEMPLATE.format(
            base_url=settings.MEDIA_BASE_URL.rstrip("/"),
            category_type="private-villas",
            slug=slug,
            id=vid,
        )

        lines = [f"Private Villa: {title}"]
        if title_km:
            lines[0] += f" ({title_km})"

        loc_str = f"Destination / Location: {dest_en}"
        if dest_km:
            loc_str += f" ({dest_km})"
        lines.append(loc_str)

        fees = []
        if v.get("cleaning_fee") and float(v["cleaning_fee"]) > 0:
            fees.append(f"Cleaning fee: ${float(v['cleaning_fee']):.2f}")
        if v.get("villa_fee") and float(v["villa_fee"]) > 0:
            fees.append(f"Villa fee: ${float(v['villa_fee']):.2f}")
        if fees:
            lines.append("Fees: " + ", ".join(fees))

        if starting_price > 0:
            lines.append(f"Starting Price: ${starting_price:.2f} per night.")

        short_desc = clean_html(v.get("short_content") or v.get("intro") or "")
        short_desc_km = clean_html(v.get("short_content_km") or "")
        if short_desc:
            lines.append(f"Overview: {short_desc}")
        if short_desc_km:
            lines.append(f"ទិដ្ឋភាពទូទៅ: {short_desc_km}")

        content = clean_html(v.get("content") or "")
        content_km = clean_html(v.get("content_km") or "")
        if content and content != short_desc:
            lines.append(f"Details: {content}")
        if content_km and content_km != short_desc_km:
            lines.append(f"ព័ត៌មានលម្អិត: {content_km}")

        extra_info = clean_html(v.get("extra_info") or "")
        if extra_info:
            lines.append(f"Amenities & Features: {extra_info}")

        policy = clean_html(v.get("policy") or "")
        if policy:
            lines.append(f"Villa Policy: {policy}")

        if rooms:
            lines.append("Available Room Configurations / Rooms:")
            for rm in rooms:
                rm_line = f"- {rm['name']}"
                if rm["name_km"]:
                    rm_line += f" ({rm['name_km']})"
                parts = []
                if rm.get("max_adults") or rm.get("max_kids"):
                    cap = []
                    if rm.get("max_adults"):
                        cap.append(f"{rm['max_adults']} adults")
                    if rm.get("max_kids"):
                        cap.append(f"{rm['max_kids']} children")
                    parts.append("Capacity: " + ", ".join(cap))
                if rm.get("price") and float(rm["price"]) > 0:
                    parts.append(f"Price: ${float(rm['price']):.2f}")
                if rm.get("description"):
                    parts.append(rm["description"])
                if parts:
                    rm_line += ": " + "; ".join(parts)
                lines.append(rm_line)

        text = "\n".join(lines)
        meta = {
            "source": "fa_private_villa",
            "item_id": int(vid),
            "entity_type": "villa",
            "category_type": category_type,
            "id": int(vid),
            "title": str(title),
            "name": str(title),
            "slug": str(slug),
            "thumbnail": str(thumbnail),
            "price": float(starting_price),
            "discount_price": float(discount_price),
            "url": str(card_url),
            "destination": str(dest_en),
        }
        docs.append((f"villa_{vid}", text, meta))

    print(f"-> fa_private_villa: Processed {len(docs)} active records.")
    return docs


def sync_activities(
    cur: pymysql.cursors.DictCursor,
    existing_tables: set[str],
    resolve_destination: Any,
) -> list[tuple[str, str, dict]]:
    """Synchronize fa_cw_things_to_do records."""
    docs = []
    if "fa_cw_things_to_do" not in existing_tables:
        return docs

    act_cols = get_table_columns(cur, "fa_cw_things_to_do")
    act_where = build_active_where(act_cols)
    cur.execute(f"SELECT * FROM fa_cw_things_to_do{act_where};")
    activities = cur.fetchall()

    for act in activities:
        aid = act["id"]
        title = clean_html(act.get("title") or act.get("name") or f"Activity #{aid}")
        title_km = clean_html(act.get("title_km") or act.get("name_km") or "")
        loc_id = act.get("location_id") or act.get("locations_id")
        dest_en, dest_km = resolve_destination(loc_id)

        lines = [f"Activity / Attraction: {title}"]
        if title_km:
            lines[0] += f" ({title_km})"

        loc_str = f"Location / Destination: {dest_en}"
        if dest_km:
            loc_str += f" ({dest_km})"
        lines.append(loc_str)

        if act.get("is_free") == 1:
            lines.append("Admission / Price: Free")

        short_desc = clean_html(act.get("short_desc") or act.get("short_content") or "")
        short_desc_km = clean_html(act.get("short_content_km") or "")
        if short_desc:
            lines.append(f"Summary: {short_desc}")
        if short_desc_km:
            lines.append(f"សេចក្តីសង្ខេប: {short_desc_km}")

        overview = clean_html(act.get("overview") or "")
        overview_km = clean_html(act.get("overview_km") or "")
        if overview and overview != short_desc:
            lines.append(f"Overview: {overview}")
        if overview_km and overview_km != short_desc_km:
            lines.append(f"ទិដ្ឋភាពទូទៅ: {overview_km}")

        sections_en = parse_sections(act.get("content"))
        for s in sections_en:
            lines.append(s)

        sections_km = parse_sections(act.get("content_km"))
        for s in sections_km:
            lines.append(s)

        for field, label in [
            ("event_info", "Event Information"),
            ("inclusions", "Inclusions"),
            ("requirements", "Requirements"),
            ("event_info_km", "ព័ត៌មានអំពីព្រឹត្តិការណ៍"),
            ("inclusions_km", "ការរួមបញ្ចូល"),
            ("requirements_km", "លក្ខខណ្ឌតម្រូវ"),
        ]:
            val = clean_html(act.get(field) or "")
            if val:
                lines.append(f"{label}: {val}")

        text = "\n".join(lines)
        meta = {
            "source": "fa_cw_things_to_do",
            "item_id": int(aid),
            "entity_type": "activity",
            "id": int(aid),
            "name": str(title),
            "destination": str(dest_en),
        }
        docs.append((f"activity_{aid}", text, meta))

    print(f"-> fa_cw_things_to_do: Processed {len(docs)} active records.")
    return docs


def sync_destinations(
    cur: pymysql.cursors.DictCursor,
    existing_tables: set[str],
) -> list[tuple[str, str, dict]]:
    """Synchronize fa_cw_destinations records."""
    docs = []
    if "fa_cw_destinations" not in existing_tables:
        return docs

    d_cols = get_table_columns(cur, "fa_cw_destinations")
    d_where = build_active_where(d_cols)
    cur.execute(f"SELECT * FROM fa_cw_destinations{d_where};")
    destinations = cur.fetchall()

    for d in destinations:
        did = d["id"]
        title = clean_html(d.get("title") or f"Destination #{did}")
        title_km = clean_html(d.get("title_km") or "")
        subtitle = clean_html(d.get("subtitle") or "")
        subtitle_km = clean_html(d.get("subtitle_km") or "")

        lines = [f"Destination Overview: {title}"]
        if title_km:
            lines[0] += f" ({title_km})"
        if subtitle:
            lines.append(f"Subtitle: {subtitle}")
        if subtitle_km:
            lines.append(f"ចំណងជើងរង: {subtitle_km}")

        overview = clean_html(d.get("overview") or "")
        overview_km = clean_html(d.get("overview_km") or "")
        if overview:
            lines.append(f"Overview: {overview}")
        if overview_km:
            lines.append(f"ទិដ្ឋភាពទូទៅ: {overview_km}")

        sections_en = parse_sections(d.get("content"))
        for s in sections_en:
            lines.append(s)

        sections_km = parse_sections(d.get("content_km"))
        for s in sections_km:
            lines.append(s)

        text = "\n".join(lines)
        meta = {
            "source": "fa_cw_destinations",
            "item_id": int(did),
            "entity_type": "destination",
            "id": int(did),
            "name": str(title),
            "destination": str(title),
        }
        docs.append((f"destination_{did}", text, meta))

    print(f"-> fa_cw_destinations: Processed {len(docs)} active records.")
    return docs


def sync_tips(
    cur: pymysql.cursors.DictCursor,
    existing_tables: set[str],
) -> list[tuple[str, str, dict]]:
    """Synchronize fa_cw_practical_tips records."""
    docs = []
    if "fa_cw_practical_tips" not in existing_tables:
        return docs

    t_cols = get_table_columns(cur, "fa_cw_practical_tips")
    t_where = build_active_where(t_cols)
    cur.execute(f"SELECT * FROM fa_cw_practical_tips{t_where};")
    tips = cur.fetchall()

    for tip in tips:
        tid = tip["id"]
        title = clean_html(tip.get("title") or f"Travel Tip #{tid}")
        title_km = clean_html(tip.get("title_km") or "")

        lines = [f"Cambodia Travel Tip / FAQ: {title}"]
        if title_km:
            lines[0] += f" ({title_km})"

        short_content = clean_html(tip.get("short_content") or "")
        short_content_km = clean_html(tip.get("short_content_km") or "")
        if short_content:
            lines.append(f"Summary: {short_content}")
        if short_content_km:
            lines.append(f"សេចក្តីសង្ខេប: {short_content_km}")

        content = clean_html(tip.get("content") or "")
        content_km = clean_html(tip.get("content_km") or "")
        if content and content != short_content:
            lines.append(f"Advice & Details: {content}")
        if content_km and content_km != short_content_km:
            lines.append(f"ព័ត៌មានលម្អិត និងការណែនាំ: {content_km}")

        text = "\n".join(lines)
        meta = {
            "source": "fa_cw_practical_tips",
            "item_id": int(tid),
            "entity_type": "tip",
            "id": int(tid),
            "name": str(title),
            "destination": "Cambodia",
        }
        docs.append((f"tip_{tid}", text, meta))

    print(f"-> fa_cw_practical_tips: Processed {len(docs)} active records.")
    return docs


def sync_transportation(
    cur: pymysql.cursors.DictCursor,
    existing_tables: set[str],
    resolve_destination: Any,
) -> list[tuple[str, str, dict]]:
    """Synchronize transportation services (airport bus, intercity, car rental, and drivers)."""
    docs = []

    car_types = []
    if "fa_car_type" in existing_tables:
        cur.execute("SELECT * FROM fa_car_type WHERE status = '1';")
        car_types = cur.fetchall()

    fares_by_service: dict[int, list[dict]] = {}
    if "fa_transportation_fare" in existing_tables:
        cur.execute("""
            SELECT f.*, ct.name as car_type_name
            FROM fa_transportation_fare f
            LEFT JOIN fa_car_type ct ON ct.id = f.car_type_id
            WHERE f.status = 1;
        """)
        for f in cur.fetchall():
            fares_by_service.setdefault(f["car_service_id"], []).append(f)

    # 1. Car Services (Bus routes, Intercity transfers, Airport transfers)
    if "fa_car_services" in existing_tables:
        cur.execute("SELECT * FROM fa_car_services WHERE status = '1';")
        for s in cur.fetchall():
            sid = s["id"]
            name = clean_html(s.get("name") or f"Service #{sid}")
            sm = s.get("service_mode") or "transportation"
            mode_display = sm.replace("_", " ").title()

            lines = [f"Transportation Service: {name}"]
            lines.append(f"Service Mode: {mode_display}")

            matched = []
            for ct in car_types:
                cs_ids = [x.strip() for x in str(ct.get("car_service_ids") or "").split(",") if x.strip()]
                if str(sid) in cs_ids:
                    matched.append(ct)

            if matched:
                lines.append("Available Vehicle Types & Routes:")
                for ct in matched:
                    ct_name = clean_html(ct.get("name") or "Vehicle")
                    price = ct.get("price")
                    unit = ct.get("price_unit") or "trip"
                    cap = ct.get("passenger_capacity")
                    desc = clean_html(ct.get("description") or "")

                    details = []
                    if cap:
                        details.append(f"Capacity: {cap} passengers")
                    if price and float(price) > 0:
                        details.append(f"Price: ${float(price):.2f}/{unit}")
                    if desc and desc != "-":
                        details.append(f"Route/Note: {desc}")
                    lines.append(f"- {ct_name}: " + ", ".join(details))

            s_fares = fares_by_service.get(sid, [])
            if s_fares:
                lines.append("Standard Fare Rates:")
                for sf in s_fares:
                    j_type = (sf.get("journey_type") or "trip").replace("_", " ").title()
                    v_name = clean_html(sf.get("car_type_name") or "Vehicle")
                    amt = sf.get("amount")
                    lines.append(f"- {v_name} ({j_type}): ${float(amt):.2f}")

            text = "\n".join(lines)
            meta = {
                "source": "fa_car_services",
                "item_id": int(sid),
                "entity_type": "transportation_service",
                "id": int(sid),
                "name": str(name),
                "destination": "Cambodia",
            }
            docs.append((f"car_service_{sid}", text, meta))

    # 2. Individual Cars for rental / ride
    if "fa_cars" in existing_tables:
        cols = get_table_columns(cur, "fa_cars")
        where = build_active_where(cols)
        cur.execute(f"SELECT * FROM fa_cars{where};")
        for c in cur.fetchall():
            cid = c["id"]
            make = clean_html(c.get("make") or "")
            model = clean_html(c.get("model") or "")
            year = c.get("year") or ""
            v_name = f"{make} {model}".strip() or f"Vehicle #{cid}"

            lines = [f"Rental / Transport Vehicle: {v_name} ({year})"]
            if c.get("color"):
                lines.append(f"Color: {clean_html(c['color'])}")
            if c.get("price") and float(c["price"]) > 0:
                lines.append(f"Daily Rate: ${float(c['price']):.2f}")
            specs = []
            if c.get("transmission"):
                specs.append(f"Transmission: {c['transmission']}")
            if c.get("fuel_type"):
                specs.append(f"Fuel: {c['fuel_type']}")
            if specs:
                lines.append("Vehicle Specs: " + ", ".join(specs))
            if c.get("passengers"):
                lines.append(f"Passenger Capacity: {c['passengers']} passengers")
            if c.get("available_for"):
                avail = ", ".join(x.replace("_", " ").title() for x in str(c["available_for"]).split(","))
                lines.append(f"Available For: {avail}")
            if c.get("description"):
                lines.append(f"Description: {clean_html(c['description'])}")

            text = "\n".join(lines)
            meta = {
                "source": "fa_cars",
                "item_id": int(cid),
                "entity_type": "car",
                "id": int(cid),
                "name": str(v_name),
                "destination": "Cambodia",
            }
            docs.append((f"car_{cid}", text, meta))

    # 3. Transportation Drivers & Operators
    if "fa_transportation" in existing_tables:
        cols = get_table_columns(cur, "fa_transportation")
        where = build_active_where(cols)
        cur.execute(f"SELECT * FROM fa_transportation{where};")
        for t in cur.fetchall():
            tid = t["id"]
            title = clean_html(t.get("title") or "")
            if not title or title == "-":
                continue
            loc_id = t.get("location_id")
            dest_en, dest_km = resolve_destination(loc_id)

            lines = [f"Transportation Operator / Vehicle: {title}"]
            lines.append(f"Location: {dest_en}")
            if t.get("experience"):
                lines.append(f"Driver Experience: {t['experience']} years")
            content = clean_html(t.get("content") or "")
            if content and content != "-":
                lines.append(f"Details: {content}")

            text = "\n".join(lines)
            meta = {
                "source": "fa_transportation",
                "item_id": int(tid),
                "entity_type": "transportation",
                "id": int(tid),
                "name": str(title),
                "destination": str(dest_en),
            }
            docs.append((f"transport_{tid}", text, meta))

    print(f"-> Transportation: Processed {len(docs)} active records across services, vehicles, and fleet.")
    return docs


# =====================================================================
# Main Ingestion & Synchronization Flow
# =====================================================================
def sync_data():
    print("=" * 60)
    print(f"Starting {settings.APP_NAME} MySQL -> Qdrant ingestion")
    print("=" * 60)

    conn = get_db_connection()
    cur = conn.cursor()

    # Discover tables present in database
    cur.execute("SHOW TABLES;")
    existing_tables = {list(r.values())[0] for r in cur.fetchall()}
    print(f"Discovered {len(existing_tables)} tables in database.")

    # -----------------------------------------------------------------
    # 1. Cache Locations and Destinations for Foreign-Key Lookups
    # -----------------------------------------------------------------
    locations_map: dict[int, dict[str, str]] = {}
    if "fa_locations" in existing_tables:
        cur.execute("SELECT * FROM fa_locations;")
        for r in cur.fetchall():
            locations_map[r["id"]] = {
                "title": r.get("title") or "",
                "title_km": r.get("title_km") or "",
            }

    destinations_map: dict[int, dict[str, str]] = {}
    if "fa_cw_destinations" in existing_tables:
        cur.execute("SELECT * FROM fa_cw_destinations;")
        for r in cur.fetchall():
            loc_id = r.get("location_id")
            if loc_id:
                destinations_map[loc_id] = {
                    "title": r.get("title") or "",
                    "title_km": r.get("title_km") or "",
                }

    def resolve_destination(location_id: int | None) -> tuple[str, str]:
        if not location_id:
            return "Cambodia", "ប្រទេសកម្ពុជា"
        info = destinations_map.get(location_id) or locations_map.get(location_id)
        if info:
            en = info.get("title") or "Cambodia"
            km = info.get("title_km") or ""
            return en, km
        return "Cambodia", ""

    # -----------------------------------------------------------------
    # 2. Cache Categories for Tag and Cuisine Resolution
    # -----------------------------------------------------------------
    categories_map: dict[str, str] = {}
    if "fa_categories" in existing_tables:
        cur.execute("SELECT id, title FROM fa_categories;")
        for r in cur.fetchall():
            categories_map[str(r["id"])] = clean_html(r["title"])

    # -----------------------------------------------------------------
    # 3. Cache Room Types & Starting Prices for Hotels and Villas
    # -----------------------------------------------------------------
    hotel_rooms_map: dict[int, list[dict]] = {}
    hotel_room_prices: dict[int, float] = {}

    if "fa_room_allotment" in existing_tables:
        cur.execute(
            "SELECT room_type_id, MIN(price) as min_p FROM fa_room_allotment "
            "WHERE property_category = 'hotel' AND price > 0 GROUP BY room_type_id;"
        )
        for r in cur.fetchall():
            hotel_room_prices[r["room_type_id"]] = float(r["min_p"])

    if "fa_hotel_room_type" in existing_tables:
        cur.execute("SELECT * FROM fa_hotel_room_type WHERE deletetime IS NULL OR deletetime = 0;")
        for r in cur.fetchall():
            hid = r.get("hotel_id")
            if not hid:
                continue
            p = hotel_room_prices.get(r.get("id"))
            hotel_rooms_map.setdefault(hid, []).append({
                "name": r.get("name") or "",
                "name_km": r.get("name_km") or "",
                "price": p,
                "max_adults": r.get("max_adults"),
                "max_kids": r.get("max_kids"),
                "description": clean_html(r.get("description") or ""),
            })

    villa_rooms_map: dict[int, list[dict]] = {}
    villa_room_prices: dict[int, float] = {}

    if "fa_room_allotment" in existing_tables:
        cur.execute(
            "SELECT room_type_id, MIN(price) as min_p FROM fa_room_allotment "
            "WHERE property_category = 'private_villa' AND price > 0 GROUP BY room_type_id;"
        )
        for r in cur.fetchall():
            villa_room_prices[r["room_type_id"]] = float(r["min_p"])

    if "fa_room_type" in existing_tables:
        v_cols = get_table_columns(cur, "fa_room_type")
        v_where = build_active_where(v_cols)
        cur.execute(f"SELECT * FROM fa_room_type{v_where};")
        for r in cur.fetchall():
            vid = r.get("private_villa_id")
            if not vid:
                continue
            row_price = r.get("price")
            p = villa_room_prices.get(r.get("id"))
            if not p and row_price and float(row_price) > 0:
                p = float(row_price)
            villa_rooms_map.setdefault(vid, []).append({
                "name": r.get("name") or "",
                "name_km": r.get("name_km") or "",
                "price": p,
                "max_adults": r.get("max_adults"),
                "max_kids": r.get("max_kids"),
                "description": clean_html(r.get("description") or ""),
            })

    # -----------------------------------------------------------------
    # 3b. Cache Campaign Discounts for Hotels and Villas
    # -----------------------------------------------------------------
    hotel_campaign_discounts: dict[int, float] = {}
    villa_campaign_discounts: dict[int, float] = {}

    if "fa_cw_campaign_item" in existing_tables:
        cur.execute("""
            SELECT ci.item_id, ci.service_type, ci.discount, rd.discount as room_discount
            FROM fa_cw_campaign_item ci
            LEFT JOIN fa_cw_campaign_room_discount rd ON rd.campaign_item_id = ci.id
            WHERE ci.service_type IN ('wtd_hotel', 'wtd_private_villa');
        """)
        for item in cur.fetchall():
            st = item.get("service_type")
            iid = item.get("item_id")
            disc = item.get("discount") or item.get("room_discount")
            if iid and disc and float(disc) > 0:
                if st == "wtd_hotel":
                    hotel_campaign_discounts[iid] = max(hotel_campaign_discounts.get(iid, 0.0), float(disc))
                elif st == "wtd_private_villa":
                    villa_campaign_discounts[iid] = max(villa_campaign_discounts.get(iid, 0.0), float(disc))

    # -----------------------------------------------------------------
    # 4. Extract Natural Language Chunks Using Dedicated Sync Functions
    # -----------------------------------------------------------------
    documents_to_upsert: list[tuple[str, str, dict]] = []

    # 1. fa_cw_happening_now (Events & Festivals) -> doc ID: happening_{id}
    documents_to_upsert.extend(sync_happening_now(cur, existing_tables, resolve_destination))

    # 2. fa_cw_campaign (Promotions & Special Offers) -> doc ID: campaign_{id}
    documents_to_upsert.extend(sync_campaigns(cur, existing_tables))

    # 3. fa_spa (Spas & Wellness) -> doc ID: spa_{id}
    documents_to_upsert.extend(sync_spas(cur, existing_tables, resolve_destination, categories_map))

    # 4. fa_eat_drink (Dining & Restaurants) -> doc ID: dining_{id}
    documents_to_upsert.extend(sync_dining(cur, existing_tables, resolve_destination, categories_map))

    # 5. fa_tourpackages (Tour Packages) -> doc ID: tour_{id}
    documents_to_upsert.extend(sync_tourpackages(cur, existing_tables, resolve_destination, categories_map))

    # 6. fa_guide / fa_guides (Tour Guides) -> doc ID: guide_{id}
    documents_to_upsert.extend(sync_guides(cur, existing_tables, resolve_destination))

    # Core base tables: Hotels, Villas, Activities, Destinations, Tips, Transportation
    documents_to_upsert.extend(sync_hotels(cur, existing_tables, resolve_destination, hotel_rooms_map, hotel_campaign_discounts))
    documents_to_upsert.extend(sync_villas(cur, existing_tables, resolve_destination, villa_rooms_map, villa_campaign_discounts))
    documents_to_upsert.extend(sync_activities(cur, existing_tables, resolve_destination))
    documents_to_upsert.extend(sync_destinations(cur, existing_tables))
    documents_to_upsert.extend(sync_tips(cur, existing_tables))
    documents_to_upsert.extend(sync_transportation(cur, existing_tables, resolve_destination))

    conn.close()

    unexpected_sources = {
        metadata.get("source")
        for _, _, metadata in documents_to_upsert
        if metadata.get("source") not in ALLOWED_DOCUMENT_SOURCES
    }
    if unexpected_sources:
        raise ValueError(
            "Refusing to embed records from unapproved MySQL sources: "
            f"{sorted(str(source) for source in unexpected_sources)}"
        )

    # -----------------------------------------------------------------
    # 5. Remove Initial Sample Mock Records If Any
    # -----------------------------------------------------------------
    try:
        vector_service.delete_documents(ids=["hotel_1", "activity_1", "food_1"])
        print("[Qdrant] Removed initial placeholder mock records.")
    except Exception as exc:
        print(f"[Qdrant] Could not remove placeholder mock records: {exc}")

    # -----------------------------------------------------------------
    # 6. Batch Vectorization and Upsert into Qdrant
    # -----------------------------------------------------------------
    source_doc_count = len(documents_to_upsert)
    print(
        f"\nTotal entities ready for embedding and Qdrant ingestion: "
        f"{source_doc_count}"
    )

    if source_doc_count == 0:
        print("No documents found to ingest.")
        return

    try:
        vector_service.ensure_collection()
    except ValueError as exc:
        print(f"[Qdrant] Existing collection is incompatible: {exc}")
        confirmation = input(
            "Switching embedding models requires deleting the current Qdrant "
            "collection and reindexing. Type RECREATE to permanently delete "
            "its contents and continue: "
        )
        if confirmation != "RECREATE":
            raise RuntimeError(
                "Qdrant collection was not changed. Confirm with RECREATE to "
                "replace its incompatible vectors."
            ) from exc
        vector_service.client.delete_collection(vector_service.collection_name)
        vector_service.ensure_collection()
        print("[Qdrant] Recreated the collection for 384-dimensional vectors.")

    existing_payloads = _existing_document_payloads()
    pending_documents = []
    for item in documents_to_upsert:
        document_id, text, metadata = item
        existing_payload = existing_payloads.get(
            vector_service.point_id(document_id)
        )
        existing_metadata = (
            {key: value for key, value in existing_payload.items() if key != "text"}
            if existing_payload is not None
            else None
        )
        if (
            existing_payload is None
            or existing_payload.get("text") != text
            or existing_metadata != metadata
        ):
            pending_documents.append(item)
    documents_to_upsert = pending_documents
    skipped_docs = source_doc_count - len(documents_to_upsert)
    print(f"Skipping {skipped_docs} documents already current in Qdrant.")

    total_docs = len(documents_to_upsert)
    if total_docs == 0:
        print("All source documents are already up to date in Qdrant.")
        return

    num_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
    print(
        f"Processing {total_docs} documents in {num_batches} batches of up to "
        f"{BATCH_SIZE} documents..."
    )

    for b_idx in range(num_batches):
        start_i = b_idx * BATCH_SIZE
        end_i = min(start_i + BATCH_SIZE, total_docs)
        batch = documents_to_upsert[start_i:end_i]

        b_ids = [item[0] for item in batch]
        b_texts = [item[1] for item in batch]
        b_metas = [item[2] for item in batch]

        print(f"  [Batch {b_idx + 1}/{num_batches}] Vectorizing {len(b_texts)} chunks (items {start_i + 1} - {end_i})...")
        embeddings = embedder.embed_documents(b_texts)

        points = [
            PointStruct(
                id=vector_service.point_id(document_id),
                vector=embedding,
                payload={**metadata, "text": text},
            )
            for document_id, text, metadata, embedding in zip(
                b_ids, b_texts, b_metas, embeddings
            )
        ]
        for upsert_start in range(0, len(points), UPSERT_BATCH_SIZE):
            upsert_points = points[upsert_start:upsert_start + UPSERT_BATCH_SIZE]
            print(
                f"    Uploading {len(upsert_points)} points to Qdrant "
                f"({upsert_start + 1} - {upsert_start + len(upsert_points)} of {len(points)})..."
            )
            vector_service.client.upsert(
                collection_name=vector_service.collection_name,
                points=upsert_points,
                timeout=120,
            )

    final_count = vector_service.client.count(
        collection_name=vector_service.collection_name,
        exact=True,
    ).count
    print("=" * 60)
    print("Ingestion Complete!")
    print(f"Final document count in Qdrant collection '{settings.QDRANT_COLLECTION}': {final_count}")
    print("=" * 60)


if __name__ == "__main__":
    sync_data()
