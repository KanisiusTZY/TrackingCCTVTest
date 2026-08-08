import cv2
import numpy as np

# Color Palette (BGR)
COLOR_BEKERJA = (0, 235, 100)       # Neon Green
COLOR_AWAY = (0, 50, 255)          # Crimson Red
COLOR_PERSON_BOX = (220, 220, 220) # Light Gray/Cyan
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_HUD_BG = (15, 18, 25)

def draw_corner_box(img, bbox, color, thickness=2, corner_len=15):
    x1, y1, x2, y2 = bbox
    # Main rectangle frame
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)

    # Accent corners
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)


def draw_filled_badge(img, text, position, bg_color, text_color=COLOR_TEXT_WHITE, font_scale=0.55, pad=6):
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, 2)

    rect_x1 = x
    rect_y1 = max(0, y - text_h - (pad * 2))
    rect_x2 = x + text_w + (pad * 2)
    rect_y2 = y

    # Semi-transparent badge background
    overlay = img.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)

    # Border
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, 2)

    # Text
    cv2.putText(
        img,
        text,
        (x + pad, y - pad),
        font,
        font_scale,
        text_color,
        2,
        cv2.LINE_AA
    )
    return rect_y1


class Visualizer:
    def __init__(self):
        pass

    def render(self, frame, tracked_persons, chair_zones, rule_engine, fps=0.0):
        out = frame.copy()
        h, w = out.shape[:2]

        bekerja_count = 0
        away_count = 0

        # 1. Render Chair Zones (Single status label per zone)
        for chair in chair_zones:
            cx1, cy1, cx2, cy2 = chair["bbox"]
            cname = chair.get("name", chair["id"])
            zone_status = chair.get("zone_status", "BEKERJA")
            zone_label = chair.get("zone_label", f"{cname}: BEKERJA")
            zone_color = chair.get("zone_color", COLOR_BEKERJA)

            if zone_status == "BEKERJA":
                bekerja_count += 1
                # Green bounding box for occupied zone
                cv2.rectangle(out, (cx1, cy1), (cx2, cy2), COLOR_BEKERJA, 2)
                draw_filled_badge(out, zone_label, (cx1, cy1), COLOR_BEKERJA, font_scale=0.55)
            else:
                away_count += 1
                # Red glowing box for empty / away zone
                overlay = out.copy()
                cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), COLOR_AWAY, -1)
                cv2.addWeighted(overlay, 0.22, out, 0.78, 0, out)
                cv2.rectangle(out, (cx1, cy1), (cx2, cy2), COLOR_AWAY, 2)
                draw_filled_badge(out, zone_label, (cx1, cy1), COLOR_AWAY, font_scale=0.55)

        # 2. Render Tracked Persons (Clean person bounding boxes)
        for p_id, p in tracked_persons.items():
            px1, py1, px2, py2 = p["bbox"]
            cv2.rectangle(out, (px1, py1), (px2, py2), COLOR_PERSON_BOX, 1, cv2.LINE_AA)
            cv2.putText(out, f"ID:{p_id}", (px1 + 4, py1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_PERSON_BOX, 1)

        # 3. Top Cyberpunk HUD Bar
        hud_h = 45
        hud_overlay = out.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), COLOR_HUD_BG, -1)
        cv2.addWeighted(hud_overlay, 0.88, out, 0.12, 0, out)
        cv2.line(out, (0, hud_h), (w, hud_h), (0, 235, 100), 1)

        # HUD Text stats
        sys_title = "SEAT MONITORING SYSTEM"
        fps_text = f"FPS: {fps:.1f}"
        persons_cnt = f"PERSONS: {len(tracked_persons)}"
        status_stats = f"BEKERJA: {bekerja_count}  |  TIDAK DI TEMPAT: {away_count}"

        cv2.putText(out, sys_title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 235, 100), 2)
        cv2.putText(out, fps_text, (290, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(out, persons_cnt, (390, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(out, status_stats, (540, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 2)

        # 4. Bottom Controls Legend Bar
        legend_str = "STATUS: [GREEN] BEKERJA | [RED] TIDAK DI TEMPAT | [R] Reset Timers | [Q] Quit"
        cv2.putText(out, legend_str, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return out
