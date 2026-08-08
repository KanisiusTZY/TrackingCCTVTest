import cv2
import numpy as np

# Color Palette (BGR)
COLOR_ON_DUTY = (0, 235, 100)      # Neon Green
COLOR_SKIVING = (0, 50, 255)       # Crimson Red
COLOR_EMPTY_CHAIR = (0, 50, 255)   # Crimson Red
COLOR_CHAIR_NORMAL = (0, 200, 255) # Yellow/Cyan
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_TEXT_BLACK = (10, 10, 10)
COLOR_HUD_BG = (15, 18, 25)

def draw_corner_box(img, bbox, color, thickness=2, corner_len=15):
    x1, y1, x2, y2 = bbox
    # Main rectangle frame
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)

    # Accent corners
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)


def draw_filled_badge(img, text, position, bg_color, text_color=COLOR_TEXT_WHITE, font_scale=0.5, pad=6):
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, 1)

    rect_x1 = x
    rect_y1 = max(0, y - text_h - (pad * 2))
    rect_x2 = x + text_w + (pad * 2)
    rect_y2 = y

    # Semi-transparent badge background overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    # Border
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, 1)

    # Text
    cv2.putText(
        img,
        text,
        (x + pad, y - pad),
        font,
        font_scale,
        text_color,
        1,
        cv2.LINE_AA
    )
    return rect_y1


class Visualizer:
    def __init__(self):
        pass

    def render(self, frame, tracked_persons, chair_zones, rule_engine, fps=0.0):
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Render Chair Zones (Rule 3)
        for chair in chair_zones:
            cx1, cy1, cx2, cy2 = chair["bbox"]
            is_empty = chair.get("is_empty", False)
            empty_label = chair.get("empty_label", "")

            if is_empty and rule_engine.is_rule_enabled("rule3_empty_seat"):
                # Glowing red outline for empty chair
                overlay = out.copy()
                cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), COLOR_EMPTY_CHAIR, -1)
                cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)
                cv2.rectangle(out, (cx1, cy1), (cx2, cy2), COLOR_EMPTY_CHAIR, 2)
                draw_filled_badge(out, f"CHAIR EMPTY | {empty_label}", (cx1, cy1 + 25), COLOR_EMPTY_CHAIR, font_scale=0.45)
            else:
                # Normal chair outline
                cv2.rectangle(out, (cx1, cy1), (cx2, cy2), COLOR_CHAIR_NORMAL, 1, lineType=cv2.LINE_AA)
                cv2.putText(out, chair.get("name", "Desk"), (cx1 + 5, cy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CHAIR_NORMAL, 1)

        # 2. Render Tracked Persons (Rules 1, 2, 4)
        for p_id, p in tracked_persons.items():
            px1, py1, px2, py2 = p["bbox"]
            gender = p.get("gender", "Unknown")
            gender_tag = "[M]" if gender == "Male" else ("[F]" if gender == "Female" else "[?]")
            status = p.get("status", "ON DUTY | 在岗")
            status_color = p.get("status_color", COLOR_ON_DUTY)

            # Draw person corner bounding box
            draw_corner_box(out, (px1, py1, px2, py2), status_color, thickness=2)

            # Primary Status Badge above head
            badge_text = f"ID:{p_id} {gender_tag} | {status}"
            badge_y1 = draw_filled_badge(out, badge_text, (px1, py1), status_color, font_scale=0.48)

            # Rule 4: Opposite Gender Interaction Badge (drawn above primary badge)
            if p.get("interaction_active", False) and rule_engine.is_rule_enabled("rule4_opposite_gender"):
                inter_label = p.get("interaction_label", "")
                partner_id = p.get("interaction_partner_id", "")
                inter_badge_text = f"PAIR({p_id}&{partner_id}) | {inter_label}"
                draw_filled_badge(out, inter_badge_text, (px1, badge_y1 - 2), COLOR_SKIVING, font_scale=0.45)

        # 3. Top Skynet Cyberpunk HUD Bar
        hud_h = 45
        hud_overlay = out.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), COLOR_HUD_BG, -1)
        cv2.addWeighted(hud_overlay, 0.88, out, 0.12, 0, out)
        cv2.line(out, (0, hud_h), (w, hud_h), (0, 255, 200), 1)

        # HUD Text items
        sys_title = "SKYNET CCTV MONITORING SYSTEM"
        fps_text = f"FPS: {fps:.1f}"
        persons_cnt = f"PERSONS: {len(tracked_persons)}"

        # Rule Toggles indicators
        r1_st = "ON" if rule_engine.is_rule_enabled("rule1_on_duty") else "OFF"
        r2_st = "ON" if rule_engine.is_rule_enabled("rule2_skiving") else "OFF"
        r3_st = "ON" if rule_engine.is_rule_enabled("rule3_empty_seat") else "OFF"
        r4_st = "ON" if rule_engine.is_rule_enabled("rule4_opposite_gender") else "OFF"

        rules_str = f"[1]Duty:{r1_st}  [2]Skiving:{r2_st}  [3]EmptySeat:{r3_st}  [4]GenderChat:{r4_st}"

        cv2.putText(out, sys_title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
        cv2.putText(out, fps_text, (330, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(out, persons_cnt, (430, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(out, rules_str, (570, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 250), 1)

        # 4. Bottom Controls Legend Bar
        legend_str = "HOTKEYS: [1-4] Toggle Rules | [R] Reset Timers | [Q] Quit"
        cv2.putText(out, legend_str, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return out
