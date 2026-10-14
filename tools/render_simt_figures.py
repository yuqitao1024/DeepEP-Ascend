"""Historical mixed-rendering experiment; current article figures use imagegen.

Outputs go to the scratch archive so this script cannot recreate retired article
assets in the published documentation directory.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


BG = "#fbf1c7"
CARD = "#ffffff"
BORDER = "#d5c4a1"
TEXT = "#3c3836"
MUTED = "#665c54"
SIMT = "#458588"
SCALAR = "#b16286"
URMA = "#d65d0e"
MTE = "#98971a"
TOKEN = "#d79921"
RANK = "#8f3f71"
RED = "#cc241d"


def font(size, bold=False):
    path = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
        if bold
        else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    return ImageFont.truetype(path, size, index=2)


def rounded(draw, box, radius=16, fill=CARD, outline=BORDER, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def center_text(draw, box, text, f, fill=TEXT):
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=f)
    x = (x1 + x2 - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (y1 + y2 - (bbox[3] - bbox[1])) / 2 - bbox[1]
    draw.text((x, y), text, font=f, fill=fill)


def soft_base(base):
    """Keep imagegen's Gruvbox texture while reducing visual noise."""
    im = base.filter(ImageFilter.GaussianBlur(0.7))
    im = ImageEnhance.Contrast(im).enhance(0.82)
    im = ImageEnhance.Brightness(im).enhance(1.06)
    return im


def style_card(draw, box, radius=20, accent=None):
    draw.rounded_rectangle(box, radius=radius, fill="#ffffffee", outline=BORDER, width=2)
    if accent:
        x1, y1, _, _ = box
        x2, _, _, y2 = box
        draw.rounded_rectangle((x1, y1, x1 + 6, y2), radius=3, fill=accent)


def render_dispatch(path):
    W, H = 1680, 945
    base_path = ".scratch/imagegen/deepep-ascend-simt-figure-base-v10.png"
    base = Image.open(base_path).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    im = soft_base(base)
    d = ImageDraw.Draw(im)
    f_title = font(31)
    f_h = font(20)
    f_small = font(16)
    f_tiny = font(14)

    d.text((64, 44), "Dispatch：SIMT 并行整理路由与通信描述", font=f_title, fill=TEXT)

    # Main matrix card.
    style_card(d, (50, 98, 1288, 822), 26, SIMT)
    x0, y0, x1, y1 = 74, 124, 1288, 800
    headers = ["Router 结果", "目标统计", "重复 rank 去重", "接收 slot 计算", "通信描述构造"]
    colw = (x1 - x0) / len(headers)
    for i, h in enumerate(headers):
        box = (x0 + i * colw, y0, x0 + (i + 1) * colw, y0 + 44)
        if i in (1, 2, 3, 4):
            d.rounded_rectangle(box, radius=10, fill=SIMT, outline=SIMT)
        else:
            d.rounded_rectangle(box, radius=10, fill="#ebdbb2", outline=BORDER)
        center_text(d, box, h, f_h, TEXT if i == 0 else "#ffffff")
        if i:
            d.line((box[0], box[1] + 8, box[0], box[3] - 8), fill=BORDER, width=2)

    tokens = [
        ("token 0", "E7 / E10", "rank 1、2", "无重复", "slot A", "描述 0"),
        ("token 1", "E3 / E12", "rank 0、3", "无重复", "slot B", "描述 1"),
        ("token 2", "E5 / E8", "rank 1、1", "去重后一份", "slot C", "描述 2"),
        ("token 3", "E15 / E2", "rank 2、3", "无重复", "slot D", "描述 3"),
        ("token 4", "E1 / E6", "rank 0、1", "无重复", "slot E", "描述 4"),
        ("token 5", "E9 / E11", "rank 2、2", "去重后一份", "slot F", "描述 5"),
        ("token 6", "E4 / E14", "rank 0、3", "无重复", "slot G", "描述 6"),
        ("token 7", "E13 / E16", "rank 1、3", "无重复", "slot H", "描述 7"),
    ]
    row_h = (y1 - y0 - 44) / len(tokens)
    for r, row in enumerate(tokens):
        ry = y0 + 44 + r * row_h
        if r % 2 == 0:
            d.rounded_rectangle((x0, ry + 2, x1, ry + row_h - 4), radius=12, fill="#f9f5d7", outline="#eee0b7")
        d.text((x0 + 10, ry + row_h / 2 - 9), row[0], font=f_small, fill=TEXT)
        for c in range(1, len(row)):
            cx1 = x0 + c * colw + 8
            cx2 = x0 + (c + 1) * colw - 8
            cy1 = ry + 9
            cy2 = ry + row_h - 9
            if c == 3 and r in (2, 5):
                rounded(d, (cx1, cy1, cx2, cy2), 12, "#f4e8c8", URMA, 2)
                center_text(d, (cx1, cy1, cx2, cy1 + 24), "去重", f_tiny, URMA)
                center_text(d, (cx1, cy1 + 20, cx2, cy2), row[c], f_tiny, TEXT)
            elif c == 5:
                rounded(d, (cx1, cy1, cx2, cy2), 12, "#dce7e7", SIMT, 2)
                center_text(d, (cx1, cy1, cx2, cy2), row[c], f_tiny, TEXT)
            else:
                rounded(d, (cx1, cy1, cx2, cy2), 12)
                center_text(d, (cx1, cy1, cx2, cy2), row[c], f_tiny, TEXT)
        if r < len(tokens) - 1:
            d.line((x0 + 10, ry + row_h, x1 - 10, ry + row_h), fill="#eee0b7", width=1)

    # AIV group labels.
    for i in range(4):
        gy = y0 + 44 + i * 2 * row_h + 4
        gh = 2 * row_h - 8
        rounded(d, (18, gy, 44, gy + gh), 12, SIMT, SIMT, 2)
        label = f"AIV {i}"
        bbox = d.textbbox((0, 0), label, font=f_tiny)
        rotated = Image.new("RGBA", (bbox[2] + 4, bbox[3] + 4), (0, 0, 0, 0))
        rd = ImageDraw.Draw(rotated)
        rd.text((2, 2), label, font=f_tiny, fill="#ffffff")
        rotated = rotated.rotate(90, expand=True)
        px = 16
        py = int(gy + gh / 2 - rotated.height / 2)
        im.paste(rotated, (px, py, px + rotated.width, py + rotated.height), rotated)

    # Right handoff cards.
    cards = [
        ("scalar", "提交描述", SCALAR),
        ("URMA", "远端写入", URMA),
        ("MTE", "本地搬运", MTE),
    ]
    for i, (name, desc, color) in enumerate(cards):
        cy = 150 + i * 190
        style_card(d, (1340, cy, 1625, cy + 145), 22, color)
        d.rounded_rectangle((1360, cy + 24, 1605, cy + 60), radius=12, fill=color, outline=color)
        center_text(d, (1360, cy + 24, 1605, cy + 60), name, f_h, "#ffffff")
        center_text(d, (1360, cy + 74, 1605, cy + 122), desc, f_small, TEXT)
        d.line((1310, cy + 72, 1340, cy + 72), fill=BORDER, width=3)

    # Legend and caption.
    rounded(d, (50, 846, 1625, 902), 20, "#f9f5d7")
    center_text(d, (50, 846, 1625, 902), "每个 token 独立处理；同一 token 的重复 rank 只发送一份，接收端再按 expert 展开", f_small, TEXT)
    d.text((64, 918), "SIMT：并行整理与构造", font=f_small, fill=SIMT)
    d.text((250, 918), "scalar：统一提交边界", font=f_small, fill=SCALAR)
    d.text((460, 918), "URMA：远端传输", font=f_small, fill=URMA)
    d.text((630, 918), "MTE：本地传输", font=f_small, fill=MTE)

    im.save(path, optimize=True)


def arrow(draw, start, end, color=BORDER, width=3):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 12
    p1 = (x2 - size * math.cos(ang - 0.4), y2 - size * math.sin(ang - 0.4))
    p2 = (x2 - size * math.cos(ang + 0.4), y2 - size * math.sin(ang + 0.4))
    draw.polygon([end, p1, p2], fill=color)


def render_combine(path):
    W, H = 1680, 945
    base_path = ".scratch/imagegen/deepep-ascend-simt-figure-base-v10.png"
    base = Image.open(base_path).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    im = soft_base(base)
    d = ImageDraw.Draw(im)
    f_title = font(31)
    f_h = font(21)
    f = font(18)
    f_small = font(16)
    f_tiny = font(14)

    d.text((64, 44), "Combine：SIMT 并行整理返回结果并流水执行", font=f_title, fill=TEXT)

    # Left task cards.
    d.text((60, 112), "返回结果分组", font=f_h, fill=MUTED)
    tasks = [
        ("task A", "来自 rank 1", RANK),
        ("task B", "来自 rank 2", RANK),
        ("task C", "来自 rank 3", RANK),
    ]
    for i, (name, source, color) in enumerate(tasks):
        y = 162 + i * 98
        style_card(d, (55, y, 285, y + 78), 22, color)
        center_text(d, (78, y + 14, 262, y + 38), name, f_h, color)
        center_text(d, (78, y + 42, 262, y + 66), source, f_small, TEXT)

    style_card(d, (55, 525, 285, 645), 22, SIMT)
    center_text(d, (78, 547, 262, 571), "task D", f_h, SIMT)
    center_text(d, (78, 597, 262, 631), "本地输出", f, TEXT)

    # Upper track.
    style_card(d, (350, 115, 1610, 410), 26, SIMT)
    d.text((385, 142), "通信准备（SIMT）", font=f_h, fill=SIMT)
    upper = [("读取返回信息", "task A"), ("组织 task", "task A"), ("构造远端写描述", "task B"), ("scalar 提交", "task B")]
    for i, (name, task) in enumerate(upper):
        x = 410 + i * 310
        rounded(d, (x, 195, x + 240, 285), 20, "#e5eeee", SIMT, 2)
        center_text(d, (x, 195, x + 240, 240), name, f, TEXT)
        center_text(d, (x, 240, x + 240, 275), task, f_small, MUTED)
        if i:
            arrow(d, (x - 62, 240), (x - 8, 240), SIMT, 3)

    # Lower track.
    style_card(d, (350, 470, 1610, 765), 26, SIMT)
    d.text((385, 497), "本地聚合（SIMT）", font=f_h, fill=SIMT)
    lower = [("本地 reduce", "task A"), ("写回 token 输出位置", "task A"), ("本地 reduce", "task B")]
    for i, (name, task) in enumerate(lower):
        x = 410 + i * 310
        rounded(d, (x, 550, x + 240, 640), 20, "#e5eeee", SIMT, 2)
        center_text(d, (x, 550, x + 240, 595), name, f, TEXT)
        center_text(d, (x, 595, x + 240, 630), task, f_small, MUTED)
        if i:
            arrow(d, (x - 62, 595), (x - 8, 595), SIMT, 3)

    # Overlap callout between tracks.
    arrow(d, (700, 550), (700, 410), URMA, 4)
    rounded(d, (750, 428, 1210, 482), 22, "#f4e8c8", URMA, 2)
    center_text(d, (750, 428, 1210, 482), "当前 task 聚合时，下一个 task 已开始准备通信", f, TEXT)

    # task D local-only path.
    arrow(d, (285, 585), (350, 585), SIMT, 4)
    rounded(d, (700, 770, 1090, 830), 20, "#e5eeee", SIMT, 2)
    center_text(d, (700, 770, 1090, 830), "本地输出不需要远端写，直接进入本地聚合", f, TEXT)

    # Connect task cards to tracks.
    for i in range(3):
        y = 200 + i * 98
        arrow(d, (285, y), (350, 240), RANK, 3)

    rounded(d, (350, 846, 1610, 902), 20, "#f9f5d7")
    center_text(d, (350, 846, 1610, 902), "通信准备与本地聚合并行推进；远端结果提前提交，本地结果直接聚合", f_small, TEXT)
    d.text((64, 918), "SIMT：信息整理与描述构造", font=f_small, fill=SIMT)
    d.text((360, 918), "scalar：提交边界", font=f_small, fill=SCALAR)
    d.text((530, 918), "task：按来源rank分组", font=f_small, fill=RANK)

    im.save(path, optimize=True)


if __name__ == "__main__":
    archive_dir = Path(".scratch/imagegen/legacy-renderer")
    archive_dir.mkdir(parents=True, exist_ok=True)
    render_dispatch(str(archive_dir / "deepep-ascend-simt-dispatch-dataflow-v6.png"))
    render_combine(str(archive_dir / "deepep-ascend-simt-combine-dataflow-v6.png"))
