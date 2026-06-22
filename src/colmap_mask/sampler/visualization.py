from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .io import atomic_text_writer
from .models import SelectionRecord, TrajectoryRecord


_HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SLAM軌跡と抽出点</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; overflow: hidden; background: #0d1117; color: #e6edf3; }
#toolbar { position: fixed; z-index: 2; top: 12px; left: 12px; display: flex; gap: 10px;
  align-items: center; padding: 9px 12px; border: 1px solid #30363d; border-radius: 7px;
  background: rgba(22,27,34,.92); }
select, button { color: inherit; background: #21262d; border: 1px solid #484f58; border-radius: 5px; padding: 5px 8px; }
#legend { position: fixed; z-index: 2; bottom: 12px; left: 12px; padding: 9px 12px;
  border: 1px solid #30363d; border-radius: 7px; background: rgba(22,27,34,.92); font-size: 13px; }
.item { display: inline-flex; align-items: center; margin-right: 13px; padding: 2px 6px;
  border-radius: 5px; cursor: pointer; user-select: none; border: 1px solid transparent; }
.item:hover { border-color: #484f58; }
.item.off { opacity: .4; }
.item.off .dot { background: transparent !important; border: 1px solid currentColor; }
.dot { width: 10px; height: 10px; margin-right: 5px; border-radius: 50%; }
#info { margin-top: 7px; color: #9da7b3; }
canvas { display: block; width: 100vw; height: 100vh; cursor: grab; }
canvas.dragging { cursor: grabbing; }
</style>
</head>
<body>
<div id="toolbar"><label>座標グループ <select id="group"></select></label><button id="reset">表示をリセット</button><span id="count"></span></div>
<canvas id="view"></canvas>
<div id="legend">
  <span class="item" data-key="trajectory" style="color:#58a6ff"><span class="dot" style="background:#58a6ff"></span>軌跡</span>
  <span class="item" data-key="coverage" style="color:#f2cc60"><span class="dot" style="background:#f2cc60"></span>coverage</span>
  <span class="item" data-key="endpoint" style="color:#ff7b72"><span class="dot" style="background:#ff7b72"></span>endpoint</span>
  <span class="item" data-key="bridge_path" style="color:#7ee787"><span class="dot" style="background:#7ee787"></span>bridge_path</span>
  <div id="info">凡例をクリックで表示切替 / 抽出点へカーソルを合わせると情報を表示</div>
</div>
<script id="trajectory-data" type="application/json">__DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById('trajectory-data').textContent);
  const canvas = document.getElementById('view');
  const ctx = canvas.getContext('2d');
  const groupSelect = document.getElementById('group');
  const count = document.getElementById('count');
  const info = document.getElementById('info');
  const colors = {coverage:'#f2cc60', endpoint:'#ff7b72', bridge_path:'#7ee787'};
  const visible = {trajectory:true, coverage:true, endpoint:true, bridge_path:true};
  let yaw = -0.65, pitch = 0.5, zoom = 1, dragging = false, lastX = 0, lastY = 0;
  let group, center = [0,0,0], extent = 1, projectedSelected = [];

  for (const item of data.groups) {
    const option = document.createElement('option');
    option.value = item.id; option.textContent = item.label;
    groupSelect.appendChild(option);
  }

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(innerWidth * ratio); canvas.height = Math.round(innerHeight * ratio);
    ctx.setTransform(ratio,0,0,ratio,0,0); draw();
  }
  function reset() { yaw = -0.65; pitch = 0.5; zoom = 1; fit(); draw(); }
  function fit() {
    group = data.groups.find(g => String(g.id) === groupSelect.value) || data.groups[0];
    const points = group.segments.flatMap(s => s.points);
    const mins = [Infinity,Infinity,Infinity], maxs = [-Infinity,-Infinity,-Infinity];
    for (const p of points) for (let i=0;i<3;i++) { mins[i]=Math.min(mins[i],p[i]); maxs[i]=Math.max(maxs[i],p[i]); }
    center = mins.map((v,i)=>(v+maxs[i])/2);
    extent = Math.max(...maxs.map((v,i)=>v-mins[i]), 1e-9);
    count.textContent = `軌跡 ${points.length.toLocaleString()}点 / 抽出 ${group.selected.length.toLocaleString()}点`;
    info.textContent = '抽出点へカーソルを合わせると情報を表示';
  }
  function project(p) {
    const x=p[0]-center[0], y=p[1]-center[1], z=p[2]-center[2];
    const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
    const x1=cy*x+sy*z, z1=-sy*x+cy*z;
    const y1=cp*y-sp*z1, depth=sp*y+cp*z1;
    const scale=Math.min(innerWidth,innerHeight)*0.76/extent*zoom;
    // Keep stella_vslam's Y-down convention: data +Y maps to screen-down.
    return [innerWidth/2+x1*scale, innerHeight/2+y1*scale, depth];
  }
  function drawAxes() {
    const axes = [[[0,0,0],[extent*.2,0,0],'#ff7b72','X'],[[0,0,0],[0,extent*.2,0],'#7ee787','Y'],[[0,0,0],[0,0,extent*.2],'#58a6ff','Z']];
    ctx.lineWidth=2; ctx.font='12px system-ui';
    for (const [a,b,color,label] of axes) {
      const pa=project(a.map((v,i)=>v+center[i])), pb=project(b.map((v,i)=>v+center[i]));
      ctx.strokeStyle=color; ctx.beginPath(); ctx.moveTo(pa[0],pa[1]); ctx.lineTo(pb[0],pb[1]); ctx.stroke();
      ctx.fillStyle=color; ctx.fillText(label,pb[0]+4,pb[1]);
    }
  }
  function draw() {
    if (!group) return;
    ctx.clearRect(0,0,innerWidth,innerHeight); drawAxes();
    if (visible.trajectory) {
      ctx.strokeStyle='#58a6ff'; ctx.globalAlpha=.72; ctx.lineWidth=1.4;
      for (const segment of group.segments) {
        ctx.beginPath();
        segment.points.forEach((p,i) => { const q=project(p); i ? ctx.lineTo(q[0],q[1]) : ctx.moveTo(q[0],q[1]); });
        ctx.stroke();
      }
      ctx.globalAlpha=1;
    }
    projectedSelected = group.selected
      .filter(s => visible[s.type] !== false)
      .map(s => ({s, p:project(s.position)})).sort((a,b)=>a.p[2]-b.p[2]);
    for (const item of projectedSelected) {
      ctx.fillStyle=colors[item.s.type] || '#d2a8ff'; ctx.strokeStyle='#0d1117'; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.arc(item.p[0],item.p[1],5.5,0,Math.PI*2); ctx.fill(); ctx.stroke();
    }
  }
  canvas.addEventListener('pointerdown', e => { dragging=true; lastX=e.clientX; lastY=e.clientY; canvas.classList.add('dragging'); canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', e => {
    if (dragging) { yaw-=(e.clientX-lastX)*.008; pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-lastY)*.008)); lastX=e.clientX; lastY=e.clientY; draw(); return; }
    let best=null, d2=100;
    for (const item of projectedSelected) { const dx=e.clientX-item.p[0], dy=e.clientY-item.p[1], d=dx*dx+dy*dy; if(d<d2){d2=d;best=item.s;} }
    info.textContent = best ? `frame ${best.frame_index} / ${best.type} / order ${best.order} / (${best.position.map(v=>v.toPrecision(5)).join(', ')})` : '抽出点へカーソルを合わせると情報を表示';
  });
  canvas.addEventListener('pointerup', () => { dragging=false; canvas.classList.remove('dragging'); });
  canvas.addEventListener('wheel', e => { e.preventDefault(); zoom=Math.max(.08,Math.min(40,zoom*Math.exp(-e.deltaY*.001))); draw(); }, {passive:false});
  groupSelect.addEventListener('change', reset); document.getElementById('reset').addEventListener('click', reset);
  for (const item of document.querySelectorAll('#legend .item')) {
    item.addEventListener('click', () => {
      const key = item.dataset.key;
      visible[key] = !visible[key];
      item.classList.toggle('off', !visible[key]);
      draw();
    });
  }
  window.addEventListener('resize', resize); fit(); resize();
})();
</script>
</body>
</html>
"""


def write_trajectory_visualization(
    path: Path,
    trajectory: list[TrajectoryRecord],
    selections: list[SelectionRecord],
) -> None:
    """Write a self-contained interactive trajectory visualization."""
    selected_by_frame = {record.frame_index: record for record in selections}
    segments: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    selected: dict[int, list[dict[str, object]]] = defaultdict(list)

    for record in sorted(trajectory, key=lambda item: item.frame_index):
        position = [record.cx, record.cy, record.cz]
        if not record.pose_valid or not all(math.isfinite(value) for value in position):
            continue
        segments[(record.coordinate_group_id, record.segment_id)].append(position)
        selection = selected_by_frame.get(record.frame_index)
        if selection is not None:
            selected[record.coordinate_group_id].append(
                {
                    "frame_index": record.frame_index,
                    "order": selection.selection_order,
                    "type": selection.selection_type,
                    "position": position,
                }
            )

    group_ids = sorted({group_id for group_id, _ in segments})
    if not group_ids:
        raise ValueError("Trajectory has no finite valid poses to visualize")
    missing = sorted(set(selected_by_frame) - {item.frame_index for item in trajectory})
    if missing:
        raise ValueError(f"Selected frames are absent from trajectory: {missing[:5]}")

    groups = []
    for group_id in group_ids:
        group_segments = [
            {"id": segment_id, "points": points}
            for (candidate_group, segment_id), points in sorted(segments.items())
            if candidate_group == group_id and points
        ]
        groups.append(
            {
                "id": group_id,
                "label": f"{group_id} ({len(group_segments)} segments)",
                "segments": group_segments,
                "selected": sorted(selected[group_id], key=lambda item: int(item["order"])),
            }
        )

    payload = json.dumps({"groups": groups}, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    with atomic_text_writer(path) as handle:
        handle.write(_HTML_TEMPLATE.replace("__DATA__", payload))
