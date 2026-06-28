import React, { useState, useMemo, useRef, useEffect } from "react";

// ──────────────────────────────────────────────────────────────
// サンプル事例データ
// 本番では各文書をAIが読み取り、課題・施策・成果を抽出して
// 埋め込みベクトルに変換します。ここではタグの重なりで
// 「意味の近さ」を擬似的に再現しています。
// ──────────────────────────────────────────────────────────────
const CASES = [
  {
    id: 1, title: "若手の早期離職を社内メンターで抑制",
    industry: "IT・SaaS", problem: "入社2年以内の若手が次々辞めてしまう",
    action: "1on1とメンター制度を導入し、配属後90日の伴走支援を仕組み化",
    result: "2年以内離職率が28%→11%に低下",
    tags: ["離職", "定着", "若手", "メンター", "1on1", "組織"],
  },
  {
    id: 2, title: "問い合わせ対応の属人化を解消",
    industry: "製造", problem: "ベテラン頼みでサポート品質がばらつく",
    action: "対応履歴をナレッジ化し、検索できる事例DBを整備",
    result: "平均解決時間を40%短縮、新人の独り立ちが2倍速に",
    tags: ["属人化", "ナレッジ", "サポート", "標準化", "検索"],
  },
  {
    id: 3, title: "ECサイトの離脱率をUI改善で改善",
    industry: "小売・EC", problem: "カート投入後に購入まで進まず離脱が多い",
    action: "決済ステップを5→2画面に短縮、入力補助を追加",
    result: "カート離脱率が68%→49%、CVが1.5倍",
    tags: ["離脱", "UI", "CVR", "EC", "決済", "改善"],
  },
  {
    id: 4, title: "現場の暗黙知を動画マニュアル化",
    industry: "製造", problem: "熟練者のノウハウが引き継がれない",
    action: "作業を撮影し手順を動画＋検索タグで整理",
    result: "教育期間を3ヶ月→1ヶ月に短縮",
    tags: ["属人化", "ナレッジ", "標準化", "教育", "現場"],
  },
  {
    id: 5, title: "営業の提案資料づくりを半自動化",
    industry: "IT・SaaS", problem: "提案準備に時間がかかり商談数が伸びない",
    action: "過去事例を検索し下書きを生成する仕組みを構築",
    result: "提案準備時間を60%削減",
    tags: ["効率化", "営業", "検索", "事例", "自動化"],
  },
  {
    id: 6, title: "従業員エンゲージメント調査で定着改善",
    industry: "サービス", problem: "退職理由が見えず手を打てない",
    action: "四半期サーベイで不満を可視化し部署別に施策化",
    result: "エンゲージメントスコア+18pt、離職率改善",
    tags: ["離職", "定着", "サーベイ", "組織", "可視化"],
  },
  {
    id: 7, title: "サポート問い合わせをFAQ自動応答で削減",
    industry: "IT・SaaS", problem: "同じ質問が多く対応コストが膨らむ",
    action: "問い合わせを分析し意味検索型FAQを設置",
    result: "有人対応件数を35%削減",
    tags: ["サポート", "検索", "効率化", "自動化", "FAQ"],
  },
  {
    id: 8, title: "店舗オペレーションの標準化",
    industry: "小売・EC", problem: "店舗ごとに接客品質がばらつく",
    action: "優良店の手順を抽出しチェックリスト化",
    result: "顧客満足度の店舗間ばらつきが半減",
    tags: ["属人化", "標準化", "現場", "品質", "改善"],
  },
];

// 全タグを語彙として扱い、各事例を多次元ベクトルに見立てる
const VOCAB = Array.from(new Set(CASES.flatMap((c) => c.tags)));

// クエリ（問題文）を入力タグ集合に変換 → コサイン類似度を計算
function vec(tags) {
  return VOCAB.map((t) => (tags.includes(t) ? 1 : 0));
}
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

// 問題文 → 関連しそうなタグを素朴に推定（本番は埋め込みモデルが担う）
function inferTags(text) {
  const t = text.toLowerCase();
  const map = {
    離職: ["辞め", "離職", "退職", "定着しない", "辞める", "やめ"],
    定着: ["定着", "残ら", "続かない"],
    属人化: ["属人", "ベテラン頼み", "特定の人", "暗黙", "ノウハウ"],
    ナレッジ: ["ナレッジ", "知識", "共有", "引き継"],
    離脱: ["離脱", "途中で", "やめてしまう", "進まない"],
    UI: ["ui", "画面", "使いにくい", "入力"],
    CVR: ["購入", "成約", "コンバージョン", "cv"],
    サポート: ["問い合わせ", "サポート", "対応", "質問"],
    効率化: ["時間がかかる", "効率", "工数", "コスト", "遅い"],
    検索: ["探せない", "見つからない", "検索"],
    標準化: ["ばらつき", "品質", "標準"],
    教育: ["教育", "育成", "新人", "独り立ち"],
    現場: ["現場", "店舗", "作業"],
    組織: ["組織", "チーム", "風土"],
  };
  const hit = [];
  for (const [tag, kws] of Object.entries(map)) {
    if (kws.some((k) => t.includes(k))) hit.push(tag);
  }
  return hit;
}

const PALETTE = {
  bg: "#0E1116", panel: "#161B22", line: "#2A323D",
  ink: "#E8EDF2", sub: "#8B96A5", accent: "#4DD6C1",
  accent2: "#F2B441", node: "#3A4757", center: "#4DD6C1",
};

export default function CaseFinder() {
  const [query, setQuery] = useState("若手がすぐ辞めてしまって定着しない");
  const [submitted, setSubmitted] = useState("若手がすぐ辞めてしまって定着しない");
  const [selected, setSelected] = useState(null);

  const inferred = useMemo(() => inferTags(submitted), [submitted]);
  const qVec = useMemo(() => vec(inferred), [inferred]);

  const ranked = useMemo(() => {
    return CASES
      .map((c) => ({ ...c, score: cosine(qVec, vec(c.tags)) }))
      .sort((a, b) => b.score - a.score);
  }, [qVec]);

  const top = ranked.filter((c) => c.score > 0).slice(0, 6);
  const hasResults = top.length > 0;

  return (
    <div style={{
      minHeight: "100vh", background: PALETTE.bg, color: PALETTE.ink,
      fontFamily: "'Inter', system-ui, sans-serif", padding: "28px 20px 60px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        * { box-sizing: border-box; }
        .gnode { cursor: pointer; transition: opacity .15s; }
        .gnode:hover { opacity: .85; }
        input::placeholder { color: ${PALETTE.sub}; }
        @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
      `}</style>

      <div style={{ maxWidth: 1080, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 4 }}>
          <h1 style={{
            fontFamily: "'Space Grotesk', sans-serif", fontSize: 26, fontWeight: 700,
            margin: 0, letterSpacing: "-0.01em",
          }}>事例ファインダー</h1>
          <span style={{ color: PALETTE.accent, fontSize: 13, fontWeight: 600 }}>意味検索 × 関係グラフ</span>
        </div>
        <p style={{ color: PALETTE.sub, fontSize: 14, margin: "0 0 22px" }}>
          困っている問題を入力すると、意味の近い事例を自動で並べ、事例同士のつながりを図で表示します。
        </p>

        {/* 検索ボックス */}
        <div style={{ display: "flex", gap: 10, marginBottom: 8 }}>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") { setSubmitted(query); setSelected(null); } }}
            placeholder="例：問い合わせ対応がベテラン頼みでばらつく"
            style={{
              flex: 1, background: PALETTE.panel, border: `1px solid ${PALETTE.line}`,
              borderRadius: 10, padding: "13px 15px", color: PALETTE.ink, fontSize: 15, outline: "none",
            }}
          />
          <button
            onClick={() => { setSubmitted(query); setSelected(null); }}
            style={{
              background: PALETTE.accent, color: "#06231F", border: "none", borderRadius: 10,
              padding: "0 22px", fontWeight: 700, fontSize: 15, cursor: "pointer",
            }}
          >探す</button>
        </div>

        {/* 推定された意味タグ */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 24, minHeight: 24, alignItems: "center" }}>
          <span style={{ color: PALETTE.sub, fontSize: 12 }}>読み取った観点：</span>
          {inferred.length ? inferred.map((t) => (
            <span key={t} style={{
              fontSize: 12, color: PALETTE.accent2, border: `1px solid ${PALETTE.accent2}55`,
              borderRadius: 20, padding: "2px 10px",
            }}>{t}</span>
          )) : <span style={{ color: PALETTE.sub, fontSize: 12 }}>—（言い回しを変えてみてください）</span>}
        </div>

        {!hasResults ? (
          <div style={{
            background: PALETTE.panel, border: `1px dashed ${PALETTE.line}`, borderRadius: 12,
            padding: 40, textAlign: "center", color: PALETTE.sub,
          }}>
            近い事例が見つかりませんでした。「離職」「属人化」「離脱」「効率化」などの観点を含む言い回しを試してください。
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1.1fr) minmax(0,1fr)", gap: 20 }}>
            {/* グラフ */}
            <div style={{
              background: PALETTE.panel, border: `1px solid ${PALETTE.line}`, borderRadius: 14, padding: 8,
            }}>
              <Graph cases={top} selected={selected} onSelect={setSelected} />
              <div style={{ color: PALETTE.sub, fontSize: 11.5, padding: "2px 10px 8px" }}>
                中心＝あなたの問題。線が太い・近いほど意味が似ています。点をタップで詳細表示。
              </div>
            </div>

            {/* ランキング／詳細 */}
            <div>
              {selected ? (
                <Detail c={selected} onBack={() => setSelected(null)} all={top} />
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {top.map((c, i) => (
                    <button key={c.id} onClick={() => setSelected(c)} style={{
                      textAlign: "left", background: PALETTE.panel, border: `1px solid ${PALETTE.line}`,
                      borderRadius: 12, padding: "13px 15px", cursor: "pointer", color: PALETTE.ink,
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
                        <span style={{ fontWeight: 600, fontSize: 14.5 }}>{c.title}</span>
                        <Match v={c.score} />
                      </div>
                      <div style={{ color: PALETTE.sub, fontSize: 12.5, marginTop: 4 }}>{c.industry}・{c.problem}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Match({ v }) {
  const pct = Math.round(v * 100);
  return (
    <span style={{
      flexShrink: 0, fontSize: 11.5, fontWeight: 700, color: PALETTE.center,
      background: "#4DD6C122", borderRadius: 20, padding: "2px 9px",
    }}>合致 {pct}%</span>
  );
}

function Detail({ c, onBack, all }) {
  const related = all.filter((x) => x.id !== c.id)
    .map((x) => ({ ...x, sim: cosine(vec(c.tags), vec(x.tags)) }))
    .filter((x) => x.sim > 0).sort((a, b) => b.sim - a.sim).slice(0, 3);
  const row = (label, val, color) => (
    <div style={{ marginBottom: 12 }}>
      <div style={{ color: PALETTE.sub, fontSize: 11.5, marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 14, lineHeight: 1.55, color: color || PALETTE.ink }}>{val}</div>
    </div>
  );
  return (
    <div style={{ background: PALETTE.panel, border: `1px solid ${PALETTE.line}`, borderRadius: 14, padding: 18 }}>
      <button onClick={onBack} style={{
        background: "none", border: "none", color: PALETTE.accent, cursor: "pointer",
        fontSize: 13, padding: 0, marginBottom: 12,
      }}>← 一覧へ戻る</button>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", gap: 8, marginBottom: 14 }}>
        <h3 style={{ margin: 0, fontSize: 17, fontWeight: 700 }}>{c.title}</h3>
        <Match v={c.score} />
      </div>
      {row("業種", c.industry)}
      {row("課題", c.problem)}
      {row("打った施策", c.action)}
      {row("成果", c.result, PALETTE.accent)}
      <div style={{ borderTop: `1px solid ${PALETTE.line}`, margin: "8px 0 12px" }} />
      <div style={{ color: PALETTE.sub, fontSize: 11.5, marginBottom: 8 }}>つながりの強い事例</div>
      {related.map((r) => (
        <div key={r.id} style={{ fontSize: 13, marginBottom: 6, color: PALETTE.ink }}>
          <span style={{ color: PALETTE.accent2 }}>◦</span> {r.title}
          <span style={{ color: PALETTE.sub }}>（類似 {Math.round(r.sim * 100)}%）</span>
        </div>
      ))}
    </div>
  );
}

// ── 力学レイアウト風の簡易グラフ ──
function Graph({ cases, selected, onSelect }) {
  const W = 520, H = 420, cx = W / 2, cy = H / 2;
  const ref = useRef(null);

  const nodes = useMemo(() => {
    return cases.map((c, i) => {
      const angle = (i / cases.length) * Math.PI * 2 - Math.PI / 2;
      const dist = 70 + (1 - c.score) * 130; // 似ているほど中心に近い
      return {
        ...c, x: cx + Math.cos(angle) * dist, y: cy + Math.sin(angle) * dist,
        r: 13 + c.score * 12,
      };
    });
  }, [cases]);

  return (
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      {/* 中心への線 */}
      {nodes.map((n) => (
        <line key={"l" + n.id} x1={cx} y1={cy} x2={n.x} y2={n.y}
          stroke={PALETTE.accent} strokeOpacity={0.15 + n.score * 0.5}
          strokeWidth={0.6 + n.score * 3} />
      ))}
      {/* 事例同士の弱いつながり */}
      {nodes.map((a, i) => nodes.slice(i + 1).map((b) => {
        const sim = cosine(vec(a.tags), vec(b.tags));
        if (sim < 0.35) return null;
        return <line key={`e${a.id}-${b.id}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
          stroke={PALETTE.sub} strokeOpacity={sim * 0.3} strokeWidth={sim * 1.5} />;
      }))}
      {/* 中心ノード */}
      <circle cx={cx} cy={cy} r={26} fill={PALETTE.center} />
      <text x={cx} y={cy - 2} textAnchor="middle" fontSize="11" fontWeight="700" fill="#06231F">あなたの</text>
      <text x={cx} y={cy + 11} textAnchor="middle" fontSize="11" fontWeight="700" fill="#06231F">問題</text>
      {/* 事例ノード */}
      {nodes.map((n) => {
        const on = selected && selected.id === n.id;
        return (
          <g key={n.id} className="gnode" onClick={() => onSelect(n)}>
            <circle cx={n.x} cy={n.y} r={n.r}
              fill={on ? PALETTE.accent2 : PALETTE.node}
              stroke={on ? PALETTE.accent2 : PALETTE.accent}
              strokeOpacity={on ? 1 : 0.5} strokeWidth={on ? 2.5 : 1.2} />
            <text x={n.x} y={n.y + n.r + 13} textAnchor="middle" fontSize="10.5"
              fill={on ? PALETTE.accent2 : PALETTE.sub}>
              {n.title.length > 12 ? n.title.slice(0, 11) + "…" : n.title}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
