"""DataSphere Streamlit companion — mirrors index.html labs in Python.

Run: streamlit run "streamlit_app/app.py"
Covers: MySQL live lab (sqlite), Mongo playground (in-memory),
Neo4j graph (networkx), security + cross-DB benchmarks.
"""
import json
import sqlite3
import time
from collections import Counter, deque

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="DataSphere — Streamlit labs", layout="wide")

# ---------------------------------------------------------------- DATA
# Same sample numbers as the DATA object in index.html
DATA = {
    "fit": {"labels": ["Joins & integrity", "Flexible schema", "Traversal",
                        "Aggregation", "Horizontal scale"],
            "sql": [5, 2, 2, 5, 2], "mongo": [2, 5, 2, 4, 5], "neo": [2, 3, 5, 2, 3]},
    "rev": {"labels": ["Shoes", "Electronics", "Home", "Books", "Beauty", "Sports"],
            "v": [412, 689, 355, 198, 243, 301]},
    "idx": {"labels": ["Point lookup", "Range scan", "3-table join", "Top-N by group"],
            "before": [420, 1900, 5200, 3100], "after": [0.4, 38, 160, 95]},
    "hop": {"labels": ["1 hop", "2 hops", "3 hops", "4 hops", "5 hops"],
            "sql": [2, 18, 240, 3800, 52000], "neo": [1, 2, 5, 22, 140]},
    "tx": {"labels": ["READ UNCOMMITTED", "READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"],
            "v": [9800, 9400, 8700, 4100]},
    "lat": {"labels": ["Point lookup", "Range scan", "Join (3)", "Aggregate",
                       "6-hop traversal", "Doc by nested field"],
            "sql": [0.4, 38, 160, 95, 52000, 210],
            "mongo": [0.5, 25, 900, 70, 24000, 3],
            "neo": [1.2, 60, 300, 180, 140, 95]},
    "scale": {"labels": ["1M", "5M", "10M", "50M", "100M"],
              "sql": [52, 44, 38, 24, 15], "mongo": [48, 45, 43, 39, 35],
              "neo": [30, 27, 24, 17, 11]},
    "store": {"labels": ["MySQL", "MongoDB", "Neo4j"], "v": [4.2, 5.6, 9.8]},
}

SQL_PRESETS = {
    "Top spenders (window)": """SELECT c.name, SUM(o.amount) AS total,
  RANK() OVER (ORDER BY SUM(o.amount) DESC) AS rnk
FROM orders o JOIN customers c ON c.id=o.customer_id
GROUP BY c.id ORDER BY total DESC LIMIT 10;""",
    "Monthly revenue (CTE)": """WITH m AS (
  SELECT substr(placed_at,1,7) AS month, SUM(amount) AS rev
  FROM orders GROUP BY month)
SELECT month, ROUND(rev) AS rev,
  ROUND(AVG(rev) OVER (ORDER BY month ROWS 2 PRECEDING)) AS avg3
FROM m ORDER BY month LIMIT 12;""",
    "Category tree (recursive)": """WITH RECURSIVE t(id,name,depth) AS (
  SELECT id,name,0 FROM categories WHERE parent_id IS NULL
  UNION ALL
  SELECT c.id,c.name,t.depth+1 FROM categories c JOIN t ON c.parent_id=t.id)
SELECT name AS category, depth FROM t;""",
    "Lookup (plan demo)": "SELECT COUNT(*), ROUND(SUM(amount)) FROM orders WHERE customer_id = 42;",
}

MONGO_PRESETS = {
    "Avg rating by category": [{"$unwind": "$reviews"},
                               {"$group": {"_id": "$category", "avg": {"$avg": "$reviews.stars"}}},
                               {"$sort": {"avg": -1}}],
    "Premium shoes": [{"$match": {"category": "Shoes"}}, {"$sort": {"price": -1}}, {"$limit": 5}],
    "Price stats": [{"$group": {"_id": "$category", "avgPrice": {"$avg": "$price"}}}],
}

# ------------------------------------------------------- cached builders
@st.cache_resource(show_spinner="Building SQLite dataset...")
def build_sqlite(n_orders: int = 100_000, n_cust: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(":memory:", check_same_thread=False)
    cur = con.cursor()
    cur.executescript("""CREATE TABLE customers(id INTEGER PRIMARY KEY,name TEXT,city TEXT);
    CREATE TABLE orders(id INTEGER PRIMARY KEY,customer_id INT,placed_at TEXT,amount REAL,category TEXT);
    CREATE TABLE categories(id INTEGER PRIMARY KEY,name TEXT,parent_id INT);
    INSERT INTO categories VALUES(1,'All',NULL),(2,'Electronics',1),(3,'Phones',2),(4,'Laptops',2),
    (5,'Home',1),(6,'Kitchen',5),(7,'Garden',5),(8,'Books',1);""")
    cities = ["London", "Bengaluru", "Pune", "Berlin", "Austin", "Lagos"]
    cats = ["Shoes", "Electronics", "Home", "Books", "Beauty", "Sports"]
    cur.executemany("INSERT INTO customers VALUES(?,?,?)",
                    [(i + 1, f"Customer {i+1}", cities[i % 6]) for i in range(n_cust)])
    cust = rng.power(1.6, n_orders)  # skew like the JS demo (heavy buyers)
    cust_ids = np.clip((cust * n_cust).astype(int) + 1, 1, n_cust)
    amounts = np.round((5 + rng.power(2, n_orders) * 400) * 100) / 100
    days = rng.integers(0, 730, n_orders)
    base = np.datetime64("2023-01-01")
    dates = [(base + int(d)).astype(str) for d in days]
    cat_pick = rng.integers(0, 6, n_orders)
    cur.executemany("INSERT INTO orders VALUES(?,?,?,?,?)",
                    [(i + 1, int(cust_ids[i]), dates[i], float(amounts[i]), cats[cat_pick[i]])
                     for i in range(n_orders)])
    con.commit()
    return con


@st.cache_data(show_spinner="Generating product documents...")
def gen_products(n: int = 20000, seed: int = 7):
    rng = np.random.default_rng(seed)
    cats = ["Shoes", "Electronics", "Home", "Books", "Beauty", "Sports"]
    docs = []
    for i in range(n):
        nrev = int(rng.integers(0, 7))
        docs.append({"_id": i + 1, "name": f"Product {i+1}",
                     "category": cats[int(rng.integers(0, 6))],
                     "price": round(float(10 + rng.power(2) * 290), 2),
                     "reviews": [{"user": int(rng.integers(0, 5000)),
                                  "stars": int(1 + (rng.power(0.6) * 5))} for _ in range(nrev)]})
    return docs


@st.cache_data(show_spinner="Generating customer network...")
def gen_graph(n: int = 60, seed: int = 7):
    s = seed
    def rnd():
        nonlocal s
        s = (s * 16807) % 2147483647
        return s / 2147483647
    edges, deg = [], [0] * n
    for i in range(1, n):
        for _ in range(1 if i < 4 else 2):
            tot = sum(d + 1 for d in deg[:i])
            r, j = rnd() * tot, 0
            while j < i - 1 and (r := r - (deg[j] + 1)) > 0:
                j += 1
            if (i, j) not in edges and (j, i) not in edges:
                edges.append((i, j))
                deg[i] += 1
                deg[j] += 1
    # PageRank (same .15/.85 power iteration as index.html)
    out = [0] * n
    for a, _ in edges:
        out[a] += 1
    pr = [1 / n] * n
    for _ in range(50):
        nxt = [.15 / n] * n
        for a, b in edges:
            nxt[b] += .85 * pr[a] / max(out[a], 1)
        pr = nxt
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return edges, deg, pr, adj


def bfs_path(adj, a, b):
    prev = {a: None}
    q = deque([a])
    while q:
        u = q.popleft()
        if u == b:
            break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    if b not in prev:
        return []
    path, cur = [], b
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def run_pipeline(docs, pipeline):
    """Tiny $match/$unwind/$group/$sort/$limit/$project engine (mirrors the JS demo)."""
    def get(d, p):
        cur = d
        for k in p.split("."):
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                return None
        return cur
    data = docs
    for stg in pipeline:
        (op, arg), = stg.items()
        if op == "$match":
            data = [d for d in data if all(
                (get(d, k) == v if not isinstance(v, dict)
                 else {"$gte": lambda a, b: a is not None and a >= b,
                       "$gt": lambda a, b: a is not None and a > b,
                       "$lte": lambda a, b: a is not None and a <= b,
                       "$lt": lambda a, b: a is not None and a < b}.get(o, lambda a, b: a == b)(get(d, k), t)
                 for o, t in (v.items() if isinstance(v, dict) else [])) or get(d, k) == v
                for k, v in arg.items())]
        elif op == "$unwind":
            f = arg[1:]
            data = [{**d, f: x} for d in data for x in (d.get(f) or [])]
        elif op == "$group":
            groups = {}
            for d in data:
                key = get(d, arg["_id"][1:]) if isinstance(arg["_id"], str) and arg["_id"].startswith("$") else arg["_id"]
                g = groups.setdefault(json.dumps(key, sort_keys=True, default=str), {"_id": key, "_rows": []})
                g["_rows"].append(d)
            out = []
            for g in groups.values():
                row = {"_id": g["_id"]}
                for f, expr in arg.items():
                    if f == "_id":
                        continue
                    (fn, src), = expr.items()
                    vals = [get(r, src[1:]) if isinstance(src, str) and src.startswith("$") else src for r in g["_rows"]]
                    vals = [v for v in vals if isinstance(v, (int, float))]
                    row[f] = {"$avg": float(np.mean(vals)) if vals else None,
                              "$sum": float(np.sum(vals)) if fn == "$sum" and all(isinstance(v, (int, float)) for v in vals) else len(g["_rows"]),
                              "$max": max(vals) if vals else None,
                              "$min": min(vals) if vals else None}.get(fn)
                out.append(row)
            data = out
        elif op == "$sort":
            for k in reversed(list(arg)):
                data = sorted(data, key=lambda d, k=k: (get(d, k) is None, get(d, k)), reverse=arg[k] < 0)
        elif op == "$limit":
            data = data[:arg]
        elif op == "$project":
            data = [{k: get(d, k) for k in arg if arg[k]} for d in data]
        else:
            raise ValueError(f"Unsupported stage {op}")
    return data


# ------------------------------------------------------------------ UI
st.title("DataSphere — one dataset, three databases")
st.caption("Streamlit companion to index.html: real SQLite queries, in-memory aggregations, live graph metrics. Sample-data charts are labelled as such.")

tab_sql, tab_mongo, tab_graph, tab_bench, tab_sec = st.tabs(
    ["MySQL lab", "MongoDB playground", "Neo4j graph", "Benchmarks", "Transactions + Security"])

with tab_sql:
    st.header("Live SQL lab")
    st.write("Builds the orders database in-process (SQLite — same idea as the sql.js demo in the page) and times real queries on your machine.")
    n_orders = st.select_slider("Dataset size", [10_000, 50_000, 100_000, 200_000], value=100_000,
                                format_func=lambda x: f"{x//1000}k orders")
    con = build_sqlite(n_orders)
    preset = st.selectbox("Preset", list(SQL_PRESETS))
    q = st.text_area("SQL", SQL_PRESETS[preset], height=140)
    c1, c2 = st.columns([1, 5])
    if c1.button("Run query", type="primary"):
        try:
            t = time.perf_counter()
            df = pd.read_sql_query(q, con)
            dt = (time.perf_counter() - t) * 1000
            c2.success(f"{len(df)} rows in {dt:.1f} ms (measured here)")
            st.dataframe(df.head(12))
            try:
                plan = pd.read_sql_query("EXPLAIN QUERY PLAN " + q, con)
                st.code("\n".join(plan.iloc[:, -1].astype(str)), language="text")
            except Exception:
                pass
        except Exception as e:
            st.error(f"Error: {e}")
    st.subheader("Indexing benchmark (measured live)")
    if st.button("Run benchmark — 300 customer lookups"):
        cur = con.cursor()
        cur.execute("DROP INDEX IF EXISTS i1")
        ids = [(1 + (i * 37) % 5000,) for i in range(300)]
        t = time.perf_counter()
        for (i,) in ids:
            cur.execute("SELECT COUNT(*),SUM(amount) FROM orders WHERE customer_id=?", (i,)).fetchone()
        t_no = (time.perf_counter() - t) / 300 * 1000
        cur.execute("CREATE INDEX i1 ON orders(customer_id)")
        t = time.perf_counter()
        for (i,) in ids:
            cur.execute("SELECT COUNT(*),SUM(amount) FROM orders WHERE customer_id=?", (i,)).fetchone()
        t_idx = (time.perf_counter() - t) / 300 * 1000
        cur.execute("DROP INDEX i1")
        fig = px.bar(x=["No index", "Index (customer_id)"], y=[t_no, t_idx], log_y=True,
                     labels={"x": "Strategy", "y": "ms per query (log)"}, title="Measured live")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Index is ~{t_no/max(t_idx,1e-9):.0f}× faster here.")
    st.subheader("Revenue by category (sample data)")
    st.plotly_chart(px.bar(x=DATA["rev"]["labels"], y=DATA["rev"]["v"],
                           labels={"x": "Category", "y": "Revenue (£k)"}), use_container_width=True)

with tab_mongo:
    st.header("Aggregation playground")
    st.write("20,000 generated product documents + a small Python aggregation engine ($match, $unwind, $group, $sort, $limit, $project) — same presets as the page.")
    docs = gen_products()
    preset = st.selectbox("Pipeline preset", list(MONGO_PRESETS))
    pipe_txt = st.text_area("Pipeline (JSON)", json.dumps(MONGO_PRESETS[preset], indent=1), height=180)
    if st.button("Run pipeline", type="primary"):
        try:
            t = time.perf_counter()
            out = run_pipeline(docs, json.loads(pipe_txt))
            dt = (time.perf_counter() - t) * 1000
            st.success(f"{len(out)} documents out in {dt:.1f} ms (measured here)")
            st.dataframe(pd.DataFrame(out).head(8))
        except Exception as e:
            st.error(f"Error: {e}")
    st.subheader("Index impact (sample data, ms log scale)")
    df = pd.DataFrame({"query": ["Equality", "Range", "Text", "Compound"],
                       "scan": [310, 420, 980, 560], "indexed": [0.8, 14, 22, 6]})
    st.plotly_chart(px.bar(df, x="query", y=["scan", "indexed"], barmode="group", log_y=True,
                           labels={"value": "ms (log)", "query": "Query type"}), use_container_width=True)

with tab_graph:
    st.header("Customer network (computed live)")
    edges, deg, pr, adj = gen_graph()
    names = [f"Customer {i+1}" for i in range(60)]
    order = sorted(range(60), key=lambda i: -pr[i])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top PageRank")
        top = order[:8]
        st.plotly_chart(px.bar(x=[pr[i] for i in top], y=[names[i] for i in top], orientation="h",
                               labels={"x": "PageRank", "y": "Customer"}), use_container_width=True)
    with c2:
        st.subheader("Degree distribution")
        hist = Counter(deg)
        st.plotly_chart(px.bar(x=list(hist), y=[hist[k] for k in sorted(hist)],
                               labels={"x": "Links per customer", "y": "Customers"}), use_container_width=True)
    st.subheader("Network (spring layout, sized by PageRank)")
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(60))
        G.add_edges_from(edges)
        pos = nx.spring_layout(G, seed=7)
        ex, ey = [], []
        for a, b in edges:
            ex += [pos[a][0], pos[b][0], None]
            ey += [pos[a][1], pos[b][1], None]
        fig = go.Figure([go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#bbb", width=1), hoverinfo="skip"),
                         go.Scatter(x=[pos[i][0] for i in range(60)], y=[pos[i][1] for i in range(60)],
                                    mode="markers", marker=dict(size=[6 + 30 * p / max(pr) for p in pr], color="#D55E00"),
                                    text=[f"{names[i]} · {deg[i]} links · PR {pr[i]:.4f}" for i in range(60)])])
        fig.update_layout(showlegend=False, height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Graph draw skipped: {e}")
    st.subheader("Shortest path (BFS — like Cypher shortestPath)")
    a = st.selectbox("From", names, index=0)
    b = st.selectbox("To", names, index=41)
    ia, ib = names.index(a), names.index(b)
    path = bfs_path(adj, ia, ib)
    st.write(f"{' → '.join(names[i] for i in path)} ({len(path)-1} hops)" if path else "No path")
    st.subheader("Traversal depth: SQL joins vs Cypher (sample data, ms log)")
    df = pd.DataFrame({"hops": DATA["hop"]["labels"], "MySQL joins": DATA["hop"]["sql"], "Neo4j": DATA["hop"]["neo"]})
    st.plotly_chart(px.line(df, x="hops", y=["MySQL joins", "Neo4j"], log_y=True, markers=True,
                            labels={"value": "ms (log)", "hops": "Hop depth"}), use_container_width=True)

with tab_bench:
    st.header("Cross-database benchmarks (sample data)")
    df = pd.DataFrame({"workload": DATA["lat"]["labels"], "MySQL": DATA["lat"]["sql"],
                       "MongoDB": DATA["lat"]["mongo"], "Neo4j": DATA["lat"]["neo"]})
    st.plotly_chart(px.bar(df, x="workload", y=["MySQL", "MongoDB", "Neo4j"], barmode="group", log_y=True,
                           labels={"value": "Median ms (log)"}, title="Query latency by workload"), use_container_width=True)
    df = pd.DataFrame({"size": DATA["scale"]["labels"], "MySQL": DATA["scale"]["sql"],
                       "MongoDB": DATA["scale"]["mongo"], "Neo4j": DATA["scale"]["neo"]})
    st.plotly_chart(px.line(df, x="size", y=["MySQL", "MongoDB", "Neo4j"], markers=True,
                            labels={"value": "Throughput (k ops/s)"}, title="Scalability"), use_container_width=True)
    st.plotly_chart(px.pie(values=DATA["store"]["v"], names=DATA["store"]["labels"],
                           title="Storage overhead (GB, same data)"), use_container_width=True)

with tab_sec:
    st.header("Transactions + Security")
    st.subheader("Throughput by isolation level (sample, 32 clients TPS)")
    st.plotly_chart(px.bar(x=DATA["tx"]["labels"], y=DATA["tx"]["v"],
                           labels={"x": "Isolation level", "y": "TPS"}), use_container_width=True)
    st.write("*InnoDB REPEATABLE READ uses gap locks + MVCC snapshot, preventing most phantoms.*")
    st.subheader("Injection: try it")
    name = st.text_input("Name input", "x' OR '1'='1")
    st.code(f"SELECT * FROM users WHERE name = '{name}'  -- unsafe: concatenated", language="sql")
    st.code(f"SELECT * FROM users WHERE name = ?  -- safe, bound param: {name!r}", language="sql")
    if "'" in name and ("or" in name.lower() or "--" in name or ";" in name):
        st.error("Unsafe query changes meaning and returns every row. The parameterised one treats input as a name.")
    else:
        st.success("Harmless input in both versions. Try one containing a quote.")
