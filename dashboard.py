"""
BankGuard Analytics — Interactive Fraud Detection Dashboard
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ──
st.set_page_config(
    page_title="BankGuard Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp { background-color: #f8f6f3; }
    .metric-card {
        background: white; border-radius: 12px; padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #ebe4dc; text-align: center;
    }
    .metric-value { font-size: 36px; font-weight: 900; color: #1a1a1a; }
    .metric-label { font-size: 12px; color: #7a7267; text-transform: uppercase; letter-spacing: 1px; }
    .risk-critical { color: #7f1d1d; } .risk-high { color: #ef4444; }
    .risk-medium { color: #f59e0b; } .risk-low { color: #22c55e; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and process data with feature engineering + models."""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.neighbors import LocalOutlierFactor

    df = pd.read_csv('bank_transactions.csv')
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format='mixed')

    # Feature engineering
    acct = df.groupby('AccountID').agg(
        acct_txn_count=('TransactionID','count'), acct_mean_amount=('TransactionAmount','mean'),
        acct_std_amount=('TransactionAmount','std'), acct_max_amount=('TransactionAmount','max'),
        acct_mean_duration=('TransactionDuration','mean'), acct_mean_login=('LoginAttempts','mean'),
        acct_unique_devices=('DeviceID','nunique'), acct_unique_locations=('Location','nunique'),
    ).reset_index()
    acct['acct_std_amount'] = acct['acct_std_amount'].fillna(0)
    df = df.merge(acct, on='AccountID', how='left')

    df['amount_zscore'] = (df['TransactionAmount'] - df['acct_mean_amount']) / df['acct_std_amount'].replace(0,1)
    df['amount_to_balance_ratio'] = df['TransactionAmount'] / df['AccountBalance'].replace(0,1)
    df['amount_to_max_ratio'] = df['TransactionAmount'] / df['acct_max_amount'].replace(0,1)
    df['duration_deviation'] = abs(df['TransactionDuration'] - df['acct_mean_duration'])
    df['high_login_flag'] = (df['LoginAttempts'] >= 3).astype(int)
    Q1, Q3 = df['TransactionAmount'].quantile(0.25), df['TransactionAmount'].quantile(0.75)
    df['amount_outlier_flag'] = (df['TransactionAmount'] > Q3 + 1.5*(Q3-Q1)).astype(int)

    dev = df.groupby('DeviceID')['AccountID'].nunique().reset_index()
    dev.columns = ['DeviceID', 'device_shared_accounts']
    df = df.merge(dev, on='DeviceID', how='left')
    df['device_shared_flag'] = (df['device_shared_accounts'] > 1).astype(int)

    ip_cnt = df.groupby('IP Address')['AccountID'].nunique().reset_index()
    ip_cnt.columns = ['IP Address', 'ip_shared_accounts']
    df = df.merge(ip_cnt, on='IP Address', how='left')

    mf = df.groupby(['AccountID','MerchantID']).size().reset_index(name='merchant_visit_count')
    df = df.merge(mf, on=['AccountID','MerchantID'], how='left')
    df['is_new_merchant'] = (df['merchant_visit_count'] == 1).astype(int)

    df['is_weekend'] = (df['TransactionDate'].dt.dayofweek >= 5).astype(int)
    df['txn_date'] = df['TransactionDate'].dt.date
    dv = df.groupby(['AccountID','txn_date']).size().reset_index(name='daily_txn_count')
    df = df.merge(dv, on=['AccountID','txn_date'], how='left')
    df = df.sort_values(['AccountID','TransactionDate'])
    df['time_since_last_txn'] = df.groupby('AccountID')['TransactionDate'].diff().dt.total_seconds()/3600
    df['time_since_last_txn'] = df['time_since_last_txn'].fillna(-1)
    df['rapid_txn_flag'] = ((df['time_since_last_txn']>=0)&(df['time_since_last_txn']<1)).astype(int)
    df.drop(columns=['txn_date'], inplace=True)

    # Rules
    df['R1_high_login'] = (df['LoginAttempts'] >= 3).astype(int)
    df['R2_amount_spike'] = (df['amount_zscore'] > 2).astype(int)
    df['R3_high_ratio'] = (df['amount_to_balance_ratio'] > 0.8).astype(int)
    df['R4_device_sharing'] = (df['device_shared_accounts'] >= 5).astype(int)
    df['R5_rapid_txn'] = (df['rapid_txn_flag'] == 1).astype(int)
    df['R6_multi_location'] = (df['acct_unique_locations'] >= 8).astype(int)
    df['R7_high_velocity'] = (df['daily_txn_count'] >= 3).astype(int)
    rule_cols = [c for c in df.columns if c.startswith('R') and c[1].isdigit()]
    weights = {'R1_high_login':3,'R2_amount_spike':2.5,'R3_high_ratio':2,'R4_device_sharing':2.5,
               'R5_rapid_txn':1.5,'R6_multi_location':2,'R7_high_velocity':1.5}
    max_w = sum(weights.values())
    df['rule_score'] = sum(df[r]*weights[r] for r in rule_cols) / max_w
    df['rules_triggered'] = df[rule_cols].sum(axis=1)

    # ML models
    features = [
        'TransactionAmount','TransactionDuration','LoginAttempts','AccountBalance',
        'amount_zscore','amount_to_balance_ratio','amount_to_max_ratio',
        'duration_deviation','high_login_flag','amount_outlier_flag',
        'device_shared_accounts','device_shared_flag','ip_shared_accounts','is_new_merchant',
        'acct_txn_count','acct_std_amount','acct_unique_devices','acct_unique_locations',
        'daily_txn_count','rapid_txn_flag','is_weekend','CustomerAge'
    ]
    X = df[features].replace([np.inf,-np.inf], np.nan).fillna(0)
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=features, index=X.index)
    Xp = PCA(n_components=5, random_state=42).fit_transform(Xs)

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    df['iso_anomaly'] = (iso.fit_predict(Xs)==-1).astype(int)
    df['iso_score'] = iso.decision_function(Xs)
    df['db_anomaly'] = (DBSCAN(eps=2.5, min_samples=10, n_jobs=-1).fit_predict(Xp)==-1).astype(int)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, n_jobs=-1)
    df['lof_anomaly'] = (lof.fit_predict(Xs)==-1).astype(int)
    df['lof_score'] = -lof.negative_outlier_factor_

    df['ml_score'] = (0.45*MinMaxScaler().fit_transform(-df[['iso_score']]).flatten() +
                      0.35*MinMaxScaler().fit_transform(df[['lof_score']]).flatten() +
                      0.20*df['db_anomaly'].values)

    df['hybrid_score'] = 0.5*df['ml_score'] + 0.5*df['rule_score']
    df['hybrid_level'] = pd.cut(df['hybrid_score'], bins=[0,0.1,0.25,0.45,1.0],
                                labels=['Low','Medium','High','Critical'], include_lowest=True)
    df['models_flagged'] = df['iso_anomaly'] + df['db_anomaly'] + df['lof_anomaly']

    return df


# ── Load Data ──
df = load_data()

# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════
st.sidebar.title("🏦 BankGuard Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["Overview", "Account Drill-Down", "Rule Engine", "Risk Explorer", "Live Monitor"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")
risk_filter = st.sidebar.multiselect("Risk Level", ['Low','Medium','High','Critical'],
                                      default=['Low','Medium','High','Critical'])
channel_filter = st.sidebar.multiselect("Channel", df['Channel'].unique().tolist(),
                                         default=df['Channel'].unique().tolist())

# Apply filters
mask = df['hybrid_level'].isin(risk_filter) & df['Channel'].isin(channel_filter)
dff = df[mask]

# ══════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════
if page == "Overview":
    st.title("Fraud Detection Dashboard")
    st.caption(f"Analyzing {len(dff):,} transactions from {dff['AccountID'].nunique()} accounts")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{len(dff):,}")
    c2.metric("Accounts", f"{dff['AccountID'].nunique()}")
    c3.metric("Avg Amount", f"${dff['TransactionAmount'].mean():.0f}")
    c4.metric("High+ Risk", f"{(dff['hybrid_level'].isin(['High','Critical'])).sum():,}")
    c5.metric("Rules Triggered", f"{(dff['rules_triggered']>0).sum():,}")

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        risk_counts = dff['hybrid_level'].value_counts().reindex(['Low','Medium','High','Critical']).fillna(0)
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                     color=risk_counts.index,
                     color_discrete_map={'Low':'#22c55e','Medium':'#f59e0b','High':'#ef4444','Critical':'#7f1d1d'},
                     title="Hybrid Risk Distribution")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(dff, x='hybrid_score', nbins=50, title="Hybrid Score Distribution",
                          color_discrete_sequence=['#6b3a2a'])
        fig.add_vline(x=0.25, line_dash="dash", line_color="#f59e0b", annotation_text="Med→High")
        fig.add_vline(x=0.45, line_dash="dash", line_color="#ef4444", annotation_text="High→Crit")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Charts row 2
    col3, col4 = st.columns(2)

    with col3:
        ch_risk = dff.groupby('Channel')['hybrid_score'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(ch_risk, x='Channel', y='hybrid_score', title="Avg Risk by Channel",
                     color='hybrid_score', color_continuous_scale='YlOrRd')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        occ_risk = dff.groupby('CustomerOccupation')['hybrid_score'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(occ_risk, x='CustomerOccupation', y='hybrid_score', title="Avg Risk by Occupation",
                     color='hybrid_score', color_continuous_scale='YlOrRd')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Top risk transactions
    st.subheader("Top 20 Riskiest Transactions")
    top = dff.nlargest(20, 'hybrid_score')[['TransactionID','AccountID','TransactionAmount',
          'Channel','LoginAttempts','rules_triggered','hybrid_score','hybrid_level']].reset_index(drop=True)
    st.dataframe(top, use_container_width=True, height=400)


# ══════════════════════════════════════════════════
# PAGE: ACCOUNT DRILL-DOWN
# ══════════════════════════════════════════════════
elif page == "Account Drill-Down":
    st.title("Account Drill-Down")

    acct_risk = dff.groupby('AccountID').agg(
        mean_risk=('hybrid_score','mean'), max_risk=('hybrid_score','max'),
        txn_count=('TransactionID','count'), total_amount=('TransactionAmount','sum'),
        high_risk=('hybrid_score', lambda x: (x>0.25).sum()),
        devices=('DeviceID','nunique'), locations=('Location','nunique')
    ).sort_values('mean_risk', ascending=False).reset_index()

    selected = st.selectbox("Select Account", acct_risk['AccountID'].tolist(),
                            format_func=lambda x: f"{x} (risk: {acct_risk[acct_risk['AccountID']==x]['mean_risk'].values[0]:.4f})")

    acct_df = dff[dff['AccountID'] == selected].sort_values('TransactionDate')
    acct_info = acct_risk[acct_risk['AccountID'] == selected].iloc[0]

    # Account KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions", f"{acct_info['txn_count']:,.0f}")
    c2.metric("Mean Risk", f"{acct_info['mean_risk']:.4f}")
    c3.metric("Total Amount", f"${acct_info['total_amount']:,.0f}")
    c4.metric("Devices Used", f"{acct_info['devices']:.0f}")
    c5.metric("Locations", f"{acct_info['locations']:.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(acct_df, x='TransactionDate', y='TransactionAmount',
                     title=f"Transaction Amount Over Time — {selected}",
                     color_discrete_sequence=['#6b3a2a'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(acct_df, x='TransactionDate', y='hybrid_score',
                        color='hybrid_level', title=f"Risk Score Over Time — {selected}",
                        color_discrete_map={'Low':'#22c55e','Medium':'#f59e0b','High':'#ef4444','Critical':'#7f1d1d'})
        fig.add_hline(y=0.25, line_dash="dash", line_color="#f59e0b")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Transaction History")
    display_cols = ['TransactionDate','TransactionAmount','TransactionType','Channel',
                   'LoginAttempts','Location','rules_triggered','hybrid_score','hybrid_level']
    st.dataframe(acct_df[display_cols].reset_index(drop=True), use_container_width=True, height=400)


# ══════════════════════════════════════════════════
# PAGE: RULE ENGINE
# ══════════════════════════════════════════════════
elif page == "Rule Engine":
    st.title("Rule-Based Engine Analysis")

    rule_info = {
        'R1_high_login': ('Login ≥ 3', 'Critical', 3.0),
        'R2_amount_spike': ('Amount Z > 2', 'High', 2.5),
        'R3_high_ratio': ('Amt/Bal > 0.8', 'High', 2.0),
        'R4_device_sharing': ('Device ≥ 5 accts', 'Critical', 2.5),
        'R5_rapid_txn': ('Rapid (< 1hr)', 'Medium', 1.5),
        'R6_multi_location': ('8+ cities', 'High', 2.0),
        'R7_high_velocity': ('3+ txn/day', 'Medium', 1.5),
    }

    rule_data = []
    for rn, (desc, sev, w) in rule_info.items():
        n = dff[rn].sum()
        rule_data.append({'Rule': rn, 'Description': desc, 'Severity': sev,
                         'Weight': w, 'Triggered': n, 'Pct': f"{n/len(dff)*100:.2f}%"})
    rule_df = pd.DataFrame(rule_data)
    st.dataframe(rule_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(rule_df.sort_values('Triggered'), x='Triggered', y='Rule',
                     orientation='h', title="Rule Trigger Frequency",
                     color='Severity', color_discrete_map={'Critical':'#ef4444','High':'#f59e0b','Medium':'#3b82f6'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rules_dist = dff['rules_triggered'].value_counts().sort_index().reset_index()
        rules_dist.columns = ['Rules', 'Count']
        fig = px.bar(rules_dist, x='Rules', y='Count', title="Rules per Transaction",
                     color='Rules', color_continuous_scale='YlOrRd')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════
# PAGE: RISK EXPLORER
# ══════════════════════════════════════════════════
elif page == "Risk Explorer":
    st.title("Risk Explorer")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(dff.sample(min(5000, len(dff)), random_state=42),
                        x='TransactionAmount', y='hybrid_score',
                        color='hybrid_level', size='rules_triggered',
                        color_discrete_map={'Low':'#22c55e','Medium':'#f59e0b','High':'#ef4444','Critical':'#7f1d1d'},
                        title="Amount vs Risk Score", opacity=0.5)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(dff.sample(min(5000, len(dff)), random_state=42),
                        x='ml_score', y='rule_score',
                        color='hybrid_level',
                        color_discrete_map={'Low':'#22c55e','Medium':'#f59e0b','High':'#ef4444','Critical':'#7f1d1d'},
                        title="ML Score vs Rule Score", opacity=0.5)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Age distribution by risk
    fig = px.box(dff, x='hybrid_level', y='CustomerAge', title="Age Distribution by Risk Level",
                 color='hybrid_level',
                 color_discrete_map={'Low':'#22c55e','Medium':'#f59e0b','High':'#ef4444','Critical':'#7f1d1d'},
                 category_orders={'hybrid_level': ['Low','Medium','High','Critical']})
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════
# PAGE: LIVE MONITOR (Real-time Simulation)
# ══════════════════════════════════════════════════
elif page == "Live Monitor":
    import time

    st.title("Live Transaction Monitor")
    st.caption("Real-time fraud detection simulation — streaming transactions from the dataset")

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1,1,2])
    speed = col_ctrl1.slider("Speed (txn/sec)", 1, 20, 5)
    batch_size = col_ctrl2.slider("Batch size", 1, 10, 3)
    alert_threshold = col_ctrl3.slider("Alert threshold (hybrid score)", 0.1, 0.5, 0.25, 0.05)

    # Placeholders for live updates
    kpi_placeholder = st.empty()
    col_chart1, col_chart2 = st.columns(2)
    chart1_placeholder = col_chart1.empty()
    chart2_placeholder = col_chart2.empty()
    feed_placeholder = st.empty()

    # Simulation state
    if 'monitor_running' not in st.session_state:
        st.session_state.monitor_running = False
    if 'monitor_idx' not in st.session_state:
        st.session_state.monitor_idx = 0

    col_btn1, col_btn2 = st.columns(2)
    start = col_btn1.button("▶ Start Monitoring", type="primary", use_container_width=True)
    stop = col_btn2.button("⏹ Stop", use_container_width=True)

    if start:
        st.session_state.monitor_running = True
    if stop:
        st.session_state.monitor_running = False

    if st.session_state.monitor_running:
        # Shuffle data for variety
        sim_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        processed = []
        alerts = []
        idx = st.session_state.monitor_idx

        for step in range(50):  # Run 50 steps then stop
            if not st.session_state.monitor_running:
                break

            batch = sim_df.iloc[idx:idx+batch_size]
            idx = (idx + batch_size) % len(sim_df)
            processed.extend(batch.to_dict('records'))

            # Detect alerts in this batch
            batch_alerts = batch[batch['hybrid_score'] >= alert_threshold]
            alerts.extend(batch_alerts.to_dict('records'))

            recent = pd.DataFrame(processed[-200:])  # Keep last 200

            # KPI row
            with kpi_placeholder.container():
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Processed", f"{len(processed):,}")
                k2.metric("Alerts", f"{len(alerts)}", delta=f"+{len(batch_alerts)}" if len(batch_alerts) > 0 else None,
                         delta_color="inverse")
                k3.metric("Alert Rate", f"{len(alerts)/max(len(processed),1)*100:.1f}%")
                avg_risk = recent['hybrid_score'].mean()
                k4.metric("Avg Risk", f"{avg_risk:.4f}")

            # Rolling risk chart
            with chart1_placeholder.container():
                if len(recent) > 5:
                    recent_copy = recent.copy()
                    recent_copy['idx'] = range(len(recent_copy))
                    fig = px.line(recent_copy, x='idx', y='hybrid_score',
                                 title="Rolling Risk Score",
                                 color_discrete_sequence=['#6b3a2a'])
                    fig.add_hline(y=alert_threshold, line_dash="dash", line_color="#ef4444",
                                 annotation_text="Alert Threshold")
                    fig.update_layout(height=280, xaxis_title="Transaction #", yaxis_title="Hybrid Score")
                    st.plotly_chart(fig, use_container_width=True, key=f"risk_{step}")

            # Risk level distribution
            with chart2_placeholder.container():
                if len(recent) > 5:
                    lvl_counts = recent['hybrid_level'].value_counts().reindex(['Low','Medium','High','Critical']).fillna(0)
                    fig = px.bar(x=lvl_counts.index, y=lvl_counts.values,
                                color=lvl_counts.index,
                                color_discrete_map={'Low':'#22c55e','Medium':'#f59e0b','High':'#ef4444','Critical':'#7f1d1d'},
                                title="Risk Level Distribution (Last 200)")
                    fig.update_layout(height=280, showlegend=False, xaxis_title="Risk Level", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True, key=f"dist_{step}")

            # Alert feed
            with feed_placeholder.container():
                if alerts:
                    st.subheader(f"Alert Feed ({len(alerts)} alerts)")
                    alert_df = pd.DataFrame(alerts[-15:])[::-1]  # Last 15, newest first
                    display = alert_df[['TransactionID','AccountID','TransactionAmount',
                                       'Channel','LoginAttempts','hybrid_score','hybrid_level']].reset_index(drop=True)
                    st.dataframe(display, use_container_width=True, height=300)

            time.sleep(1.0 / speed)

        st.session_state.monitor_idx = idx
        st.session_state.monitor_running = False
        st.info("Simulation completed (50 batches). Click **Start Monitoring** to continue.")

    else:
        st.info("Click **Start Monitoring** to begin the real-time fraud detection simulation.")


# Footer
st.sidebar.markdown("---")
st.sidebar.caption("BankGuard Analytics v2.0\nFraud Detection Dashboard")
